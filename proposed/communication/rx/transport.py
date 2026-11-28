"""Accept, validate and reconstruct data at the receiving endpoint."""
import json
import os
import socket
from time import perf_counter

from ..protocol import HEADER, NAMESPACE
from ..tx.transport import send_handoff


def accept_es(es_ip, ud_ip, port, rendezvous):
    """Open the ES side in its namespace and hand off the connected descriptor."""
    with socket.create_server((es_ip, port)) as listener:
        listener.settimeout(30)
        connection, peer = listener.accept()
        with connection:
            if peer[0] != ud_ip:
                raise RuntimeError(f"Expected UD at {ud_ip}, got {peer[0]}")
            send_handoff(connection, rendezvous)


def read_into(connection, buffer):
    remaining = memoryview(buffer)
    while remaining:
        size = connection.recv_into(remaining)
        if not size:
            raise EOFError("TCP peer disconnected while receiving a frame")
        remaining = remaining[size:]


def receive_handoff(control):
    control.settimeout(30)
    message, descriptors, flags, _ = socket.recv_fds(control, NAMESPACE.size, 1)
    if len(descriptors) != 1 or not message or flags & (socket.MSG_CTRUNC | socket.MSG_TRUNC):
        for fd in descriptors:
            os.close(fd)
        raise RuntimeError("Missing or truncated ES socket handoff")
    sock = socket.socket(fileno=descriptors[0])
    try:
        tail = bytearray(NAMESPACE.size - len(message))
        read_into(control, tail)
        peer_namespace, = NAMESPACE.unpack(message + tail)
        return sock, peer_namespace
    except BaseException:
        sock.close()
        raise


def receive_frame(connection, buffers, size):
    header = bytearray(HEADER.size)
    read_into(connection, header)
    if HEADER.unpack(header)[0] != size:
        raise RuntimeError("TCP payload length changed")
    for buffer in buffers:
        read_into(connection, buffer)


def decode_request(packet, request_type):
    data = json.loads(packet)
    data["token_ids"] = tuple(data["token_ids"])
    return request_type(**data)


def decode_kv(received, tensor_layout, packed_layout, codec, has_positions, *, timings=None):
    import torch

    frames = received[:len(tensor_layout)]
    if timings is None:
        restored = [
            torch.frombuffer(data, dtype=dtype).reshape(shape).to(device)
            for data, (shape, dtype, device) in zip(frames, tensor_layout, strict=True)
        ]
    else:
        started = perf_counter()
        host = [
            torch.frombuffer(data, dtype=dtype).reshape(shape)
            for data, (shape, dtype, _) in zip(frames, tensor_layout, strict=True)
        ]
        received_positions = json.loads(received[-1]) if has_positions else None
        timings["deserialize_ms"] = (perf_counter() - started) * 1000
        started = perf_counter()
        restored = [tensor.to(device) for tensor, (_, _, device) in zip(host, tensor_layout, strict=True)]
        timings["htod_ms"] = (perf_counter() - started) * 1000
    layers = tuple(zip(restored[::2], restored[1::2], strict=True))
    started = perf_counter()
    if packed_layout:
        layers = codec.decode(tuple(
            layer_type(payload, scales, shape, dtype)
            for (layer_type, shape, dtype), (payload, scales) in zip(packed_layout, layers, strict=True)
        ))
    if timings is None:
        received_positions = json.loads(received[-1]) if has_positions else None
    else:
        if packed_layout:
            torch.cuda.synchronize(layers[0][0].device)
        timings["decode_ms"] = (perf_counter() - started) * 1000
    return layers, received_positions
