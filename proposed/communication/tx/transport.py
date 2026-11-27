"""Connect, frame and serialize data at the sending endpoint."""
from dataclasses import asdict
import json
import socket
from time import perf_counter, sleep

from ..protocol import HEADER, NAMESPACE, namespace_id


def send_handoff(connection, rendezvous):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as control:
        control.settimeout(30)
        control.connect(rendezvous)
        socket.send_fds(control, [NAMESPACE.pack(namespace_id())], [connection.fileno()])


def connect_to_es(es_ip, ud_ip, port):
    deadline = perf_counter() + 15
    while True:
        try:
            return socket.create_connection(
                (es_ip, port), timeout=2, source_address=(ud_ip, 0),
            )
        except ConnectionRefusedError:
            if perf_counter() >= deadline:
                raise
            sleep(0.1)


def send_frame(connection, chunks, size):
    connection.sendall(HEADER.pack(size))
    for chunk in chunks:
        connection.sendall(chunk)


def encode_request(request):
    return json.dumps(asdict(request)).encode()


def encode_kv(layers, codec, positions):
    import torch

    packed = codec.encode(layers) if codec else None
    pairs = [(layer.payload, layer.scales) for layer in packed] if packed else layers
    tensors = [tensor for pair in pairs for tensor in pair]
    chunks = [
        tensor.detach().to("cpu").contiguous().view(torch.uint8).numpy()
        for tensor in tensors
    ]
    if positions is not None:
        chunks.append(json.dumps(positions).encode())
    tensor_layout = [(t.shape, t.dtype, t.device) for t in tensors]
    packed_layout = [(type(layer), layer.shape, layer.dtype) for layer in packed or ()]
    return chunks, tensor_layout, packed_layout
