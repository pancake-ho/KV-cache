"""Orchestrate the transmit and receive endpoints and the shared TCP link."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, ExitStack
from ipaddress import IPv4Address
import socket
from time import perf_counter

from .metrics.record import frame_metrics
from .protocol import HEADER, MAX_BYTES, namespace_id
from .rx.transport import receive_frame, receive_handoff
from .tx.transport import connect_to_es, send_frame


@contextmanager
def connect_ud(es_ip, ud_ip, port, rendezvous, timeout):
    """Open the UD side and verify the peer and both namespaces before timing."""
    if any(IPv4Address(ip).is_loopback for ip in (es_ip, ud_ip)):
        raise ValueError("Use the established veth IPv4 endpoints")
    with ExitStack() as stack:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(rendezvous)
            listener.listen(1)
            listener.settimeout(30)
            ud = stack.enter_context(connect_to_es(es_ip, ud_ip, port))
            control, _ = listener.accept()
            with control:
                es, es_namespace = receive_handoff(control)
                stack.enter_context(es)
        ud_namespace = namespace_id()
        if es_namespace == ud_namespace:
            raise RuntimeError("Both TCP sockets were opened in the same network namespace")
        if es.getpeername() != ud.getsockname() or es.getsockname() != ud.getpeername():
            raise RuntimeError("Handed-off socket is not the connected ES peer")
        for connection in (ud, es):
            connection.settimeout(timeout)
            connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        executor = stack.enter_context(ThreadPoolExecutor(max_workers=1))
        yield Link(ud, es, executor, ud_namespace, es_namespace)


class Link:
    """Serial, length-checked TCP exchanges with a concurrent receiver."""

    def __init__(self, ud, es, executor, ud_namespace, es_namespace):
        self.ud, self.es = ud, es
        self.executor = executor
        self.metadata = {
            "ud": ud.getsockname(), "es": es.getsockname(),
            "ud_netns_inode": ud_namespace, "es_netns_inode": es_namespace,
        }

    def exchange(self, chunks, *, from_ud):
        sender, receiver = (self.ud, self.es) if from_ud else (self.es, self.ud)
        chunks = tuple(memoryview(chunk).cast("B") for chunk in chunks)
        size = sum(map(len, chunks))
        if not 0 < size <= MAX_BYTES:
            raise ValueError("Invalid TCP frame payload size")
        received = [bytearray(len(chunk)) for chunk in chunks]

        started = perf_counter()
        pending = self.executor.submit(send_frame, sender, chunks, size)
        try:
            receive_frame(receiver, received, size)
            pending.result()
        except BaseException:
            for connection in (sender, receiver):
                try:
                    connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            raise
        return received, frame_metrics(started, size, HEADER.size)
