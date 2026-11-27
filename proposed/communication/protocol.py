"""Shared frame and namespace identifiers for the same-node TCP experiment."""
import os
from struct import Struct

NAMESPACE = Struct("!Q")
HEADER = Struct("!Q")
MAX_BYTES = 1 << 30


def namespace_id():
    return os.stat("/proc/self/ns/net").st_ino
