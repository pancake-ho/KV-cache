"""Settings for the single-node UD–ES TCP experiment."""

ES_NAMESPACE = "kv-es"
UD_NAMESPACE = "kv-ud"
ES_IP = "10.203.0.1"
UD_IP = "10.203.0.2"
VETH_PREFIX_LENGTH = 30
NETEM_DELAY = "2ms"
NETEM_RATE = "100mbit"
NAMESPACE_COMMAND = ("ip", "netns", "exec")
PORT = 38197
SOCKET_TIMEOUT = 300

MODE = "bench"  # Change to "bench" after the network smoke passes.
SMOKE_MIB = 4

PAIR = "1.7b-4b"
QA_JOB = "222976"
QA_REPORT = None
LORA_ROUTE = "medqa"
REPAIR_FRACTION = 0.30
SOURCE_PLACEMENT = "ud"
WIRE_CODEC = "raw"
SAMPLES = 2
REPEATS = 1
NEW_TOKENS = 8
MEASURE_CACHE_MISS = True
