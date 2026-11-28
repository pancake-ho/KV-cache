"""Run the configured experiment in the existing network namespaces."""

import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory, mkdtemp

from . import config
from .metrics.netem import finalize_network_report


def namespace_command(namespace, module, *args):
    return [
        *config.NAMESPACE_COMMAND, namespace, sys.executable, "-u", "-m",
        f"proposed.communication.{module}", *(str(arg) for arg in args),
    ]


def benchmark_args(*, profile_stages=False, warmup_each_question=False, native_control=False):
    if config.MODE == "smoke":
        return ["--smoke-mib", config.SMOKE_MIB]

    return [
        "--pair", config.PAIR, "--qa-job", config.QA_JOB,
        "--repair", config.REPAIR_FRACTION, "--source-placement", config.SOURCE_PLACEMENT,
        "--wire-codec", config.WIRE_CODEC, "--samples", config.SAMPLES,
        "--repeats", config.REPEATS, "--new-tokens", config.NEW_TOKENS,
        *(["--qa-report", config.QA_REPORT] if config.QA_REPORT else []),
        *(["--lora-route", config.LORA_ROUTE] if config.LORA_ROUTE else []),
        *(["--measure-cache-miss"] if config.MEASURE_CACHE_MISS else []),
        *(["--profile-stages"] if profile_stages else []),
        *(["--warmup-each-question"] if warmup_each_question else []),
        *(["--native-control"] if native_control else []),
    ]


def snapshot_network(output, phase):
    for side, namespace, local_ip, peer_ip in (
        ("es", config.ES_NAMESPACE, config.ES_IP, config.UD_IP),
        ("ud", config.UD_NAMESPACE, config.UD_IP, config.ES_IP),
    ):
        command = namespace_command(
            namespace, "metrics.netem", local_ip, peer_ip, output / side, phase,
        )
        subprocess.run(command, check=True)


def main(*, profile_stages=False, warmup_each_question=False, native_control=False):
    if config.MODE not in {"smoke", "bench"}:
        raise SystemExit("Set MODE to 'smoke' or 'bench' in proposed/communication/config.py")
    if not all((config.ES_NAMESPACE, config.UD_NAMESPACE, config.ES_IP, config.UD_IP)):
        raise SystemExit("Set both namespace names and veth IPs in proposed/communication/config.py")

    output_root = Path("results/e2e").resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    job = os.environ["SLURM_JOB_ID"]
    output = Path(mkdtemp(prefix=f"ud_es_{job}_", dir=output_root))
    print(f"Results {output}", flush=True)
    snapshot_network(output, "before")

    with TemporaryDirectory(prefix="ud-es.", dir="/tmp") as scratch:
        common = [
            "--es-ip", config.ES_IP, "--ud-ip", config.UD_IP,
            "--port", config.PORT, "--rendezvous", Path(scratch) / "es.sock",
            "--socket-timeout", config.SOCKET_TIMEOUT,
        ]
        server_command = namespace_command(config.ES_NAMESPACE, "benchmark", "es", *common)
        client_command = namespace_command(
            config.UD_NAMESPACE, "benchmark", config.MODE, *common,
            *benchmark_args(profile_stages=profile_stages, warmup_each_question=warmup_each_question,
                            native_control=native_control),
            "--out", output / "report.jsonl",
        )
        server = subprocess.Popen(server_command)
        try:
            subprocess.run(client_command, check=True)
            returncode = server.wait(timeout=10)
            if returncode:
                raise subprocess.CalledProcessError(returncode, server_command)
        finally:
            if server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()

    snapshot_network(output, "after")
    if config.MODE == "bench":
        finalize_network_report(output)


if __name__ == "__main__":
    main()
