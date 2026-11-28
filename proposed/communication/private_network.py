"""Build the veth link inside the private namespaces opened by the Slurm script."""

import os
import subprocess

from . import config, launcher


def _run(*command):
    subprocess.run(command, check=True)


def run():
    headroom = os.environ.get("KV_COMM_EXPERIMENT") == "headroom"
    rate_args = [] if headroom or config.NETEM_RATE is None else ["rate", config.NETEM_RATE]
    endpoints = (
        (config.ES_NAMESPACE, "ves", config.ES_IP),
        (config.UD_NAMESPACE, "vud", config.UD_IP),
    )

    _run("mount", "-t", "tmpfs", "tmpfs", "/run")
    _run("mkdir", "-p", "/run/netns")
    for namespace, _, _ in endpoints:
        _run("ip", "netns", "add", namespace)
    _run("ip", "link", "add", "ves", "type", "veth", "peer", "name", "vud")

    for namespace, device, address in endpoints:
        _run("ip", "link", "set", device, "netns", namespace)
        _run(
            "ip", "-n", namespace, "addr", "add",
            f"{address}/{config.VETH_PREFIX_LENGTH}", "dev", device,
        )
        _run("ip", "-n", namespace, "link", "set", device, "up")
        _run("ip", "-n", namespace, "link", "set", "lo", "up")
        _run(
            "ip", "netns", "exec", namespace, "tc", "qdisc", "replace",
            "dev", device, "root", "netem", "delay", config.NETEM_DELAY,
            *rate_args,
        )

    if not headroom:
        launcher.main()
        return

    config.MODE = "bench"
    config.SOURCE_PLACEMENT = "es"
    config.WIRE_CODEC = "bitpack5"
    config.SAMPLES = 8
    config.REPEATS = 3
    config.MEASURE_CACHE_MISS = True
    profile_stages = os.environ.get("KV_COMM_PROFILE_STAGES", "1") != "0"

    for repair in (0.30, 0.15, 0.10, 0.0):
        config.REPAIR_FRACTION = repair
        print(f"Repair sweep: {repair:.2f}, netem delay={config.NETEM_DELAY}, no rate cap",
              flush=True)
        launcher.main(profile_stages=profile_stages, warmup_each_question=True)
