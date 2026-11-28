"""Build the veth link inside the private namespaces opened by the Slurm script."""

import subprocess

from . import config, launcher


def _run(*command):
    subprocess.run(command, check=True)


def run():
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
            "rate", config.NETEM_RATE,
        )

    launcher.main()
