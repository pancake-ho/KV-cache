"""Record routing and qdisc counters in the current network namespace."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("local_ip")
    parser.add_argument("peer_ip")
    parser.add_argument("dest")
    parser.add_argument("phase", choices=("before", "after"))
    args = parser.parse_args()

    route_json = subprocess.check_output(
        ["ip", "-j", "route", "get", args.peer_ip, "from", args.local_ip], text=True,
    )
    route, = json.loads(route_json)
    interface = route["dev"]
    Path(f"{args.dest}.route.{args.phase}.json").write_text(route_json, encoding="utf-8")
    if interface == "lo":
        raise SystemExit("Expected the veth route; got loopback")
    qdisc_json = subprocess.check_output(
        ["tc", "-j", "-s", "qdisc", "show", "dev", interface], text=True,
    )
    Path(f"{args.dest}.qdisc.{args.phase}.json").write_text(qdisc_json, encoding="utf-8")


if __name__ == "__main__":
    main()
