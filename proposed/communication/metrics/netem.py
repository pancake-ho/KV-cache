"""Record routing and qdisc counters in the current network namespace."""
import argparse
import json
from pathlib import Path
import subprocess
from tempfile import NamedTemporaryFile


def read_qdiscs(output, phase):
    snapshots = {}
    for side in ("es", "ud"):
        path = output / f"{side}.qdisc.{phase}.json"
        if path.is_file():
            snapshots[side] = json.loads(path.read_text(encoding="utf-8"))
    return snapshots


def _root_bytes(qdiscs):
    for qdisc in qdiscs:
        if qdisc.get("kind") == "netem" and qdisc.get("root"):
            return qdisc.get("bytes")
    return None


def finalize_network_report(output):
    before = read_qdiscs(output, "before")
    after = read_qdiscs(output, "after")
    delta = {}
    for side in before.keys() & after.keys():
        start, end = _root_bytes(before[side]), _root_bytes(after[side])
        delta[side] = end - start if start is not None and end is not None and end >= start else None

    report = output / "report.jsonl"
    rows = report.read_text(encoding="utf-8").splitlines()
    summary = json.loads(rows[-1])
    if summary.get("kind") != "summary":
        raise RuntimeError("Expected completed benchmark summary before adding qdisc counters")
    summary.update(netem_after=after, netem_egress_bytes_delta=delta)
    rows[-1] = json.dumps(summary, ensure_ascii=False)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=output, delete=False) as temporary:
        path = Path(temporary.name)
        try:
            temporary.write("\n".join(rows) + "\n")
            temporary.close()
            path.chmod(report.stat().st_mode & 0o777)
            path.replace(report)
        finally:
            path.unlink(missing_ok=True)


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
