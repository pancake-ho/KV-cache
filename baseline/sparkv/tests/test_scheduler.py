from baseline.sparkv.scheduler import Chunk, SparKVScheduler


def make_scheduler(t=2, layers=3, heads=2, delta=2.0):
    chunks = [
        Chunk(i, layer, head)
        for i in range(t)
        for layer in range(layers)
        for head in range(heads)
    ]
    comp = {c: 1.0 for c in chunks}
    stream = {c: 1.2 for c in chunks}
    return SparKVScheduler(t, layers, heads, comp, stream, delta)


def test_each_chunk_is_processed_exactly_once():
    scheduler = make_scheduler()
    result = scheduler.run()
    scheduled = []
    for stage in result["stages"]:
        scheduled.extend(("compute", tuple(x.values())) for x in stage["compute"])
        scheduled.extend(("stream", tuple(x.values())) for x in stage["stream"])
    assert len(scheduled) == result["chunks"]
    assert len({chunk for _, chunk in scheduled}) == result["chunks"]


def test_vertical_dependency_requires_local_compute():
    scheduler = make_scheduler(t=1, layers=2, heads=1)
    parent = Chunk(0, 0, 0)
    child = Chunk(0, 1, 0)
    scheduler.done[parent] = "stream"
    assert not scheduler.compute_ready(child)
    scheduler.done[parent] = "compute"
    assert scheduler.compute_ready(child)


def test_token_dependency_accepts_either_path():
    scheduler = make_scheduler(t=2, layers=2, heads=1)
    current = Chunk(1, 0, 0)
    previous = Chunk(0, 0, 0)
    scheduler.done[previous] = "stream"
    assert scheduler.compute_ready(current)

