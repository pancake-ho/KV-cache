# Llama-3.1 half-depth state localization diagnostic

Status: frozen after the packet-aware development gate failed and before any
first-half result was run.

## Trigger

The all/last-16 factorial showed that N=512 has a strong all-layer state but a
weak last-half packet.  Always-masked packet-aware training improved last-16
INT4 dev F1 only from `.2656` to `.3146`, below its preregistered `+.10` gate,
and is therefore not evaluated on a new confirmation set.  It also produced
zero all-layer stage F1 because early-layer K/V never seen by the loss became
actively disruptive when restored.

The existing experiment cannot distinguish whether the strong N=512 state is
primarily available in the first half or requires both halves jointly.  That
distinction changes the next method: first-half success motivates routing the
packet to early layers, while failure of both halves motivates cross-depth
coding/reconstruction or simply retaining the 135 KB all-layer packet.

## Frozen diagnostic

On the already inspected 256-case test, evaluate the final N=128 and N=512
unconstrained checkpoints with `first_16` layers under both BF16 and INT4.
Run exactly these four cells and compare them with the already completed
`last_16` and `all` cells.  Report paired F1 and exact statistics; do not scan
any other first-K, last-K, interleaved, or learned masks.

This is a post-result mechanism diagnostic, not a new confirmatory result and
cannot replace the failed preregistered last-16 operating point.  If a
first-half packet is selected for a future method, its quality must be
confirmed without tuning on a newly frozen, training-disjoint set.
