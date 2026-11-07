from xmodel_kv.diverse_training import select_eligible_rows


def test_select_eligible_rows_preserves_manifest_order_and_limits():
    manifest = [
        {"id": "a", "split": "train"},
        {"id": "b", "split": "train"},
        {"id": "c", "split": "train"},
    ]
    screens = {
        key: {
            "id": key,
            "eligible": key != "b",
            "direct_reuse": {
                "target": {"compatible_em": False},
                "source": {"compatible_em": True},
            },
        }
        for key in ("a", "b", "c")
    }
    selected, audit = select_eligible_rows(
        manifest, screens, limits={"train": 2}
    )
    assert [row["id"] for row in selected["train"]] == ["a", "c"]
    assert audit["direct_reuse"]["train"]["target_accuracy"] == 0
    assert audit["direct_reuse"]["train"]["source_accuracy"] == 1
