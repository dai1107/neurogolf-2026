from __future__ import annotations

from src.task255_override_shape_prune import _make_max_chain, conservative_keep_shapes


def test_conservative_keep_shapes_keeps_corridor_family() -> None:
    keep = conservative_keep_shapes(6, 12)

    assert (6, 26) in keep
    assert (30, 12) in keep
    assert (5, 26) not in keep
    assert (13, 30) not in keep
    assert len(keep) == 28


def test_conservative_keep_shapes_accepts_single_long_side() -> None:
    keep = conservative_keep_shapes(6, 6, long_sides=(26,))

    assert keep == {(6, 26), (26, 6)}


def test_make_max_chain_preserves_requested_final_output() -> None:
    nodes = _make_max_chain(["a", "b", "c"], "final", "acc")

    assert [node.op_type for node in nodes] == ["Max", "Max"]
    assert nodes[-1].output[0] == "final"
