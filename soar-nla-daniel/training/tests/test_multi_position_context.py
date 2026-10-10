import numpy as np
import pyarrow as pa
import torch

from nla.injection import (
    fixed_norm_replace_in_residual,
    karvonen_inject_in_residual,
    marker_well_formed,
)
from nla.schema import (
    ACTIVATION_CONTEXT_COLUMN,
    activation_targets_and_context,
    expand_actor_injection_sites,
)
from nla.utils.vllm_steer import find_marker_positions


INJ, LEFT, RIGHT = 42, 41, 43


def _row_with_markers(count: int, tail: int = 9) -> list[int]:
    row = [1]
    for _ in range(count):
        row.extend([LEFT, INJ, RIGHT])
    row.append(tail)
    return row


def test_three_position_injection_is_ordered_per_row():
    ids = torch.tensor([
        _row_with_markers(3, 9),
        _row_with_markers(3, 8),
    ])
    d = 5
    residual = torch.ones(2, ids.shape[1], d)
    context = torch.arange(2 * 3 * d, dtype=torch.float32).reshape(2, 3, d) + 1

    out = karvonen_inject_in_residual(
        ids, residual, context, INJ, LEFT, RIGHT,
    )
    marker_positions = [2, 5, 8]
    for b in range(2):
        for j, position in enumerate(marker_positions):
            h = residual[b, position]
            v = context[b, j]
            expected = h + h.norm() * v / (v.norm() + 1e-9)
            assert torch.allclose(out[b, position], expected)

    changed = (out != residual).any(dim=-1)
    assert changed.nonzero().tolist() == [
        [0, 2], [0, 5], [0, 8], [1, 2], [1, 5], [1, 8],
    ]


def test_context_marker_count_is_checked_per_row():
    # Total sites still equals B*K (=6), but the rows are 2 and 4. A global
    # count check would silently shift row 0's last vector into row 1.
    rows = [_row_with_markers(2), _row_with_markers(4)]
    width = max(map(len, rows))
    ids = torch.tensor([row + [0] * (width - len(row)) for row in rows])
    try:
        karvonen_inject_in_residual(
            ids, torch.randn(2, width, 4), torch.randn(2, 3, 4),
            INJ, LEFT, RIGHT,
        )
    except RuntimeError as exc:
        assert "row 0" in str(exc)
    else:
        raise AssertionError("per-row context marker mismatch was accepted")


def test_fixed_norm_replacement_supports_context():
    ids = torch.tensor([_row_with_markers(3)])
    context = torch.randn(1, 3, 7)
    out = fixed_norm_replace_in_residual(
        ids, torch.randn(1, ids.shape[1], 7), context,
        INJ, LEFT, RIGHT, target_norm=11.0,
    )
    assert torch.allclose(
        out[0, [2, 5, 8]].float().norm(dim=-1),
        torch.full((3,), 11.0),
        atol=1e-5,
    )


def test_plural_marker_checks_agree():
    row = _row_with_markers(3)
    assert marker_well_formed(row, INJ, LEFT, RIGHT, expected_count=3)
    assert find_marker_positions(
        row, INJ, LEFT, RIGHT, expected_count=3,
    ) == [2, 5, 8]
    assert not marker_well_formed(row, INJ, LEFT, RIGHT)


def test_actor_template_expands_oldest_to_newest_sites():
    template = "before <concept>{injection_char}</concept> after"
    rendered = expand_actor_injection_sites(template, "<INJECT>", 3)
    assert rendered == (
        "before <concept><INJECT></concept>\n"
        "<concept><INJECT></concept>\n"
        "<concept><INJECT></concept> after"
    )


def test_arrow_context_keeps_last_position_target_separate():
    n, k, d = 2, 3, 4
    context = np.arange(n * k * d, dtype=np.float32).reshape(n, k, d)
    targets = context[:, -1].copy()
    target_col = pa.FixedSizeListArray.from_arrays(
        pa.array(targets.reshape(-1)), d,
    )
    inner = pa.FixedSizeListArray.from_arrays(
        pa.array(context.reshape(-1)), d,
    )
    context_col = pa.FixedSizeListArray.from_arrays(inner, k)
    batch = pa.record_batch(
        [target_col, context_col],
        names=["activation_vector", ACTIVATION_CONTEXT_COLUMN],
    )

    got_targets, got_context = activation_targets_and_context(batch)
    np.testing.assert_array_equal(got_targets, targets)
    np.testing.assert_array_equal(got_context, context)

    legacy = pa.record_batch([target_col], names=["activation_vector"])
    legacy_targets, legacy_context = activation_targets_and_context(legacy)
    np.testing.assert_array_equal(legacy_targets, targets)
    np.testing.assert_array_equal(legacy_context, targets[:, None, :])


def test_datagen_schemas_keep_context_actor_only():
    from nla.datagen.stage0_extract import _schema
    from nla.datagen.stage3_build import _schema_for

    base = _schema(d_model=4, context_size=3)
    assert base.field(ACTIVATION_CONTEXT_COLUMN).type.list_size == 3
    assert base.field(ACTIVATION_CONTEXT_COLUMN).type.value_type.list_size == 4

    av = _schema_for("av_sft", False, 4, 3, True)
    rl = _schema_for("rl", False, 4, 3, True)
    ar = _schema_for("ar_sft", False, 4, 3, True)
    assert ACTIVATION_CONTEXT_COLUMN in av.names
    assert ACTIVATION_CONTEXT_COLUMN in rl.names
    assert ACTIVATION_CONTEXT_COLUMN not in ar.names


def test_stage0_never_samples_before_full_context_exists():
    from nla.datagen.stage0_extract import _sample_positions

    positions = _sample_positions(
        list(range(80)), n_positions=80, special_ids=set(), doc_id="doc",
        seed=0, context_size=55,
    )
    assert positions
    assert min(positions) >= 54


def test_nested_context_shuffle_preserves_rows():
    from nla.datagen.stage_shuffle import _take_fixed_size_list_via_numpy

    n, k, d = 3, 3, 2
    context = np.arange(n * k * d, dtype=np.float32).reshape(n, k, d)
    inner = pa.FixedSizeListArray.from_arrays(pa.array(context.reshape(-1)), d)
    outer = pa.FixedSizeListArray.from_arrays(inner, k)
    shuffled = _take_fixed_size_list_via_numpy(
        pa.chunked_array([outer]), np.asarray([2, 0, 1], dtype=np.int64),
    )
    got = activation_targets_and_context(
        pa.record_batch([
            pa.FixedSizeListArray.from_arrays(
                pa.array(context[[2, 0, 1], -1].reshape(-1)), d,
            ),
            shuffled,
        ], names=["activation_vector", ACTIVATION_CONTEXT_COLUMN])
    )[1]
    np.testing.assert_array_equal(got, context[[2, 0, 1]])


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
