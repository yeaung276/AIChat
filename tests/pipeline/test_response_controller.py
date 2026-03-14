"""
Unit tests for aichat.pipeline.response_controller.LatencyController
"""
import pytest
from aichat.pipeline.response_controller import LatencyController


def _make_profiled(stt_time: float, tts_time: float) -> list:
    """Build a minimal profiled list with stt_out and tts_out timestamps."""
    return [
        {"component": "stt_out", "time": stt_time},
        {"component": "llm_out", "time": (stt_time + tts_time) / 2},
        {"component": "tts_out", "time": tts_time},
    ]


def _drive(lc: LatencyController, latency_ms: float, n: int) -> str:
    """Call update n times at the given latency (ms) and return the final mode."""
    result = lc.mode
    for _ in range(n):
        result = lc.update(_make_profiled(0.0, latency_ms / 1000.0))
    return result

class TestInitialState:
    def test_mode_is_medium(self):
        assert LatencyController().mode == "medium"

    def test_ewma_is_2000(self):
        assert LatencyController().ewma == 2000

    def test_turn_is_zero(self):
        assert LatencyController().turn == 0

    def test_summary_with_no_turns(self):
        lc = LatencyController()
        s = lc.summary
        assert s["turns"] == 0
        assert s["mean_latency_ms"] == 0.0
        assert s["max_latency_ms"] == 0.0
        assert s["min_latency_ms"] == float("inf")

class TestUpdateLatencyStats:
    def test_latency_ms_computed_from_timestamps(self):
        lc = LatencyController()
        latency = lc._update_latency_stats(_make_profiled(1.0, 3.5))
        assert latency == pytest.approx(2500.0)

    def test_min_max_tracked_across_calls(self):
        lc = LatencyController()
        lc._update_latency_stats(_make_profiled(0.0, 2.0))   # 2000 ms
        lc._update_latency_stats(_make_profiled(0.0, 0.5))   # 500 ms
        lc._update_latency_stats(_make_profiled(0.0, 4.0))   # 4000 ms
        assert lc.min_latency == pytest.approx(500.0)
        assert lc.max_latency == pytest.approx(4000.0)

    def test_latency_sum_accumulates(self):
        lc = LatencyController()
        lc._update_latency_stats(_make_profiled(0.0, 1.0))   # 1000 ms
        lc._update_latency_stats(_make_profiled(0.0, 2.0))   # 2000 ms
        assert lc.latency_sum == pytest.approx(3000.0)


class TestWarmup:
    def test_high_latency_does_not_change_mode_during_warmup(self):
        lc = LatencyController()
        # 5000 ms would target "short" — but warmup should block it
        for _ in range(4):
            assert lc.update(_make_profiled(0.0, 5.0)) == "medium"

    def test_mode_changes_on_turn_5_after_warmup(self):
        lc = LatencyController()
        _drive(lc, 5000.0, 4)          # warmup
        mode = lc.update(_make_profiled(0.0, 5.0))  # turn 5
        assert mode == "short"


class TestModeTransitions:
    def test_downgrade_to_short_after_single_high_latency_sample(self):
        lc = LatencyController()
        _drive(lc, 5000.0, 4)          # warmup (mode stays "medium")
        mode = _drive(lc, 5000.0, 1)   # turn 5, downgrade threshold = 1
        assert mode == "short"

    def test_upgrade_to_long_requires_four_stable_low_latency_samples(self):
        lc = LatencyController()
        lc.mode = "short"
        lc.pending = "short"
        lc.stable_count = 0
        lc.turn = 10       # past warmup
        lc.ewma = 500.0    # pre-seeded so signal stays in "long" territory

        # 3 low-latency signals — not yet enough
        for _ in range(3):
            lc._update_response_mode(500.0)
        assert lc.mode == "short"

        # 4th triggers the upgrade
        lc._update_response_mode(500.0)
        assert lc.mode == "long"

    def test_pending_resets_stable_count_when_target_changes(self):
        lc = LatencyController()
        _drive(lc, 5000.0, 5)  # mode = "short"

        # Mix of targets — stable_count should reset each time target changes
        lc.update(_make_profiled(0.0, 0.5))  # target "long", stable_count = 1
        lc.update(_make_profiled(0.0, 2.0))  # target "medium", stable_count resets to 1
        lc.update(_make_profiled(0.0, 0.5))  # target "long" again, stable_count resets to 1

        # Only 1 stable observation for "long" — not enough to upgrade
        assert lc.mode == "short"

    def test_summary_turn_count_matches_update_calls(self):
        lc = LatencyController()
        _drive(lc, 2000.0, 7)
        assert lc.summary["turns"] == 7

    def test_summary_mean_latency_correct(self):
        lc = LatencyController()
        _drive(lc, 1000.0, 4)
        assert lc.summary["mean_latency_ms"] == pytest.approx(1000.0)