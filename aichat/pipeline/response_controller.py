import time
from collections import deque


class LatencyController:
    def __init__(self):
        self.ewma = 2000
        self.window = deque(maxlen=15)
        self.mode = "medium"
        self.pending = "medium"
        self.stable_count = 0
        self.turn = 0

        self.latency_sum = 0.0
        self.max_latency = 0.0
        self.min_latency = float("inf")
        self.session_start = time.perf_counter()
        
    def _update_latency_stats(self, profiled: list):
        stt_time = next(p["time"] for p in profiled if p["component"] == "stt_out")
        tts_time = next(p["time"] for p in profiled if p["component"] == "tts_out")
        latency_ms = (tts_time - stt_time) * 1000
        
        self.latency_sum += latency_ms
        self.max_latency = max(self.max_latency, latency_ms)
        self.min_latency = min(self.min_latency, latency_ms)
        return latency_ms
        
    def _update_response_mode(self, latency_ms: float):
        self.ewma = self.ewma * 0.85 + latency_ms * 0.15
        self.window.append(latency_ms)

        p80 = sorted(self.window)[int(len(self.window) * 0.8)]
        signal = max(self.ewma, p80)

        if signal < 1600:
            target = "long"
        elif signal < 2400:
            target = "medium"
        else:
            target = "short"

        # warmup
        if self.turn <= 4:
            return self.mode

        # hysteresis
        order = ["long", "medium", "short"]
        is_downgrade = order.index(target) > order.index(self.mode)
        threshold = 1 if is_downgrade else 4

        if target == self.pending:
            self.stable_count += 1
        else:
            self.pending = target
            self.stable_count = 1

        if self.stable_count >= threshold:
            self.mode = target
            self.stable_count = 0

        return self.mode

    def update(self, profiled: list):
        self.turn += 1
        latency = self._update_latency_stats(profiled)
        return self._update_response_mode(latency)


    @property
    def summary(self):
        return {
            "mean_latency_ms": self.latency_sum / self.turn if self.turn else 0.0,
            "max_latency_ms": self.max_latency,
            "min_latency_ms": self.min_latency,
            "session_duration_s": time.perf_counter() - self.session_start,
            "turns": self.turn,
        }