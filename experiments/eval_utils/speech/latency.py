import numpy as np

def evaluate_latency(latencies):
    latencies = [l for l in latencies if l is not None]
    return {
        "mean": np.mean(latencies),
        "median": np.median(latencies),
        "p90": np.percentile(latencies, 90),
        # Add box plot statistics
        "q1": np.percentile(latencies, 25),
        "q3": np.percentile(latencies, 75),
        "min": np.min(latencies),
        "max": np.max(latencies),
        "whisker_low": np.percentile(latencies, 5),
        "whisker_high": np.percentile(latencies, 95),
        # Include raw data for box plot
        "raw": latencies,
    }
