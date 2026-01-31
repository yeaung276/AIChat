def extract_latencies(waterfall, sample_id):
    rows = []

    for i in range(len(waterfall) - 1):
        c_from = waterfall[i]["component"]
        c_to = waterfall[i + 1]["component"]
        latency_ms = (waterfall[i + 1]["time"] - waterfall[i]["time"]) * 1000

        rows.append({
            "sample": sample_id,
            "component": f"{c_from}->{c_to}",
            "latency_ms": latency_ms,
        })

    # end-to-end
    rows.append({
        "sample": sample_id,
        "component": "total",
        "latency_ms": (waterfall[-1]["time"] - waterfall[0]["time"]) * 1000,
    })

    return rows