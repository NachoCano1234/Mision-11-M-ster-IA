from src.anomaly import detect

def test_detect_returns_indices():
    indices = detect()
    assert len(indices) > 0
    assert all(0 <= index < 200 for index in indices)
