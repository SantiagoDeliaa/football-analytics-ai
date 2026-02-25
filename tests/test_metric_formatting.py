from src.utils.metric_formatting import format_metric_mean, format_metric_range

def test_format_metric_mean_handles_none():
    assert format_metric_mean(None, 1) == "N/A"
    assert format_metric_mean({"mean": None}, 1) == "N/A"

def test_format_metric_mean_formats_value():
    assert format_metric_mean({"mean": 12.345}, 1) == "12.3"
    assert format_metric_mean({"mean": 12}, 0) == "12"

def test_format_metric_range_handles_none():
    assert format_metric_range(None, 2) == ("N/A", "N/A", "N/A")
    assert format_metric_range({"mean": None, "min": 1, "max": 2}, 2) == ("N/A", "N/A", "N/A")

def test_format_metric_range_formats_values():
    assert format_metric_range({"mean": 1.234, "min": 0.1, "max": 9.99}, 2) == ("1.23", "0.10", "9.99")
