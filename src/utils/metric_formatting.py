from typing import Any, Dict, Tuple

def _to_float(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except Exception:
        return None

def format_metric_mean(values: Dict[str, Any], precision: int, default: str = "N/A") -> str:
    if not isinstance(values, dict):
        return default
    value = _to_float(values.get("mean"))
    if value is None:
        return default
    return f"{value:.{precision}f}"

def format_metric_range(values: Dict[str, Any], precision: int, default: str = "N/A") -> Tuple[str, str, str]:
    if not isinstance(values, dict):
        return default, default, default
    mean_value = _to_float(values.get("mean"))
    min_value = _to_float(values.get("min"))
    max_value = _to_float(values.get("max"))
    if mean_value is None or min_value is None or max_value is None:
        return default, default, default
    mean_text = f"{mean_value:.{precision}f}"
    min_text = f"{min_value:.{precision}f}"
    max_text = f"{max_value:.{precision}f}"
    return mean_text, min_text, max_text
