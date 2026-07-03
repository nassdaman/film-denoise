"""
Preview utility: time parsing for preview clip generation.

The actual preview encode is now handled directly in film_denoiser.py
via EncodeJob / run_encode with decoupled denoise/grain/encode params.
"""


def parse_time_to_seconds(time_str: str) -> float:
    """Parse a time string to seconds.

    Supports: "600", "10:00", "1:00:00", "90.5"
    Returns seconds as float.
    """
    time_str = time_str.strip()
    parts = time_str.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return float(h) * 3600 + float(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return float(m) * 60 + float(s)
    else:
        return float(time_str)
