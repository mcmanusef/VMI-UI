"""Small shared helpers for "estimated time to completion" displays, used
wherever a run has a knowable end (Collection parameters, Monitored
Acquisition, Parameter Sweep) -- not for open-ended runs like Diagnostics
or Quick Monitor, which have no completion to estimate.
"""
import datetime


def format_duration(seconds):
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def format_eta(seconds_remaining):
    """'~<duration> remaining (finish ~HH:MM:SS)', for a live countdown."""
    seconds_remaining = max(0.0, seconds_remaining)
    finish = datetime.datetime.now() + datetime.timedelta(seconds=seconds_remaining)
    return f"~{format_duration(seconds_remaining)} remaining (finish ~{finish.strftime('%H:%M:%S')})"
