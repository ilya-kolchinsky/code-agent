"""Format command output as structured XML observations.

Matches mini-swe-agent's observation template: wraps stdout in XML
tags with returncode, and truncates long output by showing head+tail
with a warning.
"""

from __future__ import annotations


def format_observation(
    stdout: str,
    returncode: int,
    max_chars: int = 10000,
) -> str:
    """Format command output as an XML observation block.

    When *stdout* exceeds *max_chars*, shows the first and last
    ``max_chars // 2`` characters with a warning.
    """
    rc_line = f"<returncode>{returncode}</returncode>"

    if len(stdout) <= max_chars:
        return f"{rc_line}\n<output>\n{stdout}\n</output>"

    half = max_chars // 2
    elided = len(stdout) - max_chars
    return (
        f"{rc_line}\n"
        f"<warning>Output too long. Use head/tail/grep to get smaller output.</warning>\n"
        f"<output_head>\n{stdout[:half]}\n</output_head>\n"
        f"<elided_chars>{elided} characters elided</elided_chars>\n"
        f"<output_tail>\n{stdout[-half:]}\n</output_tail>"
    )
