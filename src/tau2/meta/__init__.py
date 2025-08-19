"""
Meta-tags v2 system for tau2-bench.

This module provides deterministic parsing, validation, and processing
of structured meta tags in conversation messages.
"""

from .grammar import parse_meta_line
from .schema import MetaEvent, GoalToken, ShiftReason
from .logging import emit_meta_event

__all__ = [
    "parse_meta_line",
    "MetaEvent", 
    "GoalToken",
    "ShiftReason",
    "emit_meta_event"
] 