from datetime import datetime
from typing import Dict, Any, Optional
from .schema import MetaEvent

def emit_meta_event(
    run_id: str, 
    turn_idx: int, 
    participant: str, 
    meta: MetaEvent, 
    ts: datetime
) -> Dict[str, Any]:
    """
    Emit a structured meta event for logging to simulation records.
    
    Args:
        run_id: Unique identifier for the simulation run
        turn_idx: Turn index in the conversation
        participant: "user" or "assistant"
        meta: The MetaEvent object
        ts: Timestamp of the event
        
    Returns:
        Structured event dictionary ready for logging
    """
    return {
        "run_id": run_id,
        "turn_idx": turn_idx,
        "ts": ts.isoformat(),
        "participant": participant,
        "meta": meta.dict(by_alias=True)
    }

def log_meta_error(
    run_id: str,
    turn_idx: int,
    participant: str,
    error: str,
    raw_line: Optional[str] = None,
    ts: Optional[datetime] = None
) -> Dict[str, Any]:
    """
    Log a meta parsing error event.
    
    Args:
        run_id: Unique identifier for the simulation run
        turn_idx: Turn index in the conversation
        participant: "user" or "assistant"
        error: Error description
        raw_line: The raw meta line that failed to parse
        ts: Timestamp (defaults to now)
        
    Returns:
        Structured error event dictionary
    """
    if ts is None:
        ts = datetime.now()
        
    return {
        "run_id": run_id,
        "turn_idx": turn_idx,
        "ts": ts.isoformat(),
        "participant": participant,
        "meta_error": error,
        "raw_line": raw_line
    } 