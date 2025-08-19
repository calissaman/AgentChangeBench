"""
Goal Shift Recovery Time (GSRT) calculation using meta-tags v2.

This module computes GSRT metrics using the new deterministic meta tag system
and alignment scoring for both tool and no-tool goals.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from tau2.meta.schema import MetaEvent, GoalToken
from tau2.config.goal_map import GOAL_TOOL_MAP
from tau2.metrics.alignment import alignment_score, AlignmentDetectors


def compute_gsrt(
    run: Dict[str, Any], 
    tool_map: Dict[GoalToken, Dict], 
    detectors: AlignmentDetectors, 
    theta: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Compute GSRT for a simulation run using meta-tags v2.
    
    Args:
        run: Simulation run data with turns list
        tool_map: Goal to tools mapping 
        detectors: Alignment detector instance
        theta: Alignment threshold (default 0.7)
        
    Returns:
        List of GSRT results for each goal shift
    """
    if 'turns' not in run:
        return []
        
    turns = run['turns']
    
    # Find all goal shifts in user messages
    shifts = []
    for i, turn in enumerate(turns):
        if turn.get("role") != "user":
            continue
            
        meta_event = turn.get("meta_event")
        if meta_event and isinstance(meta_event, dict):
            if meta_event.get("event") == "GOAL_SHIFT":
                shifts.append({
                    "turn_idx": i,
                    "to": meta_event.get("to"),
                    "seq": meta_event.get("seq"),
                    "from": meta_event.get("from"),
                    "reason": meta_event.get("reason")
                })
        elif meta_event and hasattr(meta_event, "event") and meta_event.event == "GOAL_SHIFT":
            # Handle MetaEvent objects
            me = meta_event
            shifts.append({
                "turn_idx": i,
                "to": me.to,
                "seq": me.seq,
                "from": me.from_,
                "reason": me.reason
            })
    
    results = []
    for shift in shifts:
        start_turn = shift["turn_idx"]
        goal = shift["to"]
        
        if not goal:
            results.append({
                "seq": shift["seq"],
                "goal": None,
                "gsrt_turns": None,
                "gsrt_seconds": None,
                "error": "MISSING_GOAL"
            })
            continue
            
        # Convert string goal to GoalToken if needed
        if isinstance(goal, str):
            try:
                goal = GoalToken(goal)
            except ValueError:
                results.append({
                    "seq": shift["seq"],
                    "goal": goal,
                    "gsrt_turns": None,
                    "gsrt_seconds": None,
                    "error": f"INVALID_GOAL:{goal}"
                })
                continue
        
        # Search for alignment in subsequent assistant turns
        aligned_turn = None
        for j in range(start_turn + 1, len(turns)):
            turn = turns[j]
            if turn.get("role") != "assistant":
                continue
                
            # Calculate alignment score
            score = alignment_score(turn, goal, tool_map, detectors)
            if score >= theta:
                aligned_turn = j
                break
        
        if aligned_turn is None:
            results.append({
                "seq": shift["seq"],
                "goal": goal.value if isinstance(goal, GoalToken) else goal,
                "gsrt_turns": None,
                "gsrt_seconds": None,
                "aligned_score": None
            })
        else:
            turn_count = aligned_turn - start_turn
            
            # Calculate wall-clock time if timestamps available
            wall_clock_seconds = None
            if "ts" in turns[start_turn] and "ts" in turns[aligned_turn]:
                start_ts = turns[start_turn]["ts"]
                aligned_ts = turns[aligned_turn]["ts"]
                
                # Handle different timestamp formats
                if isinstance(start_ts, str):
                    start_ts = datetime.fromisoformat(start_ts.replace('Z', '+00:00'))
                if isinstance(aligned_ts, str):
                    aligned_ts = datetime.fromisoformat(aligned_ts.replace('Z', '+00:00'))
                    
                if isinstance(start_ts, datetime) and isinstance(aligned_ts, datetime):
                    wall_clock_seconds = (aligned_ts - start_ts).total_seconds()
            
            results.append({
                "seq": shift["seq"],
                "goal": goal.value if isinstance(goal, GoalToken) else goal,
                "gsrt_turns": turn_count,
                "gsrt_seconds": wall_clock_seconds,
                "aligned_score": alignment_score(turns[aligned_turn], goal, tool_map, detectors)
            })
    
    return results


def calculate_gsrt_statistics(gsrt_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate aggregate GSRT statistics.
    
    Args:
        gsrt_results: List of GSRT results from compute_gsrt
        
    Returns:
        Dictionary with aggregate statistics
    """
    if not gsrt_results:
        return {
            "total_shifts": 0,
            "successful_recoveries": 0,
            "median_gsrt_turns": None,
            "median_gsrt_seconds": None,
            "worst_case_gsrt_turns": None,
            "worst_case_gsrt_seconds": None,
            "recovery_rate": 0.0
        }
    
    # Filter successful recoveries
    successful = [r for r in gsrt_results if r.get("gsrt_turns") is not None]
    
    turn_times = [r["gsrt_turns"] for r in successful]
    wall_times = [r["gsrt_seconds"] for r in successful if r.get("gsrt_seconds") is not None]
    
    import statistics
    
    return {
        "total_shifts": len(gsrt_results),
        "successful_recoveries": len(successful),
        "median_gsrt_turns": statistics.median(turn_times) if turn_times else None,
        "median_gsrt_seconds": statistics.median(wall_times) if wall_times else None,
        "worst_case_gsrt_turns": max(turn_times) if turn_times else None,
        "worst_case_gsrt_seconds": max(wall_times) if wall_times else None,
        "recovery_rate": len(successful) / len(gsrt_results) if gsrt_results else 0.0
    } 