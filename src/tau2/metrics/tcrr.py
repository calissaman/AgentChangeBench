from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from loguru import logger

from tau2.data_model.message import Message, ToolMessage, AssistantMessage, UserMessage
from tau2.data_model.simulation import SimulationRun
from tau2.metrics.config import get_tcrr_window_size


@dataclass
class ToolCallInfo:
    """Information about a tool call with turn context."""
    name: str
    params: dict
    turn_idx: int
    assistant_turn_idx: int  # Which assistant turn this belongs to
    call_id: str
    correct: bool
    params_valid: bool


@dataclass
class TCRRResult:
    """Result of TCRR computation."""
    total_calls: int
    redundant_calls: int
    redundancy_ratio: float
    redundant_by_turn: Dict[int, int]  # Turn index -> number of redundant calls
    window_size: int


def normalized_params(params: dict) -> tuple:
    """Recursively normalize parameters for consistency."""
    if not isinstance(params, dict):
        return str(params)

    normalized_items = []
    for k, v in sorted(params.items()):
        if isinstance(v, dict):
            normalized_items.append((k, normalized_params(v)))
        elif isinstance(v, list):
            try:
                sorted_v = tuple(sorted(v))
            except TypeError:
                sorted_v = tuple(str(item) for item in v)
            normalized_items.append((k, sorted_v))
        else:
            normalized_items.append((k, v))
    return tuple(normalized_items)


def extract_tool_calls_with_turns(messages: List[Message]) -> List[ToolCallInfo]:
    """Extract tool calls with turn context information."""
    tool_calls = []
    
    # Create mapping of tool call IDs to their results
    tool_results = {}
    for msg in messages:
        if isinstance(msg, ToolMessage):
            tool_results[msg.id] = {
                "success": not msg.error,
                "error": msg.error,
                "content": msg.content,
            }
    
    # Track assistant turn indices
    assistant_turn_count = 0
    
    for msg in messages:
        # Count assistant turns
        if isinstance(msg, AssistantMessage):
            assistant_turn_count += 1
            
            # Extract tool calls from this assistant message
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    # Determine correctness
                    correct = True
                    if hasattr(tool_call, "id") and tool_call.id in tool_results:
                        correct = tool_results[tool_call.id]["success"]
                    
                    # Determine parameter validity
                    params_valid = True
                    if hasattr(tool_call, "id") and tool_call.id in tool_results:
                        result = tool_results[tool_call.id]
                        if result["error"] and result.get("content"):
                            error_content = str(result["content"]).lower()
                            if any(
                                keyword in error_content
                                for keyword in [
                                    "invalid parameter",
                                    "missing parameter", 
                                    "parameter error",
                                    "bad parameter",
                                    "invalid argument",
                                    "missing argument",
                                ]
                            ):
                                params_valid = False
                    
                    tool_call_info = ToolCallInfo(
                        name=tool_call.name,
                        params=tool_call.arguments,
                        turn_idx=msg.turn_idx or 0,
                        assistant_turn_idx=assistant_turn_count,
                        call_id=getattr(tool_call, "id", ""),
                        correct=correct,
                        params_valid=params_valid
                    )
                    tool_calls.append(tool_call_info)
    
    return tool_calls


def compute_tcrr_windowed(tool_calls: List[ToolCallInfo], window_size: int = 3) -> TCRRResult:
    """
    Compute TCRR using window-based approach.
    
    Args:
        tool_calls: List of tool calls with turn information
        window_size: Number of previous assistant turns to consider for redundancy
    
    Returns:
        TCRRResult with redundancy statistics
    """
    if not tool_calls:
        return TCRRResult(
            total_calls=0,
            redundant_calls=0,
            redundancy_ratio=0.0,
            redundant_by_turn={},
            window_size=window_size
        )
    
    total_calls = len(tool_calls)
    redundant_calls = 0
    redundant_by_turn = {}
    
    # Group tool calls by assistant turn
    calls_by_turn: Dict[int, List[ToolCallInfo]] = {}
    for call in tool_calls:
        turn = call.assistant_turn_idx
        if turn not in calls_by_turn:
            calls_by_turn[turn] = []
        calls_by_turn[turn].append(call)
    
    # Sort turns to process in order
    sorted_turns = sorted(calls_by_turn.keys())
    
    for current_turn in sorted_turns:
        current_calls = calls_by_turn[current_turn]
        redundant_by_turn[current_turn] = 0
        
        # Look back at previous turns within window
        window_start = max(1, current_turn - window_size)
        previous_calls = []
        
        for prev_turn in range(window_start, current_turn):
            if prev_turn in calls_by_turn:
                previous_calls.extend(calls_by_turn[prev_turn])
        
        # Build set of previous call identities
        previous_identities: Set[Tuple[str, tuple]] = set()
        for prev_call in previous_calls:
            try:
                params_norm = normalized_params(prev_call.params)
                identity = (prev_call.name, params_norm)
                previous_identities.add(identity)
            except Exception as e:
                logger.warning(f"Error normalizing params for TCRR: {e}")
                # Fallback to string representation
                identity = (prev_call.name, str(prev_call.params))
                previous_identities.add(identity)
        
        # Check current calls for redundancy
        for call in current_calls:
            try:
                params_norm = normalized_params(call.params)
                identity = (call.name, params_norm)
                
                if identity in previous_identities:
                    redundant_calls += 1
                    redundant_by_turn[current_turn] += 1
                    
            except Exception as e:
                logger.warning(f"Error normalizing params for TCRR: {e}")
                # Fallback to string representation
                identity = (call.name, str(call.params))
                if identity in previous_identities:
                    redundant_calls += 1
                    redundant_by_turn[current_turn] += 1
    
    redundancy_ratio = redundant_calls / total_calls if total_calls > 0 else 0.0
    
    return TCRRResult(
        total_calls=total_calls,
        redundant_calls=redundant_calls,
        redundancy_ratio=redundancy_ratio,
        redundant_by_turn=redundant_by_turn,
        window_size=window_size
    )


def compute_tcrr_enhanced(
    simulations: List[SimulationRun], 
    window_size: Optional[int] = None
) -> TCRRResult:
    """
    Compute enhanced TCRR across all simulations with window-based redundancy detection.
    
    Args:
        simulations: List of simulation runs
        window_size: Number of previous assistant turns to consider (uses config default if None)
        
    Returns:
        Aggregated TCRRResult across all simulations
    """
    if window_size is None:
        window_size = get_tcrr_window_size()
        
    all_tool_calls = []
    
    for sim in simulations:
        tool_calls = extract_tool_calls_with_turns(sim.messages)
        all_tool_calls.extend(tool_calls)
    
    return compute_tcrr_windowed(all_tool_calls, window_size)


def compute_tcrr_by_task(
    simulations: List[SimulationRun],
    window_size: Optional[int] = None
) -> Dict[str, TCRRResult]:
    """
    Compute TCRR for each task separately.
    
    Args:
        simulations: List of simulation runs
        window_size: Number of previous assistant turns to consider (uses config default if None)
        
    Returns:
        Dictionary mapping task_id to TCRRResult
    """
    if window_size is None:
        window_size = get_tcrr_window_size()
        
    results_by_task = {}
    
    # Group simulations by task
    sims_by_task: Dict[str, List[SimulationRun]] = {}
    for sim in simulations:
        task_id = sim.task_id
        if task_id not in sims_by_task:
            sims_by_task[task_id] = []
        sims_by_task[task_id].append(sim)
    
    # Compute TCRR for each task
    for task_id, task_sims in sims_by_task.items():
        results_by_task[task_id] = compute_tcrr_enhanced(task_sims, window_size)
    
    return results_by_task
