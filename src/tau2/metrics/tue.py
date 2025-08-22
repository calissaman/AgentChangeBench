from typing import List, Dict, Optional
from dataclasses import dataclass
from loguru import logger

from tau2.data_model.simulation import SimulationRun
from tau2.metrics.config import get_tue_weights


@dataclass
class TUEResult:
    """Result of TUE computation."""
    overall_tue: float
    tool_correctness: float
    param_accuracy: float
    total_calls: int
    correct_calls: int
    valid_param_calls: int
    components: Dict[str, float]


def compute_tue_enhanced(tool_calls: List[dict]) -> TUEResult:
    """
    Compute enhanced Tool Usage Efficiency without latency component.
    
    Enhanced TUE = 60% × Tool_Correctness + 40% × Param_Accuracy
    
    Args:
        tool_calls: List of tool call dictionaries with keys: correct, params_valid, cost, latency
        
    Returns:
        TUEResult with detailed breakdown
    """
    if not tool_calls:
        return TUEResult(
            overall_tue=0.0,
            tool_correctness=0.0,
            param_accuracy=0.0,
            total_calls=0,
            correct_calls=0,
            valid_param_calls=0,
            components={"tool_correctness": 0.0, "param_accuracy": 0.0}
        )
    
    total_calls = len(tool_calls)
    
    # Tool Correctness: Fraction of tool calls that executed without errors
    correct_calls = sum(1 for call in tool_calls if call.get("correct", False))
    tool_correctness = correct_calls / total_calls
    
    # Parameter Accuracy: Fraction of tool calls with valid parameters
    valid_param_calls = sum(1 for call in tool_calls if call.get("params_valid", False))
    param_accuracy = valid_param_calls / total_calls
    
    # Enhanced TUE formula (without latency)
    # Use configured weights for TUE calculation
    weights = get_tue_weights()
    overall_tue = (
        weights['tool_correctness'] * tool_correctness + 
        weights['param_accuracy'] * param_accuracy
    )
    
    components = {
        "tool_correctness": tool_correctness,
        "param_accuracy": param_accuracy
    }
    
    return TUEResult(
        overall_tue=overall_tue,
        tool_correctness=tool_correctness,
        param_accuracy=param_accuracy,
        total_calls=total_calls,
        correct_calls=correct_calls,
        valid_param_calls=valid_param_calls,
        components=components
    )


def extract_tool_calls_for_tue(messages: List) -> List[dict]:
    """Extract tool calls from messages for TUE computation."""
    tool_calls = []
    
    # Create mapping of tool call IDs to their results
    tool_results = {}
    for msg in messages:
        if hasattr(msg, 'role') and msg.role == 'tool':
            tool_results[msg.id] = {
                "success": not getattr(msg, 'error', False),
                "error": getattr(msg, 'error', False),
                "content": getattr(msg, 'content', None),
            }
    
    for msg in messages:
        # Look for tool calls in AssistantMessage and UserMessage objects
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tool_call in msg.tool_calls:
                # Determine correctness based on whether the tool executed without errors
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
                
                tool_call_dict = {
                    "name": tool_call.name,
                    "params": getattr(tool_call, 'arguments', {}),
                    "correct": correct,
                    "params_valid": params_valid,
                }
                tool_calls.append(tool_call_dict)
    
    return tool_calls


def compute_tue_enhanced_for_simulations(
    simulations: List[SimulationRun]
) -> TUEResult:
    """
    Compute enhanced TUE across all simulations.
    
    Args:
        simulations: List of simulation runs
        
    Returns:
        Aggregated TUEResult across all simulations
    """
    all_tool_calls = []
    
    for sim in simulations:
        tool_calls = extract_tool_calls_for_tue(sim.messages)
        all_tool_calls.extend(tool_calls)
    
    return compute_tue_enhanced(all_tool_calls)


def compute_tue_by_task(
    simulations: List[SimulationRun]
) -> Dict[str, TUEResult]:
    """
    Compute TUE for each task separately.
    
    Args:
        simulations: List of simulation runs
        
    Returns:
        Dictionary mapping task_id to TUEResult
    """
    results_by_task = {}
    
    # Group simulations by task
    sims_by_task: Dict[str, List[SimulationRun]] = {}
    for sim in simulations:
        task_id = sim.task_id
        if task_id not in sims_by_task:
            sims_by_task[task_id] = []
        sims_by_task[task_id].append(sim)
    
    # Compute TUE for each task
    for task_id, task_sims in sims_by_task.items():
        results_by_task[task_id] = compute_tue_enhanced_for_simulations(task_sims)
    
    return results_by_task
