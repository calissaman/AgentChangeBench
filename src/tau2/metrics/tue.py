from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

from tau2.data_model.simulation import SimulationRun
from tau2.metrics.config import get_tue_weights


@dataclass
class TUEResult:
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
            components={"tool_correctness": 0.0, "param_accuracy": 0.0},
        )

    total_calls = len(tool_calls)

    correct_calls = sum(1 for call in tool_calls if call.get("correct", False))
    tool_correctness = correct_calls / total_calls

    valid_param_calls = sum(1 for call in tool_calls if call.get("params_valid", False))
    param_accuracy = valid_param_calls / total_calls

    weights = get_tue_weights()
    overall_tue = (
        weights["tool_correctness"] * tool_correctness
        + weights["param_accuracy"] * param_accuracy
    )

    components = {
        "tool_correctness": tool_correctness,
        "param_accuracy": param_accuracy,
    }

    return TUEResult(
        overall_tue=overall_tue,
        tool_correctness=tool_correctness,
        param_accuracy=param_accuracy,
        total_calls=total_calls,
        correct_calls=correct_calls,
        valid_param_calls=valid_param_calls,
        components=components,
    )


def extract_tool_calls_for_tue(messages: List) -> List[dict]:
    """Extract tool calls from messages for TUE computation."""
    tool_calls = []

    tool_results = {}
    for msg in messages:
        if hasattr(msg, "role") and msg.role == "tool":
            tool_results[msg.id] = {
                "success": not getattr(msg, "error", False),
                "error": getattr(msg, "error", False),
                "content": getattr(msg, "content", None),
            }

    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tool_call in msg.tool_calls:
                correct = True
                if hasattr(tool_call, "id") and tool_call.id in tool_results:
                    correct = tool_results[tool_call.id]["success"]

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
                    "params": getattr(tool_call, "arguments", {}),
                    "correct": correct,
                    "params_valid": params_valid,
                }
                tool_calls.append(tool_call_dict)

    return tool_calls


def compute_tue(
    simulations: List[SimulationRun],
) -> Tuple[TUEResult, Dict[str, TUEResult]]:
    """
    Compute TUE for each task separately and return both aggregated and per-task results.

    Args:
        simulations: List of simulation runs

    Returns:
        Tuple of (aggregated_result, results_by_task)
    """
    results_by_task = {}
    all_tool_calls = []

    # Group simulations by task
    sims_by_task: Dict[str, List[SimulationRun]] = {}
    for sim in simulations:
        task_id = sim.task_id
        if task_id not in sims_by_task:
            sims_by_task[task_id] = []
        sims_by_task[task_id].append(sim)

    # Compute TUE for each task
    for task_id, task_sims in sims_by_task.items():
        task_tool_calls = []
        for sim in task_sims:
            tool_calls = extract_tool_calls_for_tue(sim.messages)
            task_tool_calls.extend(tool_calls)
            all_tool_calls.extend(tool_calls)

        results_by_task[task_id] = compute_tue_enhanced(task_tool_calls)

    aggregated_result = compute_tue_enhanced(all_tool_calls)

    return aggregated_result, results_by_task
