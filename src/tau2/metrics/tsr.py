from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from loguru import logger

from tau2.data_model.simulation import SimulationRun
from tau2.data_model.tasks import Task, EvaluationCriteria, Action
from tau2.metrics.config import get_tsr_weights


@dataclass
class ActionSet:
    """New action format for enhanced evaluation."""

    action_id: str
    allowed_tools: List[Dict[str, Any]]  # List of {function_name, params}


@dataclass
class TSRChannelResult:
    """Result for a single TSR channel."""

    success_count: int
    total_count: int
    success_rate: float
    tasks_evaluated: int


@dataclass
class TSRResult:
    """Complete TSR computation result."""

    overall_tsr: float
    communicate_info: TSRChannelResult
    action: TSRChannelResult
    nl_assertion: TSRChannelResult
    # Weights used in calculation
    weights: Dict[str, float]
    # Task-level breakdown
    by_task: Dict[str, Dict[str, float]]


def extract_action_sets_from_task(task: Task) -> List[ActionSet]:
    """Extract action_sets from task, prioritizing action_sets over legacy actions."""
    if not task.evaluation_criteria:
        return []

    # Priority 1: Use action_sets if available (even if actions also exists)
    if (
        hasattr(task.evaluation_criteria, "action_sets")
        and task.evaluation_criteria.action_sets
    ):
        action_sets = []
        for action_set_data in task.evaluation_criteria.action_sets:
            if isinstance(action_set_data, dict):
                action_sets.append(
                    ActionSet(
                        action_id=action_set_data.get("action_id", ""),
                        allowed_tools=action_set_data.get("allowed_tools", []),
                    )
                )
        return action_sets

    # Priority 2: Fallback to legacy actions format only if no action_sets
    if task.evaluation_criteria.actions:
        action_sets = []
        for action in task.evaluation_criteria.actions:
            # Convert legacy Action to ActionSet format
            allowed_tool = {"function_name": action.name, "params": action.arguments}
            action_sets.append(
                ActionSet(action_id=action.action_id, allowed_tools=[allowed_tool])
            )
        return action_sets

    return []


def evaluate_communicate_info_for_task(
    task: Task, sim: SimulationRun
) -> tuple[float, bool]:
    """Evaluate communicate_info component for a single task.

    Returns:
        tuple: (score, has_communicate_info) where has_communicate_info indicates if task has communicate requirements
    """
    if not task.evaluation_criteria or not task.evaluation_criteria.communicate_info:
        return 0.0, False  # No communicate_info to evaluate

    communicate_info = task.evaluation_criteria.communicate_info
    if not communicate_info:
        return 0.0, False

    # Extract all assistant messages
    assistant_messages = [
        msg
        for msg in sim.messages
        if hasattr(msg, "role")
        and msg.role == "assistant"
        and hasattr(msg, "content")
        and msg.content
    ]

    if not assistant_messages:
        return 0.0, True

    # Combine all assistant text
    full_response = " ".join(msg.content for msg in assistant_messages)
    full_response_lower = full_response.lower()

    # Check for each required info
    matched_count = 0
    for info in communicate_info:
        if info.lower() in full_response_lower:
            matched_count += 1

    return (matched_count / len(communicate_info) if communicate_info else 1.0), True


def evaluate_actions_for_task(task: Task, sim: SimulationRun) -> tuple[float, bool]:
    """Evaluate actions component for a single task using new action_sets format.

    Returns:
        tuple: (score, has_actions) where has_actions indicates if task has action requirements
    """
    action_sets = extract_action_sets_from_task(task)
    if not action_sets:
        return 0.0, False  # No actions to evaluate

    # Extract tool calls from simulation
    tool_calls = []
    for msg in sim.messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tool_call in msg.tool_calls:
                tool_calls.append(
                    {
                        "name": tool_call.name,
                        "arguments": getattr(tool_call, "arguments", {}),
                    }
                )

    total_score = 0.0

    for action_set in action_sets:
        action_score = 0.0

        # Check if any allowed tool was used
        for allowed_tool in action_set.allowed_tools:
            function_name = allowed_tool.get("function_name", "")
            expected_params = allowed_tool.get("params", {})

            # Look for matching tool calls
            for tool_call in tool_calls:
                if tool_call["name"] == function_name:
                    # Tool correctness: 0.5 points for using the right tool
                    tool_correctness = 0.5

                    # Parameter correctness: 0.5 points for correct params
                    param_correctness = 0.0
                    if _params_match(tool_call["arguments"], expected_params):
                        param_correctness = 0.5

                    # Take the best score for this action set
                    action_score = max(
                        action_score, tool_correctness + param_correctness
                    )
                    break

        total_score += action_score

    return total_score / len(action_sets), True  # Has actions to evaluate


def _params_match(actual_params: dict, expected_params: dict) -> bool:
    """Check if actual parameters match expected parameters."""
    if not expected_params:  # No expected params means any params are fine
        return True

    for key, expected_value in expected_params.items():
        if key not in actual_params:
            return False
        if actual_params[key] != expected_value:
            return False

    return True


def evaluate_nl_assertions_for_task(
    task: Task, sim: SimulationRun
) -> tuple[float, bool, list]:
    """Evaluate NL assertions component for a single task.

    Returns:
        tuple: (score, has_nl_assertions, detailed_checks) where:
        - score: fraction of assertions that passed
        - has_nl_assertions: indicates if task has NL requirements
        - detailed_checks: list of NLAssertionCheck objects with justifications
    """
    if not task.evaluation_criteria or not task.evaluation_criteria.nl_assertions:
        return 0.0, False, []  # No NL assertions to evaluate

    nl_assertions = task.evaluation_criteria.nl_assertions
    if not nl_assertions:
        return 0.0, False, []

    # Use NLAssertionsEvaluator to evaluate assertions
    try:
        from tau2.evaluator.evaluator_nl_assertions import NLAssertionsEvaluator

        # Use the class method that returns detailed results
        nl_checks = NLAssertionsEvaluator.evaluate_nl_assertions(
            [msg for msg in sim.messages], nl_assertions
        )
        passed_count = sum(1 for check in nl_checks if check.met)
        return passed_count / len(nl_assertions), True, nl_checks

    except Exception as e:
        # If NL evaluation fails, exclude from scoring rather than guessing
        import logging

        logging.warning(
            f"NL assertion evaluation failed for task, excluding from TSR: {e}"
        )
        return 0.0, False, []  # Exclude from scoring to maintain data integrity


def compute_tsr_for_task(
    task: Task, sim: SimulationRun, weights: Dict[str, float]
) -> Dict[str, float]:
    """Compute TSR for a single task with dynamic reweighting for missing components."""
    communicate_score, has_communicate_info = evaluate_communicate_info_for_task(
        task, sim
    )
    action_score, has_actions = evaluate_actions_for_task(task, sim)
    nl_score, has_nl_assertions, nl_assertion_details = evaluate_nl_assertions_for_task(
        task, sim
    )

    # Dynamic reweighting: exclude components that don't exist
    active_weights = {}
    if has_actions:
        active_weights["action"] = weights["action"]
    if has_nl_assertions:
        active_weights["nl_assertion"] = weights["nl_assertion"]
    if has_communicate_info:
        active_weights["communicate_info"] = weights["communicate_info"]

    # Normalize weights to sum to 1.0
    total_weight = sum(active_weights.values())
    if total_weight > 0:
        normalized_weights = {k: v / total_weight for k, v in active_weights.items()}
    else:
        # Fallback: if no components exist, return 0
        normalized_weights = {}

    # Weighted TSR calculation using normalized weights
    overall_tsr = 0.0
    if "communicate_info" in normalized_weights and has_communicate_info:
        overall_tsr += normalized_weights["communicate_info"] * communicate_score
    if "action" in normalized_weights and has_actions:
        overall_tsr += normalized_weights["action"] * action_score
    if "nl_assertion" in normalized_weights and has_nl_assertions:
        overall_tsr += normalized_weights["nl_assertion"] * nl_score

    return {
        "overall_tsr": overall_tsr,
        "communicate_info": communicate_score if has_communicate_info else None,
        "action": action_score if has_actions else None,
        "nl_assertion": nl_score if has_nl_assertions else None,
        "weights_used": normalized_weights,
        "has_actions": has_actions,
        "has_nl_assertions": has_nl_assertions,
        "has_communicate_info": has_communicate_info,
        "nl_assertion_details": nl_assertion_details,
    }


def compute_reward_from_tsr(
    task: Task, sim: SimulationRun, weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Compute the reward based on TSR calculation.
    This is the standard reward system.

    Args:
        task: Task definition
        sim: Simulation run
        weights: Channel weights

    Returns:
        Reward score (0.0 to 1.0)
    """
    if weights is None:
        weights = get_tsr_weights()

    task_scores = compute_tsr_for_task(task, sim, weights)
    return task_scores["overall_tsr"]
