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
    if hasattr(task.evaluation_criteria, 'action_sets') and task.evaluation_criteria.action_sets:
        action_sets = []
        for action_set_data in task.evaluation_criteria.action_sets:
            if isinstance(action_set_data, dict):
                action_sets.append(ActionSet(
                    action_id=action_set_data.get('action_id', ''),
                    allowed_tools=action_set_data.get('allowed_tools', [])
                ))
        return action_sets
    
    # Priority 2: Fallback to legacy actions format only if no action_sets
    if task.evaluation_criteria.actions:
        action_sets = []
        for action in task.evaluation_criteria.actions:
            # Convert legacy Action to ActionSet format
            allowed_tool = {
                'function_name': action.name,
                'params': action.arguments
            }
            action_sets.append(ActionSet(
                action_id=action.action_id,
                allowed_tools=[allowed_tool]
            ))
        return action_sets
    
    return []


def evaluate_communicate_info_for_task(task: Task, sim: SimulationRun) -> float:
    """Evaluate communicate_info component for a single task."""
    if not task.evaluation_criteria or not task.evaluation_criteria.communicate_info:
        return 1.0  # No requirements means perfect score
    
    communicate_info = task.evaluation_criteria.communicate_info
    if not communicate_info:
        return 1.0
    
    # Extract all assistant messages
    assistant_messages = [
        msg for msg in sim.messages 
        if hasattr(msg, 'role') and msg.role == 'assistant' and hasattr(msg, 'content') and msg.content
    ]
    
    if not assistant_messages:
        return 0.0
    
    # Combine all assistant text
    full_response = " ".join(msg.content for msg in assistant_messages)
    full_response_lower = full_response.lower()
    
    # Check for each required info
    matched_count = 0
    for info in communicate_info:
        if info.lower() in full_response_lower:
            matched_count += 1
    
    return matched_count / len(communicate_info) if communicate_info else 1.0


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
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tool_call in msg.tool_calls:
                tool_calls.append({
                    'name': tool_call.name,
                    'arguments': getattr(tool_call, 'arguments', {})
                })
    
    total_score = 0.0
    
    for action_set in action_sets:
        action_score = 0.0
        
        # Check if any allowed tool was used
        for allowed_tool in action_set.allowed_tools:
            function_name = allowed_tool.get('function_name', '')
            expected_params = allowed_tool.get('params', {})
            
            # Look for matching tool calls
            for tool_call in tool_calls:
                if tool_call['name'] == function_name:
                    # Tool correctness: 0.5 points for using the right tool
                    tool_correctness = 0.5
                    
                    # Parameter correctness: 0.5 points for correct params
                    param_correctness = 0.0
                    if _params_match(tool_call['arguments'], expected_params):
                        param_correctness = 0.5
                    
                    # Take the best score for this action set
                    action_score = max(action_score, tool_correctness + param_correctness)
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


def evaluate_nl_assertions_for_task(task: Task, sim: SimulationRun) -> tuple[float, bool]:
    """Evaluate NL assertions component for a single task.
    
    Returns:
        tuple: (score, has_nl_assertions) where has_nl_assertions indicates if task has NL requirements
    """
    if not task.evaluation_criteria or not task.evaluation_criteria.nl_assertions:
        return 0.0, False  # No NL assertions to evaluate
    
    nl_assertions = task.evaluation_criteria.nl_assertions
    if not nl_assertions:
        return 0.0, False
    
    # Try to get evaluation results from reward_info.nl_assertions (primary method)
    if hasattr(sim, 'reward_info') and sim.reward_info and hasattr(sim.reward_info, 'nl_assertions'):
        nl_results = sim.reward_info.nl_assertions
        if isinstance(nl_results, list) and nl_results:
            passed_count = sum(1 for result in nl_results if result.get('met', False))
            return passed_count / len(nl_results), True
    
    # Try to get from info.nl_assertions (backup method)
    if hasattr(sim, 'reward_info') and sim.reward_info and hasattr(sim.reward_info, 'info'):
        reward_info = sim.reward_info.info
        if isinstance(reward_info, dict) and 'nl_assertions' in reward_info:
            nl_results = reward_info['nl_assertions']
            if isinstance(nl_results, list) and nl_results:
                passed_count = sum(1 for result in nl_results if result.get('met', False))
                return passed_count / len(nl_results), True
    
    # Try to get from nl field in reward_info (legacy method)
    if hasattr(sim, 'reward_info') and sim.reward_info and hasattr(sim.reward_info, 'nl'):
        if sim.reward_info.nl is not None:
            return float(sim.reward_info.nl), True
    
    # Fallback: Use NLAssertionsEvaluator if available
    try:
        from tau2.evaluation.nl_assertions_evaluator import NLAssertionsEvaluator
        evaluator = NLAssertionsEvaluator()
        
        # Extract assistant messages
        assistant_messages = [
            msg.content for msg in sim.messages 
            if hasattr(msg, 'role') and msg.role == 'assistant' and hasattr(msg, 'content') and msg.content
        ]
        
        if assistant_messages:
            conversation_text = "\n".join(assistant_messages)
            passed_count = 0
            for assertion in nl_assertions:
                if evaluator.evaluate_assertion(assertion, conversation_text):
                    passed_count += 1
            return passed_count / len(nl_assertions), True
    except ImportError:
        pass
    
    # Final fallback: assume 50% success if no evaluation method available
    return 0.5, True


def compute_tsr_for_task(task: Task, sim: SimulationRun, weights: Dict[str, float]) -> Dict[str, float]:
    """Compute TSR for a single task with dynamic reweighting for missing components."""
    communicate_score = evaluate_communicate_info_for_task(task, sim)
    action_score, has_actions = evaluate_actions_for_task(task, sim)
    nl_score, has_nl_assertions = evaluate_nl_assertions_for_task(task, sim)
    
    # Dynamic reweighting: exclude components that don't exist
    active_weights = {}
    if has_actions:
        active_weights['action'] = weights['action']
    if has_nl_assertions:
        active_weights['nl_assertion'] = weights['nl_assertion']
    
    # Communicate info is always included (tasks always have some form of communication requirement)
    active_weights['communicate_info'] = weights['communicate_info']
    
    # Normalize weights to sum to 1.0
    total_weight = sum(active_weights.values())
    if total_weight > 0:
        normalized_weights = {k: v / total_weight for k, v in active_weights.items()}
    else:
        # Fallback: only communicate_info
        normalized_weights = {'communicate_info': 1.0}
    
    # Weighted TSR calculation using normalized weights
    overall_tsr = 0.0
    if 'communicate_info' in normalized_weights:
        overall_tsr += normalized_weights['communicate_info'] * communicate_score
    if 'action' in normalized_weights and has_actions:
        overall_tsr += normalized_weights['action'] * action_score
    if 'nl_assertion' in normalized_weights and has_nl_assertions:
        overall_tsr += normalized_weights['nl_assertion'] * nl_score
    
    return {
        'overall_tsr': overall_tsr,
        'communicate_info': communicate_score,
        'action': action_score if has_actions else None,
        'nl_assertion': nl_score if has_nl_assertions else None,
        'weights_used': normalized_weights,
        'has_actions': has_actions,
        'has_nl_assertions': has_nl_assertions
    }


def compute_tsr_enhanced(
    tasks: List[Task],
    simulations: List[SimulationRun],
    weights: Optional[Dict[str, float]] = None
) -> TSRResult:
    """
    Compute enhanced TSR with multi-channel breakdown.
    
    Args:
        tasks: List of tasks
        simulations: List of simulation runs
        weights: Channel weights (defaults to 50% communicate_info, 30% action, 20% nl)
    
    Returns:
        TSRResult with detailed breakdown
    """
    if weights is None:
        weights = get_tsr_weights()
    
    # Create task lookup
    task_map = {task.id: task for task in tasks}
    
    # Results by channel
    communicate_successes = 0
    action_successes = 0
    nl_successes = 0
    overall_successes = 0
    total_tasks = 0
    
    by_task = {}
    
    for sim in simulations:
        task = task_map.get(sim.task_id)
        if not task:
            continue
        
        total_tasks += 1
        
        # Compute scores for this task
        task_scores = compute_tsr_for_task(task, sim, weights)
        
        # Track channel successes using configured threshold
        from tau2.metrics.config import MetricsConfig
        threshold = MetricsConfig.TSR_SUCCESS_THRESHOLD
        
        if task_scores['communicate_info'] >= threshold:
            communicate_successes += 1
        if task_scores['action'] >= threshold:
            action_successes += 1
        if task_scores['nl_assertion'] >= threshold:
            nl_successes += 1
        if task_scores['overall_tsr'] >= threshold:
            overall_successes += 1
        
        # Store task-level results
        by_task[sim.task_id] = task_scores
    
    if total_tasks == 0:
        return TSRResult(
            overall_tsr=0.0,
            communicate_info=TSRChannelResult(0, 0, 0.0, 0),
            action=TSRChannelResult(0, 0, 0.0, 0),
            nl_assertion=TSRChannelResult(0, 0, 0.0, 0),
            weights=weights,
            by_task={}
        )
    
    return TSRResult(
        overall_tsr=overall_successes / total_tasks,
        communicate_info=TSRChannelResult(
            communicate_successes, total_tasks, 
            communicate_successes / total_tasks, total_tasks
        ),
        action=TSRChannelResult(
            action_successes, total_tasks,
            action_successes / total_tasks, total_tasks
        ),
        nl_assertion=TSRChannelResult(
            nl_successes, total_tasks,
            nl_successes / total_tasks, total_tasks
        ),
        weights=weights,
        by_task=by_task
    )


def compute_reward_from_tsr(task: Task, sim: SimulationRun, weights: Optional[Dict[str, float]] = None) -> float:
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
    return task_scores['overall_tsr']
