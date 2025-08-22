import json
import statistics
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from tau2.utils.llm_utils import generate
from tau2.data_model.message import (
    SystemMessage,
    UserMessage,
)
from tau2.data_model.simulation import SimulationRun
from tau2.data_model.tasks import Task
from tau2.metrics.config import get_gsrt_judge_config


@dataclass
class GSRTShiftResult:
    shift_turn: int
    from_goal: str
    to_goal: str
    acknowledgment_turn: Optional[int] = None
    tool_usage_turn: Optional[int] = None
    outcome_turn: Optional[int] = None
    transferred_to_human: bool = False
    recovery_successful: bool = False


@dataclass
class GSRTTaskResult:
    task_id: str
    shifts: List[GSRTShiftResult]
    total_shifts: int
    recovery_times_ack: List[int]
    recovery_times_tool: List[int]
    recovery_times_outcome: List[int]
    median_gsrt_ack: Optional[float] = None
    median_gsrt_tool: Optional[float] = None
    median_gsrt_outcome: Optional[float] = None
    recovery_rate: float = 0.0
    transfer_rate: float = 0.0


@dataclass
class GSRTAggregateResult:
    by_task: Dict[str, GSRTTaskResult]
    aggregate_median_ack: Optional[float] = None
    aggregate_median_tool: Optional[float] = None
    aggregate_median_outcome: Optional[float] = None
    total_shifts: int = 0
    overall_recovery_rate: float = 0.0
    overall_transfer_rate: float = 0.0
    never_recovered_rate: float = 0.0


JUDGE_SYSTEM_PROMPT = """
You are a judge. Given a task description (with expected goal order) and a full conversation transcript (user/assistant turns), you must:

STRICT GOAL RULES
- Only use goals from the provided expected_goal_sequence (allowed_goals).
- Follow the order as given. Do not introduce new goals not in the list.
- Cap the number of goal shifts to required_shifts.
- Collapse sub-intents into their parent goal token (e.g., "file the dispute" is still within "dispute").
- The first time the USER establishes the first goal, record it as start_goal (this is NOT a shift event).
- A goal shift occurs ONLY when the USER moves from the current allowed goal token to the next allowed goal token in sequence.

WHAT TO OUTPUT
1) start_goal: the first user turn index where the first allowed goal is clearly established.
   - {"turn": <int>, "goal": <token>}
2) user_goal_shifts: a list of user-initiated shifts that comply with allowed_goals order and required_shifts cap.
   - For each shift: {"turn": <int>, "from": <token>, "to": <token>, "type": "GOAL_SHIFT", "agent_responded": <bool>, "agent_turn": <int or null>, "acknowledgment_turn": <int or null>, "tool_turn": <int or null>, "outcome_turn": <int or null>, "transferred_to_human": <bool>}

RECOVERY DETECTION
For each goal shift, identify:
- acknowledgment_turn: First assistant turn that meaningfully acknowledges the new goal (NOT transfer attempts)
- tool_turn: First assistant turn that uses tools relevant to the new goal
- outcome_turn: First turn where the new goal is actually achieved/completed
- transferred_to_human: True if agent attempts to transfer instead of handling the goal shift

TRANSFER DETECTION
Mark transferred_to_human as true if the assistant says phrases like:
- "let me transfer you"
- "connecting you to a specialist"
- "I'll get a human agent"
- "transferring to customer service"
- Similar transfer language

Return STRICT JSON with exactly these keys at top-level: {"start_goal": {...}, "user_goal_shifts": [ ... ]}. No extra keys or explanations.
""".strip()


def _safe_goal_sequence(task: Task) -> List[str]:
    """Extract goal sequence from task safely."""
    goals: List[str] = []
    user_scenario = getattr(task, "user_scenario", None)
    if not user_scenario:
        return goals
    goal_shifts = getattr(user_scenario, "goal_shifts", None)
    if isinstance(goal_shifts, dict):
        gs = goal_shifts.get("goals")
        if isinstance(gs, list):
            goals = [str(g) for g in gs]
    else:
        maybe_goals = getattr(goal_shifts, "goals", None)
        if isinstance(maybe_goals, list):
            goals = [str(g) for g in maybe_goals]
    return goals


def _safe_required_shifts(task: Task, allowed_goals: List[str]) -> int:
    """Extract required shifts from task safely."""
    default = max(0, len(allowed_goals) - 1)
    user_scenario = getattr(task, "user_scenario", None)
    if not user_scenario:
        return default
    goal_shifts = getattr(user_scenario, "goal_shifts", None)
    if isinstance(goal_shifts, dict):
        rs = goal_shifts.get("required_shifts")
        if isinstance(rs, int) and rs >= 0:
            return rs
    else:
        rs = getattr(goal_shifts, "required_shifts", None)
        if isinstance(rs, int) and rs >= 0:
            return rs
    return default


def _update_user_prompt(task: Task, sim: SimulationRun) -> str:
    """Build enhanced user prompt for GSRT detection."""
    allowed_goals = _safe_goal_sequence(task)
    required_shifts = _safe_required_shifts(task, allowed_goals)
    goal_str = ", ".join(allowed_goals) if allowed_goals else ""

    lines: List[str] = []
    for i, m in enumerate(sim.messages):
        role = m.role
        text = m.content or ""
        lines.append(f"{i}:{role}:{text}")
    transcript = "\n".join(lines)

    prompt = f"""TASK CONTEXT
- allowed_goals (ordered): [{goal_str}]
- required_shifts: {required_shifts}

CONVERSATION (one line per turn)
Format: <turn_index>:<role>:<text>
{transcript}

OUTPUT FORMAT (STRICT)
{{
  "start_goal": {{"turn": <int>, "goal": <token>}},
  "user_goal_shifts": [
    {{"turn": <int>, "from": <token>, "to": <token>, "type": "GOAL_SHIFT", "agent_responded": <bool>, "agent_turn": <int or null>, "acknowledgment_turn": <int or null>, "tool_turn": <int or null>, "outcome_turn": <int or null>, "transferred_to_human": <bool>}}
  ]
}}
"""
    return prompt


def detect_gsrt_enhanced(
    task: Task,
    sim: SimulationRun,
    model: Optional[str] = None,
    llm_args: Optional[dict] = None,
) -> Dict[str, Any]:
    """Enhanced GSRT detection with multi-variant recovery and transfer detection."""
    if model is None or llm_args is None:
        default_model, default_args = get_gsrt_judge_config()
        if model is None:
            model = default_model
        if llm_args is None:
            llm_args = default_args

    messages = [
        SystemMessage(role="system", content=JUDGE_SYSTEM_PROMPT),
        UserMessage(role="user", content=_update_user_prompt(task, sim)),
    ]
    assistant_message = generate(model=model, messages=messages, **llm_args)

    try:
        data = json.loads(assistant_message.content)
        if (
            not isinstance(data, dict)
            or "user_goal_shifts" not in data
            or "start_goal" not in data
        ):
            raise ValueError("Bad judge JSON structure")
        shifts = data.get("user_goal_shifts", [])
        if not isinstance(shifts, list):
            raise ValueError("user_goal_shifts must be a list")
        return data
    except Exception:
        return {"start_goal": {"turn": None, "goal": None}, "user_goal_shifts": []}


def compute_gsrt_for_task(
    task: Task,
    sim: SimulationRun,
    model: Optional[str] = None,
    llm_args: Optional[dict] = None,
) -> GSRTTaskResult:
    """Compute GSRT metrics for a single task."""
    detection_result = detect_gsrt_enhanced(task, sim, model, llm_args)
    shifts_data = detection_result.get("user_goal_shifts", [])

    shifts = []
    recovery_times_ack = []
    recovery_times_tool = []
    recovery_times_outcome = []

    for shift_data in shifts_data:
        shift_turn = shift_data.get("turn", 0)
        from_goal = shift_data.get("from", "")
        to_goal = shift_data.get("to", "")

        ack_turn = shift_data.get("acknowledgment_turn")
        tool_turn = shift_data.get("tool_turn")
        outcome_turn = shift_data.get("outcome_turn")
        transferred = shift_data.get("transferred_to_human", False)

        recovery_successful = False
        if ack_turn is not None and ack_turn > shift_turn:
            recovery_times_ack.append(ack_turn - shift_turn)
            recovery_successful = True

        if tool_turn is not None and tool_turn > shift_turn:
            recovery_times_tool.append(tool_turn - shift_turn)

        if outcome_turn is not None and outcome_turn > shift_turn:
            recovery_times_outcome.append(outcome_turn - shift_turn)

        shift_result = GSRTShiftResult(
            shift_turn=shift_turn,
            from_goal=from_goal,
            to_goal=to_goal,
            acknowledgment_turn=ack_turn,
            tool_usage_turn=tool_turn,
            outcome_turn=outcome_turn,
            transferred_to_human=transferred,
            recovery_successful=recovery_successful,
        )
        shifts.append(shift_result)

    total_shifts = len(shifts)
    recovery_rate = (
        sum(1 for s in shifts if s.recovery_successful) / total_shifts
        if total_shifts > 0
        else 0.0
    )
    transfer_rate = (
        sum(1 for s in shifts if s.transferred_to_human) / total_shifts
        if total_shifts > 0
        else 0.0
    )

    median_ack = statistics.median(recovery_times_ack) if recovery_times_ack else None
    median_tool = (
        statistics.median(recovery_times_tool) if recovery_times_tool else None
    )
    median_outcome = (
        statistics.median(recovery_times_outcome) if recovery_times_outcome else None
    )

    return GSRTTaskResult(
        task_id=sim.task_id,
        shifts=shifts,
        total_shifts=total_shifts,
        recovery_times_ack=recovery_times_ack,
        recovery_times_tool=recovery_times_tool,
        recovery_times_outcome=recovery_times_outcome,
        median_gsrt_ack=median_ack,
        median_gsrt_tool=median_tool,
        median_gsrt_outcome=median_outcome,
        recovery_rate=recovery_rate,
        transfer_rate=transfer_rate,
    )


def compute_gsrt_aggregate(
    results_by_task: Dict[str, GSRTTaskResult],
) -> GSRTAggregateResult:
    """Aggregate GSRT results across all tasks."""
    all_recovery_times_ack = []
    all_recovery_times_tool = []
    all_recovery_times_outcome = []
    total_shifts = 0
    total_recovered = 0
    total_transferred = 0

    for task_result in results_by_task.values():
        all_recovery_times_ack.extend(task_result.recovery_times_ack)
        all_recovery_times_tool.extend(task_result.recovery_times_tool)
        all_recovery_times_outcome.extend(task_result.recovery_times_outcome)
        total_shifts += task_result.total_shifts
        total_recovered += sum(1 for s in task_result.shifts if s.recovery_successful)
        total_transferred += sum(
            1 for s in task_result.shifts if s.transferred_to_human
        )

    aggregate_median_ack = (
        statistics.median(all_recovery_times_ack) if all_recovery_times_ack else None
    )
    aggregate_median_tool = (
        statistics.median(all_recovery_times_tool) if all_recovery_times_tool else None
    )
    aggregate_median_outcome = (
        statistics.median(all_recovery_times_outcome)
        if all_recovery_times_outcome
        else None
    )

    overall_recovery_rate = total_recovered / total_shifts if total_shifts > 0 else 0.0
    overall_transfer_rate = (
        total_transferred / total_shifts if total_shifts > 0 else 0.0
    )
    never_recovered_rate = (
        (total_shifts - total_recovered) / total_shifts if total_shifts > 0 else 0.0
    )

    return GSRTAggregateResult(
        by_task=results_by_task,
        aggregate_median_ack=aggregate_median_ack,
        aggregate_median_tool=aggregate_median_tool,
        aggregate_median_outcome=aggregate_median_outcome,
        total_shifts=total_shifts,
        overall_recovery_rate=overall_recovery_rate,
        overall_transfer_rate=overall_transfer_rate,
        never_recovered_rate=never_recovered_rate,
    )


def compute_gsrt_enhanced_metrics(
    tasks: List[Task],
    simulations: List[SimulationRun],
    model: Optional[str] = None,
    llm_args: Optional[dict] = None,
) -> GSRTAggregateResult:
    """Compute enhanced GSRT metrics for all tasks and simulations."""
    task_map = {task.id: task for task in tasks}
    results_by_task: Dict[str, GSRTTaskResult] = {}

    for sim in simulations:
        task = task_map.get(sim.task_id)
        if not task:
            continue

        cached_result = None
        if (
            sim.reward_info
            and sim.reward_info.info
            and isinstance(sim.reward_info.info, dict)
        ):
            cached_result = sim.reward_info.info.get("gsrt_enhanced")

        if not cached_result:
            task_result = compute_gsrt_for_task(task, sim, model, llm_args)

            try:
                if sim.reward_info is not None:
                    if sim.reward_info.info is None:
                        sim.reward_info.info = {}
                    if isinstance(sim.reward_info.info, dict):
                        sim.reward_info.info["gsrt_enhanced"] = {
                            "task_id": task_result.task_id,
                            "total_shifts": task_result.total_shifts,
                            "recovery_rate": task_result.recovery_rate,
                            "transfer_rate": task_result.transfer_rate,
                            "median_gsrt_ack": task_result.median_gsrt_ack,
                            "median_gsrt_tool": task_result.median_gsrt_tool,
                            "median_gsrt_outcome": task_result.median_gsrt_outcome,
                        }
            except Exception:
                pass
        else:
            task_result = GSRTTaskResult(
                task_id=cached_result["task_id"],
                shifts=[],
                total_shifts=cached_result["total_shifts"],
                recovery_times_ack=[],
                recovery_times_tool=[],
                recovery_times_outcome=[],
                median_gsrt_ack=cached_result["median_gsrt_ack"],
                median_gsrt_tool=cached_result["median_gsrt_tool"],
                median_gsrt_outcome=cached_result["median_gsrt_outcome"],
                recovery_rate=cached_result["recovery_rate"],
                transfer_rate=cached_result["transfer_rate"],
            )

        if task_result.task_id not in results_by_task:
            results_by_task[task_result.task_id] = task_result
        else:
            existing = results_by_task[task_result.task_id]
            existing.total_shifts += task_result.total_shifts
            existing.recovery_times_ack.extend(task_result.recovery_times_ack)
            existing.recovery_times_tool.extend(task_result.recovery_times_tool)
            existing.recovery_times_outcome.extend(task_result.recovery_times_outcome)

            existing.median_gsrt_ack = (
                statistics.median(existing.recovery_times_ack)
                if existing.recovery_times_ack
                else None
            )
            existing.median_gsrt_tool = (
                statistics.median(existing.recovery_times_tool)
                if existing.recovery_times_tool
                else None
            )
            existing.median_gsrt_outcome = (
                statistics.median(existing.recovery_times_outcome)
                if existing.recovery_times_outcome
                else None
            )

    return compute_gsrt_aggregate(results_by_task)
