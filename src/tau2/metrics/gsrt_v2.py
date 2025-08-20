import json
from typing import Any, Dict, List, Optional, Tuple

from tau2.utils.llm_utils import generate
from tau2.data_model.message import SystemMessage, UserMessage, Message
from tau2.data_model.simulation import SimulationRun
from tau2.data_model.tasks import Task


SYSTEM_PROMPT = """
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
   - For each shift: {"turn": <int>, "from": <token>, "to": <token>, "type": "GOAL_SHIFT", "agent_responded": <bool>, "agent_turn": <int or null>}
   - Find the first ASSISTANT turn index that meaningfully responds to the new goal (acknowledges or acts toward it).

Return STRICT JSON with exactly these keys at top-level: {"start_goal": {...}, "user_goal_shifts": [ ... ]}. No extra keys or explanations.
""".strip()


def _safe_goal_sequence(task: Task) -> List[str]:
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


def _build_user_prompt(task: Task, sim: SimulationRun) -> str:
    # Task context: allowed goals and required shifts
    allowed_goals = _safe_goal_sequence(task)
    required_shifts = _safe_required_shifts(task, allowed_goals)
    goal_str = ", ".join(allowed_goals) if allowed_goals else ""

    # Build transcript
    lines: List[str] = []
    for i, m in enumerate(sim.messages):
        role = m.role
        text = m.content or ""
        lines.append(f"{i}:{role}:{text}")
    transcript = "\n".join(lines)

    return (
        "TASK CONTEXT\n"
        f"- allowed_goals (ordered): [{goal_str}]\n"
        f"- required_shifts: {required_shifts}\n\n"
        "CONVERSATION (one line per turn)\n"
        "Format: <turn_index>:<role>:<text>\n"
        f"{transcript}\n\n"
        "OUTPUT FORMAT (STRICT)\n"
        "{\n"
        '  "start_goal": {"turn": <int>, "goal": <token>},\n'
        '  "user_goal_shifts": [\n'
        '    {"turn": <int>, "from": <token>, "to": <token>, "type": "GOAL_SHIFT", "agent_responded": <bool>, "agent_turn": <int or null>}\n'
        "  ]\n"
        "}\n"
    )


def detect_gsrt_v2(
    task: Task,
    sim: SimulationRun,
    model: Optional[str] = None,
    llm_args: Optional[dict] = None,
) -> Dict[str, Any]:
    if model is None:
        model = "gpt-4o-mini"  # lightweight default judge
    if llm_args is None:
        llm_args = {"temperature": 0.0}

    messages = [
        SystemMessage(role="system", content=SYSTEM_PROMPT),
        UserMessage(role="user", content=_build_user_prompt(task, sim)),
    ]
    assistant_message = generate(model=model, messages=messages, **llm_args)

    try:
        data = json.loads(assistant_message.content)
        # Minimal schema sanity
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
        # On judge failure, return empty shifts to avoid breaking metrics
        return {"start_goal": {"turn": None, "goal": None}, "user_goal_shifts": []}
