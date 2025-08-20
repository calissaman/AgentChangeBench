"""
GSRT (Goal Shift Recovery Time) calculator for AgentChangeBench.

Implements Phase 2: Goal shift detection + keyword-based alignment scoring.
"""

import re
import statistics
from typing import List, Optional, Tuple

from tau2.data_model.gsrt_models import GoalShift, GSRTResult, GSRTConfig
from tau2.data_model.message import Message
from tau2.data_model.simulation import SimulationRun
from tau2.data_model.tasks import Task


def detect_goal_shifts(messages: List[Message]) -> List[GoalShift]:
    """
    Detect user-initiated GOAL_SHIFT events.
    - Prefer parsed v2 meta from message.meta_event
    - Fallback: parse first line of original_content/content with v2 grammar
    - Count only user turns
    """
    goal_shifts: List[GoalShift] = []

    # Lazy import to avoid circulars
    from tau2.meta.grammar import parse_meta_line

    for i, msg in enumerate(messages):
        # Only user turns initiate shifts for GSRT
        if getattr(msg, "role", None) != "user":
            continue

        # 1) Prefer parsed meta_event if present
        meta = getattr(msg, "meta_event", None)
        if meta is not None:
            try:
                # meta may be a dict or a MetaEvent instance
                event = (
                    meta.get("event")
                    if isinstance(meta, dict)
                    else getattr(meta, "event", None)
                )
                if event == "GOAL_SHIFT":
                    goal_shifts.append(
                        GoalShift(
                            turn_index=i,
                            shift_type="GOAL_SHIFT",
                            metadata={
                                "message_role": msg.role,
                                "turn_idx": getattr(msg, "turn_idx", i),
                                "raw": getattr(meta, "raw", None)
                                if not isinstance(meta, dict)
                                else meta.get("raw"),
                            },
                        )
                    )
                    continue
            except Exception:
                # If malformed, fall through to parsing fallback
                pass

        # 2) Fallback: parse the first line of original_content/content
        raw_text = (
            getattr(msg, "original_content", None)
            or getattr(msg, "content", None)
            or ""
        )
        first_line = raw_text.splitlines()[0].strip() if raw_text else ""
        if first_line.startswith("<meta>") and first_line.endswith("</meta>"):
            parsed, err = parse_meta_line(first_line)
            if parsed is not None and parsed.get("event") == "GOAL_SHIFT":
                goal_shifts.append(
                    GoalShift(
                        turn_index=i,
                        shift_type="GOAL_SHIFT",
                        metadata={
                            "message_role": msg.role,
                            "turn_idx": getattr(msg, "turn_idx", i),
                            "raw": first_line,
                            "parse_error": err,
                        },
                    )
                )

    return goal_shifts


def infer_goal_from_context(
    messages: List[Message], turn_index: int, config: GSRTConfig, window_size: int = 3
) -> Optional[str]:
    """
    Infer the goal at a specific turn by analyzing surrounding context.

    Args:
        messages: Conversation messages
        turn_index: Turn to analyze
        config: GSRT configuration with goal keywords
        window_size: Number of messages before/after to consider

    Returns:
        Inferred goal name or None
    """
    # Get context window around the turn
    start_idx = max(0, turn_index - window_size)
    end_idx = min(len(messages), turn_index + window_size + 1)
    context_messages = messages[start_idx:end_idx]

    # Combine content from context messages
    context_text = " ".join(
        [msg.content.lower() if msg.content else "" for msg in context_messages]
    )

    # Score each goal based on keyword matches
    goal_scores = {}
    for goal_name, keywords in config.goal_keywords.items():
        score = 0
        for keyword in keywords:
            # Count keyword occurrences (case-insensitive)
            score += len(
                re.findall(r"\b" + re.escape(keyword.lower()) + r"\b", context_text)
            )
        goal_scores[goal_name] = score

    # Return goal with highest score (if any matches)
    if goal_scores and max(goal_scores.values()) > 0:
        return max(goal_scores, key=goal_scores.get)

    return None


def calculate_alignment_score_keywords(
    agent_response: str, target_goal: str, config: GSRTConfig
) -> float:
    """
    Calculate alignment score based on keyword overlap between response and goal.

    Args:
        agent_response: Agent's response text
        target_goal: Target goal name
        config: GSRT configuration with goal keywords

    Returns:
        Alignment score 0.0-1.0 (1.0 = perfect alignment)
    """
    if not agent_response or target_goal not in config.goal_keywords:
        return 0.0

    response_text = agent_response.lower()
    goal_keywords = config.goal_keywords[target_goal]

    # Count keyword matches
    matches = 0
    for keyword in goal_keywords:
        if re.search(r"\b" + re.escape(keyword.lower()) + r"\b", response_text):
            matches += 1

    # Normalize by number of keywords for this goal
    if len(goal_keywords) == 0:
        return 0.0

    alignment_score = matches / len(goal_keywords)
    return min(1.0, alignment_score)  # Cap at 1.0


def find_recovery_turn(
    messages: List[Message], shift_turn: int, target_goal: str, config: GSRTConfig
) -> Optional[int]:
    """
    Find the turn where the agent recovers and aligns with the new goal.

    Args:
        messages: Conversation messages
        shift_turn: Turn index where goal shift occurred
        target_goal: Target goal to align with
        config: GSRT configuration

    Returns:
        Turn index where recovery occurs, or None if no recovery found
    """
    max_search_turn = min(len(messages), shift_turn + config.max_recovery_window)

    for turn_idx in range(shift_turn + 1, max_search_turn):
        if turn_idx >= len(messages):
            break

        msg = messages[turn_idx]

        # Only check agent responses for alignment
        if msg.role != "assistant" or not msg.content:
            continue

        alignment_score = calculate_alignment_score_keywords(
            msg.content, target_goal, config
        )

        if alignment_score >= config.alignment_threshold:
            return turn_idx

    return None


def compute_gsrt_for_simulation(
    simulation: SimulationRun, task: Task, config: Optional[GSRTConfig] = None
) -> GSRTResult:
    """
    Compute GSRT for a single simulation.

    Args:
        simulation: Simulation run data
        task: Task information
        config: GSRT configuration (uses default if None)

    Returns:
        GSRT results with recovery times and statistics
    """
    if config is None:
        config = GSRTConfig()

    messages = simulation.messages

    # Step 1: Detect goal shifts
    goal_shifts = detect_goal_shifts(messages)

    if not goal_shifts:
        return GSRTResult(
            goal_shifts=[],
            recovery_times=[],
            median_gsrt=None,
            worst_case_gsrt=None,
            num_shifts=0,
            alignment_threshold=config.alignment_threshold,
        )

    # Step 2: Infer goals and calculate recovery times
    recovery_times = []

    for shift in goal_shifts:
        # Infer the new goal from context after the shift
        new_goal = infer_goal_from_context(
            messages, shift.turn_index, config, window_size=3
        )

        if new_goal:
            shift.new_goal = new_goal

            # Find recovery turn
            recovery_turn = find_recovery_turn(
                messages, shift.turn_index, new_goal, config
            )

            if recovery_turn is not None:
                recovery_time = recovery_turn - shift.turn_index
                recovery_times.append(recovery_time)

    # Step 3: Calculate statistics
    median_gsrt = None
    worst_case_gsrt = None

    if recovery_times:
        median_gsrt = statistics.median(recovery_times)
        worst_case_gsrt = max(recovery_times)

    return GSRTResult(
        goal_shifts=goal_shifts,
        recovery_times=recovery_times,
        median_gsrt=median_gsrt,
        worst_case_gsrt=worst_case_gsrt,
        num_shifts=len(goal_shifts),
        alignment_threshold=config.alignment_threshold,
    )


def aggregate_gsrt_results(
    gsrt_results: List[GSRTResult],
) -> Tuple[Optional[float], Optional[int], int]:
    """
    Aggregate GSRT results across multiple simulations.

    Args:
        gsrt_results: List of GSRT results from multiple simulations

    Returns:
        Tuple of (median_gsrt, worst_case_gsrt, total_num_shifts)
    """
    all_recovery_times = []
    total_shifts = 0

    for result in gsrt_results:
        all_recovery_times.extend(result.recovery_times)
        total_shifts += result.num_shifts

    if not all_recovery_times:
        return None, None, total_shifts

    median_gsrt = statistics.median(all_recovery_times)
    worst_case_gsrt = max(all_recovery_times)

    return median_gsrt, worst_case_gsrt, total_shifts
