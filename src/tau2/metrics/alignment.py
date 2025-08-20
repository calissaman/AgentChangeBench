"""
Alignment scoring for goal-based metrics.

Implements alignment scoring for both tool-required and no-tool goals
using various detectors and scoring mechanisms.
"""

import re
from typing import Dict, List, Optional, Any
from tau2.meta.schema import GoalToken
from tau2.config.goal_map import GOAL_TOOL_MAP


class AlignmentDetectors:
    """
    Stub implementations of alignment detectors.
    These can be upgraded later with more sophisticated implementations.
    """

    def intent_prob(self, text: str, goal: GoalToken) -> float:
        """
        Calculate intent probability using keyword baseline.

        Args:
            text: Message text to analyze
            goal: Target goal token

        Returns:
            Intent probability score 0.0-1.0
        """
        if not text:
            return 0.0

        text_lower = text.lower()
        goal_config = GOAL_TOOL_MAP.get(goal, {})
        description = goal_config.get("description", "").lower()

        # Simple keyword matching based on goal description and requirements
        score = 0.0
        keywords = []

        # Extract keywords from description
        if description:
            keywords.extend(description.split())

        # Add domain-specific keywords based on goal
        if goal == GoalToken.auth_access:
            keywords.extend(["login", "authenticate", "verify", "password", "username"])
        elif goal == GoalToken.transactions_review:
            keywords.extend(["transaction", "history", "activity", "charges"])
        elif goal == GoalToken.dispute_tx:
            keywords.extend(["dispute", "fraud", "unauthorized", "suspicious"])
        elif goal == GoalToken.transfers:
            keywords.extend(["transfer", "send", "move", "funds"])
        elif goal == GoalToken.billpay:
            keywords.extend(["payment", "bill", "pay", "autopay"])
        elif goal == GoalToken.card_services:
            keywords.extend(["card", "block", "activate", "replace", "lost"])
        elif goal == GoalToken.account_info:
            keywords.extend(["account", "balance", "information", "details"])
        elif goal == GoalToken.statements:
            keywords.extend(["statement", "document", "download", "pdf"])
        elif goal == GoalToken.product_info:
            keywords.extend(["product", "service", "offer", "apply"])
        elif goal == GoalToken.policy_explain_reg_e:
            keywords.extend(["regulation", "reg_e", "policy", "rights", "liability"])

        # Score based on keyword matches
        if keywords:
            matches = sum(1 for kw in keywords if kw.lower() in text_lower)
            score = min(1.0, matches / max(len(keywords), 1))

        return score

    def rubric_coverage(self, text: str, rubric: Dict[str, str]) -> float:
        """
        Calculate rubric coverage for no-tool goals.

        Args:
            text: Message text to analyze
            rubric: Rubric with criteria and descriptions

        Returns:
            Coverage score 0.0-1.0
        """
        if not rubric or not text:
            return 0.0

        text_lower = text.lower()
        covered_criteria = 0

        for criterion, description in rubric.items():
            # Extract keywords from criterion and description
            criterion_keywords = criterion.replace("_", " ").split()
            desc_keywords = description.lower().split()
            all_keywords = criterion_keywords + desc_keywords

            # Check if any keywords are present
            if any(kw.lower() in text_lower for kw in all_keywords):
                covered_criteria += 1

        return covered_criteria / len(rubric) if rubric else 0.0

    def policy_hooks(self, text: str, hooks: List[str]) -> float:
        """
        Calculate policy hook coverage.

        Args:
            text: Message text to analyze
            hooks: List of policy hooks to check for

        Returns:
            Policy coverage score 0.0-1.0
        """
        if not hooks or not text:
            return 0.0

        text_lower = text.lower()
        matched_hooks = 0

        for hook in hooks:
            # Simple keyword matching for policy hooks
            hook_keywords = hook.replace("_", " ").split()
            if any(kw.lower() in text_lower for kw in hook_keywords):
                matched_hooks += 1

        return matched_hooks / len(hooks) if hooks else 0.0

    def slot_coverage(self, text: str, required_slots: List[str]) -> float:
        """
        Calculate required slot coverage.

        Args:
            text: Message text to analyze
            required_slots: List of required slots/fields

        Returns:
            Slot coverage score 0.0-1.0
        """
        if not required_slots or not text:
            return 0.0

        text_lower = text.lower()
        covered_slots = 0

        for slot in required_slots:
            # Check if slot is mentioned or prompted for
            slot_variations = [
                slot.lower(),
                slot.replace("_", " ").lower(),
                slot.replace("_", "").lower(),
            ]

            if any(var in text_lower for var in slot_variations):
                covered_slots += 1

        return covered_slots / len(required_slots) if required_slots else 0.0

    def next_steps_or_slots(self, text: str, goal: GoalToken) -> float:
        """
        Calculate next steps or slot prompting score.

        Args:
            text: Message text to analyze
            goal: Target goal

        Returns:
            Next steps score 0.0-1.0
        """
        if not text:
            return 0.0

        text_lower = text.lower()

        # Look for next step indicators
        next_step_patterns = [
            r"what .*(?:would|do) you",
            r"(?:can|may) (?:i|you)",
            r"(?:need|require)",
            r"next step",
            r"how (?:can|do)",
            r"please (?:provide|enter|confirm)",
        ]

        matches = sum(
            1 for pattern in next_step_patterns if re.search(pattern, text_lower)
        )

        # Normalize by number of patterns
        return min(1.0, matches / len(next_step_patterns))


def alignment_score(
    msg: Any,
    new_goal: GoalToken,
    tool_map: Dict[GoalToken, Dict],
    detectors: AlignmentDetectors,
) -> float:
    """
    Calculate alignment score for a message against a target goal.

    Uses different scoring mechanisms for tool-required vs no-tool goals.

    Args:
        msg: Message object with text and tools OR dict with 'content' and 'tool_calls'
        new_goal: Target goal to align with
        tool_map: Mapping of goals to their configurations
        detectors: Alignment detector instance

    Returns:
        Alignment score 0.0-1.0
    """
    # Handle both message objects and dict formats
    if isinstance(msg, dict):
        content = msg.get("content")
        tool_calls = msg.get("tool_calls", [])
    else:
        content = getattr(msg, "content", None)
        tool_calls = getattr(msg, "tool_calls", None) or []

    if not content:
        return 0.0

    goal_config = tool_map.get(new_goal, {})
    requires_tool = goal_config.get("requires_tool", False)

    # Calculate base components
    intent = detectors.intent_prob(content, new_goal)
    hooks = detectors.policy_hooks(content, goal_config.get("policy_hooks", []))

    # Tool usage score
    tool_names = []
    if tool_calls:
        for tc in tool_calls:
            if isinstance(tc, dict) and "name" in tc:
                tool_names.append(tc["name"])
            elif hasattr(tc, "name"):
                tool_names.append(tc.name)
            else:
                tool_names.append(str(tc))

    allowed_tools = goal_config.get("allowed_tools", [])
    tool_score = (
        1.0 if (tool_names and any(t in allowed_tools for t in tool_names)) else 0.0
    )

    if requires_tool:
        # Tool-required goal scoring
        slots = detectors.slot_coverage(content, goal_config.get("required_slots", []))
        score = 0.35 * intent + 0.35 * tool_score + 0.20 * slots + 0.10 * hooks
        return score
    else:
        # No-tool goal scoring
        rubric = goal_config.get("rubric", {})
        rubric_score = detectors.rubric_coverage(content, rubric)
        next_steps = detectors.next_steps_or_slots(content, new_goal)

        # Abstention bonus for no-tool goals (not using tools when not required)
        abstain_score = 1.0 if not tool_names else 0.0

        score = (
            0.40 * intent
            + 0.30 * rubric_score
            + 0.20 * hooks
            + 0.10 * next_steps
            + 0.10 * abstain_score
        )
        return score


def calculate_tool_alignment(
    tool_calls: List[Any], goal: GoalToken, tool_map: Dict[GoalToken, Dict]
) -> float:
    """
    Calculate tool alignment score for a goal.

    Args:
        tool_calls: List of tool call objects
        goal: Target goal
        tool_map: Goal to tools mapping

    Returns:
        Tool alignment score 0.0-1.0
    """
    if not tool_calls:
        return 0.0

    goal_config = tool_map.get(goal, {})
    allowed_tools = goal_config.get("allowed_tools", [])

    if not allowed_tools:
        return 0.0

    # Extract tool names
    tool_names = []
    for tc in tool_calls:
        if hasattr(tc, "name"):
            tool_names.append(tc.name)
        elif isinstance(tc, dict) and "name" in tc:
            tool_names.append(tc["name"])
        else:
            tool_names.append(str(tc))

    # Calculate alignment based on correct tool usage
    correct_tools = sum(1 for tool in tool_names if tool in allowed_tools)
    return min(1.0, correct_tools / len(allowed_tools)) if allowed_tools else 0.0
