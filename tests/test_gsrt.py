"""
Unit tests for GSRT calculation with meta-tags v2 system.
"""

import pytest
from datetime import datetime, timedelta
from tau2.metrics.goal_shift import compute_gsrt, calculate_gsrt_statistics
from tau2.metrics.alignment import AlignmentDetectors
from tau2.config.goal_map import GOAL_TOOL_MAP
from tau2.meta.schema import GoalToken, ShiftReason, MetaEvent


class TestGSRTCalculation:
    """Test GSRT calculation with the new meta system."""

    def test_simple_goal_shift_recovery(self):
        """Test basic goal shift and recovery calculation."""
        # Create mock run data
        base_time = datetime.now()

        # Mock MetaEvent for goal shift
        goal_shift_event = MetaEvent(
            event="GOAL_SHIFT",
            seq=1,
            from_=GoalToken.auth_access,
            to=GoalToken.transactions_review,
            reason=ShiftReason.MANUAL,
        )

        turns = [
            {
                "role": "user",
                "content": "Show me recent transactions",
                "meta_event": goal_shift_event,
                "ts": base_time,
                "tool_calls": [],
            },
            {
                "role": "assistant",
                "content": "I'll help you review your transaction history",
                "ts": base_time + timedelta(seconds=30),
                "tool_calls": [{"name": "get_transactions"}],  # Correct tool for goal
            },
        ]

        run = {"turns": turns}
        detectors = AlignmentDetectors()

        results = compute_gsrt(
            run, GOAL_TOOL_MAP, detectors, theta=0.5
        )  # Lower threshold for test

        assert len(results) == 1
        result = results[0]
        assert result["seq"] == 1
        assert result["goal"] == GoalToken.transactions_review.value
        assert result["gsrt_turns"] == 1  # Recovery after 1 turn
        assert result["gsrt_seconds"] == 30.0  # 30 seconds wall clock time
        assert result["aligned_score"] is not None

    def test_no_recovery_found(self):
        """Test case where no recovery is found within threshold."""
        goal_shift_event = MetaEvent(
            event="GOAL_SHIFT",
            seq=1,
            from_=GoalToken.auth_access,
            to=GoalToken.transactions_review,
            reason=ShiftReason.MANUAL,
        )

        turns = [
            {
                "role": "user",
                "content": "Show me transactions",
                "meta_event": goal_shift_event,
                "ts": datetime.now(),
                "tool_calls": [],
            },
            {
                "role": "assistant",
                "content": "I can help with general banking",  # Poor alignment
                "ts": datetime.now() + timedelta(seconds=30),
                "tool_calls": [],  # No relevant tools
            },
        ]

        run = {"turns": turns}
        detectors = AlignmentDetectors()

        results = compute_gsrt(run, GOAL_TOOL_MAP, detectors, theta=0.7)

        assert len(results) == 1
        result = results[0]
        assert result["gsrt_turns"] is None
        assert result["gsrt_seconds"] is None
        assert result["aligned_score"] is None

    def test_tool_goal_alignment(self):
        """Test alignment scoring for tool-required goals."""
        goal_shift_event = MetaEvent(
            event="GOAL_SHIFT",
            seq=1,
            from_=GoalToken.auth_access,
            to=GoalToken.dispute_tx,
            reason=ShiftReason.MANUAL,
        )

        turns = [
            {
                "role": "user",
                "content": "I want to dispute a charge",
                "meta_event": goal_shift_event,
                "ts": datetime.now(),
                "tool_calls": [],
            },
            {
                "role": "assistant",
                "content": "I'll help you file a dispute for that transaction",
                "ts": datetime.now() + timedelta(seconds=45),
                "tool_calls": [{"name": "file_dispute"}],  # Correct tool for dispute
            },
        ]

        run = {"turns": turns}
        detectors = AlignmentDetectors()

        results = compute_gsrt(run, GOAL_TOOL_MAP, detectors, theta=0.5)

        assert len(results) == 1
        result = results[0]
        assert result["gsrt_turns"] == 1
        assert (
            abs(result["gsrt_seconds"] - 45.0) < 0.1
        )  # Allow for floating point precision
        assert result["goal"] == GoalToken.dispute_tx.value

    def test_no_tool_goal_alignment(self):
        """Test alignment scoring for no-tool goals."""
        goal_shift_event = MetaEvent(
            event="GOAL_SHIFT",
            seq=1,
            from_=GoalToken.auth_access,
            to=GoalToken.policy_explain_reg_e,
            reason=ShiftReason.MANUAL,
        )

        turns = [
            {
                "role": "user",
                "content": "Can you explain Regulation E?",
                "meta_event": goal_shift_event,
                "ts": datetime.now(),
                "tool_calls": [],
            },
            {
                "role": "assistant",
                "content": "Regulation E covers electronic fund transfer rights and liability limits for unauthorized transactions. You have specific rights and reporting timelines...",
                "ts": datetime.now() + timedelta(seconds=20),
                "tool_calls": [],  # No tools needed for policy explanation
            },
        ]

        run = {"turns": turns}
        detectors = AlignmentDetectors()

        results = compute_gsrt(run, GOAL_TOOL_MAP, detectors, theta=0.4)

        assert len(results) == 1
        result = results[0]
        assert result["gsrt_turns"] == 1
        assert result["goal"] == GoalToken.policy_explain_reg_e.value

    def test_multiple_goal_shifts(self):
        """Test handling of multiple goal shifts in one conversation."""
        shift1 = MetaEvent(
            event="GOAL_SHIFT",
            seq=1,
            from_=GoalToken.auth_access,
            to=GoalToken.account_info,
            reason=ShiftReason.MANUAL,
        )

        shift2 = MetaEvent(
            event="GOAL_SHIFT",
            seq=2,
            from_=GoalToken.account_info,
            to=GoalToken.transfers,
            reason=ShiftReason.MANUAL,
        )

        base_time = datetime.now()

        turns = [
            {
                "role": "user",
                "content": "What's my balance?",
                "meta_event": shift1,
                "ts": base_time,
                "tool_calls": [],
            },
            {
                "role": "assistant",
                "content": "Your current account balance is $1,234.56",
                "ts": base_time + timedelta(seconds=10),
                "tool_calls": [{"name": "get_balance"}],
            },
            {
                "role": "user",
                "content": "I want to transfer money",
                "meta_event": shift2,
                "ts": base_time + timedelta(seconds=60),
                "tool_calls": [],
            },
            {
                "role": "assistant",
                "content": "I can help you transfer funds between accounts",
                "ts": base_time + timedelta(seconds=75),
                "tool_calls": [{"name": "transfer_funds"}],
            },
        ]

        run = {"turns": turns}
        detectors = AlignmentDetectors()

        results = compute_gsrt(run, GOAL_TOOL_MAP, detectors, theta=0.3)

        assert len(results) == 2

        # First shift recovery
        assert results[0]["seq"] == 1
        assert results[0]["goal"] == GoalToken.account_info.value
        assert results[0]["gsrt_turns"] == 1
        assert results[0]["gsrt_seconds"] == 10.0

        # Second shift recovery
        assert results[1]["seq"] == 2
        assert results[1]["goal"] == GoalToken.transfers.value
        assert results[1]["gsrt_turns"] == 1
        assert results[1]["gsrt_seconds"] == 15.0

    def test_invalid_goal_handling(self):
        """Test handling of invalid goal tokens."""
        # Meta event with invalid goal
        invalid_event = {
            "event": "GOAL_SHIFT",
            "seq": 1,
            "from": "auth_access",
            "to": "invalid_goal",  # Invalid goal token
            "reason": "MANUAL",
        }

        turns = [
            {
                "role": "user",
                "content": "Test message",
                "meta_event": invalid_event,
                "ts": datetime.now(),
                "tool_calls": [],
            }
        ]

        run = {"turns": turns}
        detectors = AlignmentDetectors()

        results = compute_gsrt(run, GOAL_TOOL_MAP, detectors)

        assert len(results) == 1
        result = results[0]
        assert result["error"] == "INVALID_GOAL:invalid_goal"
        assert result["gsrt_turns"] is None

    def test_missing_goal_handling(self):
        """Test handling of goal shifts with missing goal."""
        incomplete_event = MetaEvent(
            event="GOAL_SHIFT",
            seq=1,
            reason=ShiftReason.MANUAL,
            # Missing from and to
        )

        turns = [
            {
                "role": "user",
                "content": "Test message",
                "meta_event": incomplete_event,
                "ts": datetime.now(),
                "tool_calls": [],
            }
        ]

        run = {"turns": turns}
        detectors = AlignmentDetectors()

        results = compute_gsrt(run, GOAL_TOOL_MAP, detectors)

        assert len(results) == 1
        result = results[0]
        assert result["error"] == "MISSING_GOAL"


class TestGSRTStatistics:
    """Test GSRT statistics calculation."""

    def test_empty_results_statistics(self):
        """Test statistics calculation with no results."""
        stats = calculate_gsrt_statistics([])

        assert stats["total_shifts"] == 0
        assert stats["successful_recoveries"] == 0
        assert stats["median_gsrt_turns"] is None
        assert stats["recovery_rate"] == 0.0

    def test_mixed_results_statistics(self):
        """Test statistics with mix of successful and failed recoveries."""
        results = [
            {"seq": 1, "gsrt_turns": 1, "gsrt_seconds": 10.0},
            {"seq": 2, "gsrt_turns": None, "gsrt_seconds": None},  # Failed recovery
            {"seq": 3, "gsrt_turns": 3, "gsrt_seconds": 45.0},
            {"seq": 4, "gsrt_turns": 2, "gsrt_seconds": 30.0},
        ]

        stats = calculate_gsrt_statistics(results)

        assert stats["total_shifts"] == 4
        assert stats["successful_recoveries"] == 3
        assert stats["median_gsrt_turns"] == 2.0  # Median of [1, 3, 2]
        assert stats["median_gsrt_seconds"] == 30.0  # Median of [10, 45, 30]
        assert stats["worst_case_gsrt_turns"] == 3
        assert stats["worst_case_gsrt_seconds"] == 45.0
        assert stats["recovery_rate"] == 0.75  # 3/4

    def test_all_successful_statistics(self):
        """Test statistics with all successful recoveries."""
        results = [
            {"seq": 1, "gsrt_turns": 1, "gsrt_seconds": 15.0},
            {"seq": 2, "gsrt_turns": 2, "gsrt_seconds": 25.0},
        ]

        stats = calculate_gsrt_statistics(results)

        assert stats["total_shifts"] == 2
        assert stats["successful_recoveries"] == 2
        assert stats["recovery_rate"] == 1.0
