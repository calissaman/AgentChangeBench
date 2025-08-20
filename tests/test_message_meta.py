"""
Unit tests for message integration with meta-tags v2 system.
"""

import pytest
from tau2.data_model.message import UserMessage, AssistantMessage
from tau2.meta.schema import MetaEvent, GoalToken, ShiftReason


class TestMessageMetaIntegration:
    """Test message integration with meta-tags v2."""

    def test_valid_goal_shift_extraction(self):
        """Test extraction of valid GOAL_SHIFT meta tag."""
        content = "<meta>GOAL_SHIFT seq=1 from=auth_access to=transactions_review reason=MANUAL</meta>\nCan you show me yesterday's transactions?"
        message = UserMessage(role="user", content=content)

        message.extract_meta_from_content()

        # Check meta_event is properly populated
        assert message.meta_event is not None
        assert message.meta_event.event == "GOAL_SHIFT"
        assert message.meta_event.seq == 1
        assert message.meta_event.from_ == GoalToken.auth_access
        assert message.meta_event.to == GoalToken.transactions_review
        assert message.meta_event.reason == ShiftReason.MANUAL
        assert (
            message.meta_event.raw
            == "<meta>GOAL_SHIFT seq=1 from=auth_access to=transactions_review reason=MANUAL</meta>"
        )
        assert message.meta_event.is_valid() is True

        # Check content filtering
        assert message.original_content == content
        assert message.content == "Can you show me yesterday's transactions?"
        assert message.display_content == "Can you show me yesterday's transactions?"

        # No error should be present
        assert message.meta_error is None

    def test_invalid_meta_fields_error(self):
        """Test handling of invalid meta fields."""
        content = "<meta>GOAL_SHIFT seq=1</meta>\nMessage content"
        message = UserMessage(role="user", content=content)

        message.extract_meta_from_content()

        # Meta event should be created but marked as invalid
        assert message.meta_event is not None
        assert message.meta_event.event == "GOAL_SHIFT"
        assert message.meta_event.seq == 1
        assert message.meta_event.is_valid() is False

        # Error should be set
        assert message.meta_error == "INVALID_META_FIELDS"

        # Content should still be filtered
        assert message.content == "Message content"

    def test_park_event_extraction(self):
        """Test extraction of PARK event."""
        content = '<meta>PARK seq=2 task=card_services note="awaiting verification"</meta>\nI need to wait for the SMS code.'
        message = UserMessage(role="user", content=content)

        message.extract_meta_from_content()

        assert message.meta_event is not None
        assert message.meta_event.event == "PARK"
        assert message.meta_event.seq == 2
        assert message.meta_event.task == "card_services"
        assert message.meta_event.note == "awaiting verification"
        assert message.meta_event.is_valid() is True
        assert message.meta_error is None

    def test_resume_event_extraction(self):
        """Test extraction of RESUME event."""
        content = (
            "<meta>RESUME seq=2 task=card_services</meta>\nI got the SMS code: 123456"
        )
        message = UserMessage(role="user", content=content)

        message.extract_meta_from_content()

        assert message.meta_event is not None
        assert message.meta_event.event == "RESUME"
        assert message.meta_event.seq == 2
        assert message.meta_event.task == "card_services"
        assert message.meta_event.is_valid() is True

    def test_legacy_migration_integration(self):
        """Test integration with legacy meta tag migration."""
        content = "<meta>GOAL_SHIFT:transactions</meta>\nShow me my recent transactions"
        message = UserMessage(role="user", content=content)

        message.extract_meta_from_content()

        assert message.meta_event is not None
        assert message.meta_event.event == "GOAL_SHIFT"
        assert message.meta_event.to == GoalToken.transactions_review
        assert message.meta_event.reason == ShiftReason.MANUAL
        assert message.meta_event.is_valid() is False  # Missing seq and from
        assert "MISSING_FIELDS" in message.meta_error

    def test_no_meta_tag(self):
        """Test message without meta tag."""
        content = "Just regular message content here"
        message = UserMessage(role="user", content=content)

        message.extract_meta_from_content()

        assert message.meta_event is None
        assert message.meta_error is None
        assert message.content == content
        assert message.display_content == content
        assert message.original_content == content

    def test_stray_meta_tags_removal(self):
        """Test removal of stray meta tags in message body."""
        content = "Regular content with <meta>stray tag</meta> in the middle and <meta>another</meta> at end"
        message = UserMessage(role="user", content=content)

        message.extract_meta_from_content()

        # Stray meta tags should be removed
        assert "<meta>" not in message.content
        assert "</meta>" not in message.content
        assert message.content == "Regular content with  in the middle and  at end"
        assert message.meta_event is None  # No valid meta event from first line

    def test_meta_tag_not_first_line(self):
        """Test that meta tags not on first line are ignored as events."""
        content = "First line of message\n<meta>GOAL_SHIFT seq=1</meta>\nMore content"
        message = UserMessage(role="user", content=content)

        message.extract_meta_from_content()

        # No meta event should be detected (not first line)
        assert message.meta_event is None
        assert message.meta_error is None

        # But the meta tag should be filtered out
        assert "<meta>" not in message.content
        # Note: removing the meta tag leaves an empty line
        assert message.content == "First line of message\n\nMore content"

    def test_malformed_meta_tag_error(self):
        """Test handling of malformed meta tags."""
        content = "<meta>GOAL_SHIFT seq=invalid_number</meta>\nMessage content"
        message = UserMessage(role="user", content=content)

        message.extract_meta_from_content()

        # Should create meta event but with validation error
        # Note: may be None if validation completely fails
        assert "META_VALIDATION_ERROR" in message.meta_error
        assert message.content == "Message content"

    def test_original_content_preservation(self):
        """Test that original content is always preserved."""
        content = "<meta>GOAL_SHIFT seq=1 from=auth_access to=dispute_tx reason=MANUAL</meta>\nOriginal message"
        message = UserMessage(role="user", content=content)

        message.extract_meta_from_content()

        assert message.original_content == content
        assert message.content != content  # Should be filtered
        assert message.display_content == "Original message"

    def test_assistant_message_meta_processing(self):
        """Test meta processing works for assistant messages too."""
        content = "<meta>RESUME seq=1 task=dispute_tx</meta>\nI can help you with that dispute."
        message = AssistantMessage(role="assistant", content=content)

        message.extract_meta_from_content()

        assert message.meta_event is not None
        assert message.meta_event.event == "RESUME"
        assert message.content == "I can help you with that dispute."

    def test_empty_content_handling(self):
        """Test handling of messages with no content."""
        message = UserMessage(role="user", content=None)

        # Should not crash
        message.extract_meta_from_content()

        assert message.meta_event is None
        assert message.meta_error is None
        assert message.original_content is None

    def test_content_cleanup_whitespace(self):
        """Test proper cleanup of whitespace after meta tag removal."""
        content = "<meta>GOAL_SHIFT seq=1 from=auth_access to=billpay reason=MANUAL</meta>\n\n\nMessage with extra whitespace\n\n"
        message = UserMessage(role="user", content=content)

        message.extract_meta_from_content()

        # Whitespace should be cleaned up
        assert message.content == "Message with extra whitespace"
        assert message.display_content == "Message with extra whitespace"


class TestMetaEventValidation:
    """Test MetaEvent validation logic."""

    def test_goal_shift_validation_complete(self):
        """Test GOAL_SHIFT validation with all required fields."""
        meta_event = MetaEvent(
            event="GOAL_SHIFT",
            seq=1,
            from_=GoalToken.auth_access,
            to=GoalToken.transactions_review,
            reason=ShiftReason.MANUAL,
        )

        assert meta_event.is_valid() is True

    def test_goal_shift_validation_missing_fields(self):
        """Test GOAL_SHIFT validation with missing required fields."""
        # Missing 'from' field
        meta_event = MetaEvent(
            event="GOAL_SHIFT",
            seq=1,
            to=GoalToken.transactions_review,
            reason=ShiftReason.MANUAL,
        )

        assert meta_event.is_valid() is False

    def test_park_validation_complete(self):
        """Test PARK validation with required fields."""
        meta_event = MetaEvent(event="PARK", seq=1, task="card_services")

        assert meta_event.is_valid() is True

    def test_park_validation_missing_task(self):
        """Test PARK validation with missing task."""
        meta_event = MetaEvent(event="PARK", seq=1)

        assert meta_event.is_valid() is False

    def test_resume_validation_complete(self):
        """Test RESUME validation with required fields."""
        meta_event = MetaEvent(event="RESUME", seq=1, task="billpay")

        assert meta_event.is_valid() is True

    def test_seq_validation_positive(self):
        """Test that seq must be positive."""
        with pytest.raises(ValueError, match="seq must be >= 1"):
            MetaEvent(
                event="GOAL_SHIFT",
                seq=0,
                from_=GoalToken.auth_access,
                to=GoalToken.transactions_review,
                reason=ShiftReason.MANUAL,
            )
