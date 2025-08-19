"""
Unit tests for meta-tags v2 legacy migration.
"""

import pytest
from tau2.meta.migrations import migrate_legacy_meta, is_legacy_meta_line, _map_topic_to_goal_token
from tau2.meta.grammar import validate_max_length, is_meta_line
from tau2.meta.schema import GoalToken, ShiftReason


class TestLegacyMigration:
    """Test cases for migrating legacy meta tags."""
    
    def test_basic_goal_shift_migration(self):
        """Test migration of basic GOAL_SHIFT without topic."""
        line = '<meta>GOAL_SHIFT</meta>'
        meta_dict, error = migrate_legacy_meta(line)
        
        assert meta_dict is not None
        assert meta_dict["event"] == "GOAL_SHIFT"
        assert meta_dict["reason"] == ShiftReason.MANUAL.value
        assert error == "MISSING_FIELDS:seq,from,to"  # Required fields missing
    
    def test_goal_shift_with_topic_migration(self):
        """Test migration of GOAL_SHIFT with topic."""
        line = '<meta>GOAL_SHIFT:transactions</meta>'
        meta_dict, error = migrate_legacy_meta(line)
        
        assert meta_dict is not None
        assert meta_dict["event"] == "GOAL_SHIFT"
        assert meta_dict["to"] == GoalToken.transactions_review.value
        assert meta_dict["reason"] == ShiftReason.MANUAL.value
        assert error == "MISSING_FIELDS:seq,from"  # Still missing seq and from
    
    def test_unknown_legacy_topic(self):
        """Test handling of unknown legacy topics."""
        line = '<meta>GOAL_SHIFT:unknown_topic</meta>'
        meta_dict, error = migrate_legacy_meta(line)
        
        assert meta_dict is not None
        assert meta_dict["event"] == "GOAL_SHIFT"
        assert error == "UNKNOWN_LEGACY_TOPIC:unknown_topic"
    
    def test_topic_mapping_accuracy(self):
        """Test accuracy of topic to goal token mapping."""
        test_cases = [
            ("transaction_dispute", GoalToken.dispute_tx),
            ("transactions", GoalToken.transactions_review),
            ("account_balance", GoalToken.account_info),
            ("bill_payment", GoalToken.billpay),
            ("card_services", GoalToken.card_services),
            ("transfers", GoalToken.transfers),
            ("statements", GoalToken.statements),
            ("auth", GoalToken.auth_access),
            ("authentication", GoalToken.auth_access),
            ("product_info", GoalToken.product_info),
            ("policy_explain_reg_e", GoalToken.policy_explain_reg_e),
        ]
        
        for topic, expected_goal in test_cases:
            result = _map_topic_to_goal_token(topic)
            assert result == expected_goal, f"Failed to map {topic} to {expected_goal}"
    
    def test_fuzzy_topic_mapping(self):
        """Test fuzzy matching for topic variations."""
        fuzzy_cases = [
            ("dispute charge", GoalToken.dispute_tx),
            ("fraud report", GoalToken.dispute_tx),
            ("transaction history", GoalToken.transactions_review),
            ("account details", GoalToken.account_info),
            ("card replacement", GoalToken.card_services),
            ("fund transfer", GoalToken.transfers),
            ("monthly statement", GoalToken.statements),
            ("login help", GoalToken.auth_access),
            ("product comparison", GoalToken.product_info),
            ("regulation policy", GoalToken.policy_explain_reg_e),
        ]
        
        for topic, expected_goal in fuzzy_cases:
            result = _map_topic_to_goal_token(topic)
            assert result == expected_goal, f"Failed to fuzzy map {topic} to {expected_goal}"
    
    def test_unmappable_topic(self):
        """Test handling of topics that can't be mapped."""
        unmappable_topics = [
            "completely_unknown",
            "random_text",
            "xyz123",
            ""
        ]
        
        for topic in unmappable_topics:
            result = _map_topic_to_goal_token(topic)
            assert result is None, f"Unexpectedly mapped {topic} to {result}"
    
    def test_is_legacy_meta_line_detection(self):
        """Test detection of legacy meta lines."""
        # Valid legacy patterns
        assert is_legacy_meta_line('<meta>GOAL_SHIFT</meta>') is True
        assert is_legacy_meta_line('<meta>GOAL_SHIFT:transactions</meta>') is True
        assert is_legacy_meta_line('  <meta>GOAL_SHIFT:auth</meta>  ') is True
        
        # Invalid patterns
        assert is_legacy_meta_line('<meta>PARK seq=1</meta>') is False
        assert is_legacy_meta_line('<meta>GOAL_SHIFT seq=1</meta>') is False
        assert is_legacy_meta_line('Not a meta line') is False
        assert is_legacy_meta_line('<meta>OTHER_EVENT</meta>') is False
    
    def test_no_legacy_meta_case(self):
        """Test handling when no legacy meta is found."""
        line = 'Regular message content'
        meta_dict, error = migrate_legacy_meta(line)
        
        assert meta_dict is None
        assert error == "NO_LEGACY_META"
    
    def test_whitespace_in_legacy_topic(self):
        """Test handling of whitespace in legacy topics."""
        line = '<meta>GOAL_SHIFT: transaction dispute </meta>'
        meta_dict, error = migrate_legacy_meta(line)
        
        assert meta_dict is not None
        assert meta_dict["to"] == GoalToken.dispute_tx.value
    
    def test_case_insensitive_topic_mapping(self):
        """Test case insensitive topic mapping."""
        line = '<meta>GOAL_SHIFT:TRANSACTIONS</meta>'
        meta_dict, error = migrate_legacy_meta(line)
        
        assert meta_dict is not None
        assert meta_dict["to"] == GoalToken.transactions_review.value


class TestMetaGrammarValidation:
    """Test validation functions."""
    
    def test_max_length_validation(self):
        """Test max length validation function."""
        short_line = '<meta>GOAL_SHIFT seq=1</meta>'
        assert validate_max_length(short_line) is True
        
        # Exactly at boundary
        base = '<meta>GOAL_SHIFT '
        ending = '</meta>'
        padding_needed = 256 - len(base) - len(ending)
        line_256 = base + 'x' * padding_needed + ending
        assert len(line_256) == 256
        assert validate_max_length(line_256) is True
        
        # Over boundary
        line_257 = base + 'x' * (padding_needed + 1) + ending
        assert len(line_257) == 257
        assert validate_max_length(line_257) is False
    
    def test_is_meta_line_utility(self):
        """Test is_meta_line utility function."""
        valid_cases = [
            '<meta>GOAL_SHIFT</meta>',
            '<meta>PARK seq=1</meta>',
            '  <meta>RESUME task=test</meta>  ',
        ]
        
        for case in valid_cases:
            assert is_meta_line(case) is True
            
        invalid_cases = [
            'Not meta',
            '<meta>incomplete',
            'incomplete</meta>',
            '<other>tag</other>',
            '',
        ]
        
        for case in invalid_cases:
            assert is_meta_line(case) is False 