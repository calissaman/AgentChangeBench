"""
Unit tests for meta-tags v2 grammar parsing.
"""

import pytest
from tau2.meta.grammar import parse_meta_line, validate_max_length, is_meta_line


class TestMetaGrammarParsing:
    """Test cases for the deterministic meta grammar."""
    
    def test_valid_goal_shift_full(self):
        """Test parsing a valid GOAL_SHIFT with all required fields."""
        line = '<meta>GOAL_SHIFT seq=1 from=auth_access to=transactions_review reason=MANUAL</meta>'
        meta_dict, error = parse_meta_line(line)
        
        assert error is None
        assert meta_dict is not None
        assert meta_dict["event"] == "GOAL_SHIFT"
        assert meta_dict["seq"] == "1"
        assert meta_dict["from"] == "auth_access"
        assert meta_dict["to"] == "transactions_review"
        assert meta_dict["reason"] == "MANUAL"
    
    def test_valid_park_event(self):
        """Test parsing a valid PARK event."""
        line = '<meta>PARK seq=2 task=card_services note="awaiting MFA sms"</meta>'
        meta_dict, error = parse_meta_line(line)
        
        assert error is None
        assert meta_dict is not None
        assert meta_dict["event"] == "PARK"
        assert meta_dict["seq"] == "2"
        assert meta_dict["task"] == "card_services"
        assert meta_dict["note"] == "awaiting MFA sms"
    
    def test_valid_resume_event(self):
        """Test parsing a valid RESUME event."""
        line = '<meta>RESUME seq=2 task=card_services</meta>'
        meta_dict, error = parse_meta_line(line)
        
        assert error is None
        assert meta_dict is not None
        assert meta_dict["event"] == "RESUME"
        assert meta_dict["seq"] == "2"
        assert meta_dict["task"] == "card_services"
    
    def test_quoted_values(self):
        """Test parsing with quoted values containing spaces."""
        line = '<meta>PARK seq=1 task="complex task name" note="waiting for user response"</meta>'
        meta_dict, error = parse_meta_line(line)
        
        assert error is None
        assert meta_dict is not None
        assert meta_dict["task"] == "complex task name"
        assert meta_dict["note"] == "waiting for user response"
    
    def test_invalid_event_type(self):
        """Test rejection of invalid event types."""
        line = '<meta>INVALID_EVENT seq=1</meta>'
        meta_dict, error = parse_meta_line(line)
        
        assert meta_dict is None
        assert error == "INVALID_EVENT:INVALID_EVENT"
    
    def test_invalid_key(self):
        """Test rejection of invalid keys."""
        line = '<meta>GOAL_SHIFT seq=1 invalid_key=value</meta>'
        meta_dict, error = parse_meta_line(line)
        
        assert meta_dict is None
        assert error == "INVALID_KEY:invalid_key"
    
    def test_bad_key_value_format(self):
        """Test rejection of malformed key-value pairs."""
        line = '<meta>GOAL_SHIFT seq=1 badformat</meta>'
        meta_dict, error = parse_meta_line(line)
        
        assert meta_dict is None
        assert error == "BAD_KV:badformat"
    
    def test_invalid_token_value(self):
        """Test rejection of invalid token values."""
        line = '<meta>GOAL_SHIFT seq=1 from=invalid@token</meta>'
        meta_dict, error = parse_meta_line(line)
        
        assert meta_dict is None
        assert error == "INVALID_TOKEN_VALUE:invalid@token"
    
    def test_empty_meta_body(self):
        """Test rejection of empty meta body."""
        line = '<meta></meta>'
        meta_dict, error = parse_meta_line(line)
        
        assert meta_dict is None
        assert error == "EMPTY_META_BODY"
    
    def test_not_meta_tag(self):
        """Test rejection of non-meta lines."""
        line = 'This is just regular text'
        meta_dict, error = parse_meta_line(line)
        
        assert meta_dict is None
        assert error == "NO_META_OR_BAD_TAG"
    
    def test_malformed_meta_tag(self):
        """Test rejection of malformed meta tags."""
        line = '<meta>GOAL_SHIFT seq=1'  # Missing closing tag
        meta_dict, error = parse_meta_line(line)
        
        assert meta_dict is None
        assert error == "NO_META_OR_BAD_TAG"
    
    def test_max_length_validation(self):
        """Test maximum length validation."""
        short_line = '<meta>GOAL_SHIFT seq=1</meta>'
        assert validate_max_length(short_line) is True
        
        # Create a line that exceeds 256 characters
        long_value = "x" * 250
        long_line = f'<meta>GOAL_SHIFT seq=1 note="{long_value}"</meta>'
        assert validate_max_length(long_line) is False
    
    def test_is_meta_line_utility(self):
        """Test the utility function for detecting meta lines."""
        assert is_meta_line('<meta>GOAL_SHIFT</meta>') is True
        assert is_meta_line('  <meta>PARK seq=1</meta>  ') is True
        assert is_meta_line('Not a meta line') is False
        assert is_meta_line('<meta>incomplete') is False
    
    def test_whitespace_handling(self):
        """Test proper handling of whitespace."""
        line = '  <meta>GOAL_SHIFT seq=1 from=auth_access to=dispute_tx reason=MANUAL</meta>  '
        meta_dict, error = parse_meta_line(line)
        
        assert error is None
        assert meta_dict is not None
        assert meta_dict["event"] == "GOAL_SHIFT"
    
    def test_complex_quoted_note(self):
        """Test parsing with complex quoted note containing punctuation."""
        line = '<meta>PARK seq=3 task=billpay note="Waiting for MFA code, sent via SMS"</meta>'
        meta_dict, error = parse_meta_line(line)
        
        assert error is None
        assert meta_dict is not None
        assert meta_dict["note"] == "Waiting for MFA code, sent via SMS"


class TestMetaGrammarEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_boundary_length_exactly_256(self):
        """Test parsing at exactly 256 character boundary."""
        # Create content that makes the total exactly 256 chars
        base = '<meta>GOAL_SHIFT seq=1 from=auth_access to=transactions_review reason=MANUAL note="'
        closing = '"</meta>'
        remaining = 256 - len(base) - len(closing)
        note_content = "x" * remaining
        line = base + note_content + closing
        
        assert len(line) == 256
        meta_dict, error = parse_meta_line(line)
        
        # Should succeed at exactly 256 chars
        assert error is None
        assert meta_dict is not None
    
    def test_over_length_boundary(self):
        """Test rejection when over 256 characters."""
        base = '<meta>GOAL_SHIFT seq=1 from=auth_access to=transactions_review reason=MANUAL note="'
        note_content = "x" * 300  # Definitely over limit
        line = base + note_content + '"</meta>'
        
        meta_dict, error = parse_meta_line(line)
        assert meta_dict is None
        assert error == "NO_META_OR_BAD_TAG"  # Regex won't match due to length limit
    
    def test_empty_quoted_value(self):
        """Test handling of empty quoted values."""
        line = '<meta>PARK seq=1 task=billpay note=""</meta>'
        meta_dict, error = parse_meta_line(line)
        
        assert error is None
        assert meta_dict is not None
        assert meta_dict["note"] == ""
    
    def test_minimal_valid_events(self):
        """Test minimal valid events."""
        # Minimal GOAL_SHIFT (though will be invalid due to missing required fields)
        line = '<meta>GOAL_SHIFT seq=1</meta>'
        meta_dict, error = parse_meta_line(line)
        
        assert error is None  # Grammar is valid
        assert meta_dict is not None
        assert meta_dict["event"] == "GOAL_SHIFT"
        assert meta_dict["seq"] == "1"
        
        # Minimal PARK
        line = '<meta>PARK seq=1 task=test</meta>'
        meta_dict, error = parse_meta_line(line)
        
        assert error is None
        assert meta_dict is not None
        assert meta_dict["event"] == "PARK" 