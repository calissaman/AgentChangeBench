import re
from typing import Dict, Optional, Tuple
from .schema import GoalToken, ShiftReason

# Legacy meta tag patterns
LEGACY_GOAL_SHIFT_RE = re.compile(r'^\s*<meta>GOAL_SHIFT(?::(?P<topic>[^<]*?))?</meta>\s*$', re.ASCII)

def migrate_legacy_meta(line: str) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Convert legacy meta tags to new format.
    
    Handles:
    - <meta>GOAL_SHIFT</meta> → GOAL_SHIFT with missing fields marked as error
    - <meta>GOAL_SHIFT:topic</meta> → GOAL_SHIFT with topic mapped to enum
    
    Args:
        line: The first line to check for legacy patterns
        
    Returns:
        Tuple of (meta_dict, error) similar to parse_meta_line
    """
    line = line.strip()
    
    # Check for legacy GOAL_SHIFT patterns
    legacy_match = LEGACY_GOAL_SHIFT_RE.match(line)
    if legacy_match:
        topic = legacy_match.group("topic")
        
        # Create base meta dict for GOAL_SHIFT
        meta_dict = {
            "event": "GOAL_SHIFT",
            "reason": ShiftReason.MANUAL.value,  # Default for legacy
        }
        
        # Try to map topic to goal token if provided
        if topic:
            mapped_goal = _map_topic_to_goal_token(topic.strip())
            if mapped_goal:
                meta_dict["to"] = mapped_goal.value
            else:
                # Unknown topic, store as error but keep parsing
                return meta_dict, f"UNKNOWN_LEGACY_TOPIC:{topic}"
        
        # Mark missing required fields as error
        missing_fields = []
        if "seq" not in meta_dict:
            missing_fields.append("seq")
        if "from" not in meta_dict:
            missing_fields.append("from")
        if "to" not in meta_dict:
            missing_fields.append("to")
            
        if missing_fields:
            error = f"MISSING_FIELDS:{','.join(missing_fields)}"
            return meta_dict, error
        
        return meta_dict, None
    
    return None, "NO_LEGACY_META"


def _map_topic_to_goal_token(topic: str) -> Optional[GoalToken]:
    """
    Map legacy topic strings to new GoalToken enums.
    
    Args:
        topic: Legacy topic string
        
    Returns:
        Mapped GoalToken or None if no mapping found
    """
    topic_lower = topic.lower().replace(" ", "_").replace("-", "_")
    
    # Direct mappings
    direct_mappings = {
        "transaction_dispute": GoalToken.dispute_tx,
        "transactions": GoalToken.transactions_review,
        "transaction_review": GoalToken.transactions_review,
        "account_balance": GoalToken.account_info,
        "account_info": GoalToken.account_info,
        "bill_payment": GoalToken.billpay,
        "billpay": GoalToken.billpay,
        "card_services": GoalToken.card_services,
        "cards": GoalToken.card_services,
        "transfers": GoalToken.transfers,
        "transfer": GoalToken.transfers,
        "statements": GoalToken.statements,
        "statement": GoalToken.statements,
        "auth": GoalToken.auth_access,
        "authentication": GoalToken.auth_access,
        "auth_access": GoalToken.auth_access,
        "product_info": GoalToken.product_info,
        "policy_explain_reg_e": GoalToken.policy_explain_reg_e,
    }
    
    if topic_lower in direct_mappings:
        return direct_mappings[topic_lower]
    
    # Fuzzy matching for common variations
    if "dispute" in topic_lower or "fraud" in topic_lower:
        return GoalToken.dispute_tx
    elif "transaction" in topic_lower:
        return GoalToken.transactions_review
    elif "account" in topic_lower:
        return GoalToken.account_info
    elif "card" in topic_lower:
        return GoalToken.card_services
    elif "transfer" in topic_lower:
        return GoalToken.transfers
    elif "statement" in topic_lower:
        return GoalToken.statements
    elif "bill" in topic_lower or "pay" in topic_lower:
        return GoalToken.billpay
    elif "auth" in topic_lower or "login" in topic_lower:
        return GoalToken.auth_access
    elif "product" in topic_lower:
        return GoalToken.product_info
    elif "policy" in topic_lower or "reg_e" in topic_lower:
        return GoalToken.policy_explain_reg_e
    
    return None


def is_legacy_meta_line(line: str) -> bool:
    """Quick check if a line contains a legacy meta tag."""
    return bool(LEGACY_GOAL_SHIFT_RE.match(line.strip())) 