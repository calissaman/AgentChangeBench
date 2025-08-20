"""
Goal to tools mapping for alignment scoring.

This mapping defines which tools are allowed/required for each goal
and other metadata needed for alignment scoring.
"""

from tau2.meta.schema import GoalToken

# Central mapping of goals to their allowed tools and requirements
GOAL_TOOL_MAP = {
    GoalToken.auth_access: {
        "allowed_tools": ["verify_identity", "login", "authenticate_user"],
        "requires_tool": True,
        "required_slots": ["username", "password"],
        "policy_hooks": ["verification", "security"],
        "description": "User authentication and access verification",
    },
    GoalToken.authentication: {
        "allowed_tools": [
            "get_customer_by_phone",
            "get_customer_by_name",
            "get_customer_by_id",
        ],
        "requires_tool": True,
        "required_slots": ["phone_number", "name", "date_of_birth"],
        "policy_hooks": ["verification", "identity"],
        "description": "Customer identity verification and authentication",
    },
    GoalToken.account_info: {
        "allowed_tools": ["get_accounts", "get_account"],
        "requires_tool": True,
        "required_slots": ["account_id", "customer_id"],
        "policy_hooks": ["account_access"],
        "description": "Account information and balance inquiries",
    },
    GoalToken.card_services: {
        "allowed_tools": ["lock_card", "unlock_card"],
        "requires_tool": True,
        "required_slots": ["card_id"],
        "policy_hooks": ["card_management", "security"],
        "description": "Card management services",
    },
    GoalToken.cards: {
        "allowed_tools": ["lock_card", "unlock_card"],
        "requires_tool": True,
        "required_slots": ["card_id"],
        "policy_hooks": ["card_management", "security"],
        "description": "Card services and management",
    },
    GoalToken.billpay: {
        "allowed_tools": ["add_payee", "create_payment_request", "make_payment"],
        "requires_tool": True,
        "required_slots": ["customer_id", "amount"],
        "policy_hooks": ["payment_authorization"],
        "description": "Bill payment services",
    },
    GoalToken.payments: {
        "allowed_tools": [
            "add_payee",
            "create_payment_request",
            "make_payment",
            "authorize_payment_request",
        ],
        "requires_tool": True,
        "required_slots": ["customer_id", "amount", "from_account_id"],
        "policy_hooks": ["payment_authorization", "confirmation"],
        "description": "Payment and transfer services",
    },
    GoalToken.transfers: {
        "allowed_tools": ["create_payment_request", "make_payment"],
        "requires_tool": True,
        "required_slots": ["from_account_id", "amount"],
        "policy_hooks": ["transfer_limits"],
        "description": "Money transfers between accounts",
    },
    GoalToken.statements: {
        "allowed_tools": ["get_statements"],
        "requires_tool": True,
        "required_slots": ["account_id"],
        "policy_hooks": ["statement_access"],
        "description": "Account statements and monthly reports",
    },
    GoalToken.transactions: {
        "allowed_tools": ["get_transactions"],
        "requires_tool": True,
        "required_slots": ["account_id"],
        "policy_hooks": ["transaction_access"],
        "description": "Transaction history and recent activity",
    },
    GoalToken.transactions_review: {
        "allowed_tools": ["get_transactions"],
        "requires_tool": True,
        "required_slots": ["account_id"],
        "policy_hooks": ["transaction_review"],
        "description": "Transaction review and analysis",
    },
    GoalToken.dispute: {
        "allowed_tools": ["file_dispute", "get_dispute"],
        "requires_tool": True,
        "required_slots": ["account_id", "tx_id"],
        "policy_hooks": ["dispute_filing"],
        "description": "Transaction disputes and fraud reporting",
    },
    GoalToken.dispute_tx: {
        "allowed_tools": ["file_dispute", "get_dispute"],
        "requires_tool": True,
        "required_slots": ["account_id", "tx_id"],
        "policy_hooks": ["dispute_resolution"],
        "description": "Transaction dispute resolution",
    },
    GoalToken.fraud_response: {
        "allowed_tools": ["lock_card", "file_dispute"],
        "requires_tool": True,
        "required_slots": ["card_id", "account_id"],
        "policy_hooks": ["fraud_protection", "security"],
        "description": "Fraud response and security measures",
    },
    GoalToken.alerts: {
        "allowed_tools": [],
        "requires_tool": False,
        "required_slots": [],
        "policy_hooks": ["alert_setup"],
        "rubric": {
            "explanation": "Clear explanation of alert types and setup process",
            "options": "Discussion of available alert options",
            "confirmation": "Confirmation of alert preferences",
        },
        "description": "Account alerts and notification setup",
    },
    GoalToken.product_info: {
        "allowed_tools": [],
        "requires_tool": False,
        "required_slots": [],
        "policy_hooks": ["product_explanation"],
        "rubric": {
            "features": "Clear explanation of product features",
            "benefits": "Discussion of benefits and advantages",
            "eligibility": "Explanation of eligibility requirements",
        },
        "description": "Product information and features",
    },
    GoalToken.policy_explain_reg_e: {
        "allowed_tools": [],
        "requires_tool": False,
        "required_slots": [],
        "policy_hooks": ["policy_explanation"],
        "rubric": {
            "regulation": "Clear explanation of Regulation E",
            "rights": "Discussion of consumer rights",
            "procedures": "Explanation of dispute procedures",
        },
        "description": "Regulation E policy explanation",
    },
}


def get_goal_config(goal: GoalToken) -> dict:
    """Get configuration for a specific goal."""
    return GOAL_TOOL_MAP.get(goal, {})


def is_tool_required_goal(goal: GoalToken) -> bool:
    """Check if a goal requires tool usage."""
    return GOAL_TOOL_MAP.get(goal, {}).get("requires_tool", False)


def get_allowed_tools(goal: GoalToken) -> list:
    """Get allowed tools for a goal."""
    return GOAL_TOOL_MAP.get(goal, {}).get("allowed_tools", [])
