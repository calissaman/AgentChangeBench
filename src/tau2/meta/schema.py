from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict


class GoalToken(str, Enum):
    """Enumeration of valid banking goal tokens."""

    # Authentication and access
    auth_access = "auth_access"
    authentication = "authentication"  # Common in tasks

    # Account services
    account_info = "account_info"

    # Card services
    card_services = "card_services"
    cards = "cards"  # Common in tasks

    # Payment services
    billpay = "billpay"
    payments = "payments"  # Common in tasks
    transfers = "transfers"

    # Transaction services
    statements = "statements"  # Common in tasks
    transactions = "transactions"  # Common in tasks
    transactions_review = "transactions_review"

    # Dispute and fraud
    dispute = "dispute"  # Common in tasks
    dispute_tx = "dispute_tx"
    fraud_response = "fraud_response"  # Common in tasks

    # Alerts and notifications
    alerts = "alerts"  # Common in tasks

    # Information and policy
    product_info = "product_info"
    policy_explain_reg_e = "policy_explain_reg_e"  # example no-tool goal


class ShiftReason(str, Enum):
    """Enumeration of valid shift reasons."""

    MANUAL = "MANUAL"
    FORCED_MAX_TURNS = "FORCED_MAX_TURNS"
    FORCED_TRANSFER_OFFER = "FORCED_TRANSFER_OFFER"
    FORCED_REPEATED_VERIFICATION = "FORCED_REPEATED_VERIFICATION"
    FORCED_AGENT_OPEN_PROMPT = "FORCED_AGENT_OPEN_PROMPT"
    FORCED_UNABLE_TO_HELP = "FORCED_UNABLE_TO_HELP"


class MetaEvent(BaseModel):
    """Structured representation of a parsed meta tag event."""

    model_config = ConfigDict(populate_by_name=True, use_enum_values=True)

    event: str = Field(..., pattern="^(GOAL_SHIFT|PARK|RESUME)$")
    seq: Optional[int] = None
    from_: Optional[GoalToken] = Field(None, alias="from")
    to: Optional[GoalToken] = None
    reason: Optional[ShiftReason] = None
    task: Optional[str] = None
    note: Optional[str] = None
    raw: Optional[str] = None  # Original meta line
    error: Optional[str] = None  # Validation error if any

    @field_validator("seq")
    @classmethod
    def seq_pos(cls, v):
        """Validate that seq is positive."""
        if v is not None and v < 1:
            raise ValueError("seq must be >= 1")
        return v

    def is_valid(self) -> bool:
        """Check if the meta event has all required fields for its type."""
        if self.event == "GOAL_SHIFT":
            return (
                self.seq is not None
                and self.from_ is not None
                and self.to is not None
                and self.reason is not None
            )
        elif self.event == "PARK":
            return self.seq is not None and self.task is not None
        elif self.event == "RESUME":
            return self.seq is not None and self.task is not None
        return False
