from dataclasses import dataclass
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


@dataclass
class GoalShift:
    """Represents a detected goal shift in a conversation."""
    turn_index: int
    shift_type: str = "GOAL_SHIFT"
    original_goal: Optional[str] = None
    new_goal: Optional[str] = None
    metadata: Optional[Dict] = None


class GSRTResult(BaseModel):
    """Results of GSRT calculation for a conversation."""
    goal_shifts: List[GoalShift] = Field(default_factory=list)
    recovery_times: List[int] = Field(default_factory=list)  # Recovery time for each shift
    median_gsrt: Optional[float] = None
    worst_case_gsrt: Optional[int] = None
    wall_clock_gsrt: Optional[float] = None  # If timestamps available
    alignment_threshold: float = 0.7
    num_shifts: int = 0
    
    class Config:
        arbitrary_types_allowed = True


@dataclass 
class GSRTConfig:
    """Configuration for GSRT calculation."""
    alignment_threshold: float = 0.7
    alignment_method: str = "keywords"  # "keywords", "semantic", "llm"
    max_recovery_window: int = 10  # Max turns to look for recovery
    require_explicit_acknowledgment: bool = False
    
    # Keywords for different banking goals
    goal_keywords: Dict[str, List[str]] = None
    
    def __post_init__(self):
        if self.goal_keywords is None:
            self.goal_keywords = {
                "authentication": [
                    "login", "password", "username", "verify", "identity", "authenticate",
                    "account access", "sign in", "credentials", "verification", "2fa", "mfa"
                ],
                "transactions": [
                    "transaction", "payment", "transfer", "deposit", "withdrawal", "balance",
                    "recent activity", "statement", "history", "charges", "purchases"
                ],
                "dispute": [
                    "dispute", "fraud", "unauthorized", "suspicious", "fraudulent", "charge back",
                    "report", "claim", "investigation", "refund", "error", "incorrect"
                ],
                "account_info": [
                    "account", "information", "details", "profile", "contact", "address",
                    "phone", "email", "settings", "preferences"
                ],
                "cards": [
                    "card", "debit", "credit", "block", "freeze", "activate", "replace",
                    "lost", "stolen", "pin", "limit"
                ]
            } 