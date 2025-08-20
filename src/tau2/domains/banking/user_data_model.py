from typing import Any, Dict, Optional, Union

from pydantic import Field
from tau2.environment.db import DB
from tau2.utils.pydantic_utils import BaseModelNoExtra
from tau2.domains.banking.utilts import BANKING_DATA_DIR


# Simple flat user database
class BankingUserDB(DB):
    # User identification
    phone_number: Optional[str] = None
    customer_id: Optional[str] = None

    # Account info
    primary_account_id: Optional[str] = None
    secondary_account_id: Optional[str] = None  # Added for business accounts
    primary_account_active: Optional[bool] = None
    account_balance: Optional[float] = None

    # Card info
    primary_card_id: Optional[str] = None
    primary_card_active: Optional[bool] = None

    # Payment info
    payment_request_id: Optional[str] = None
    latest_dispute_id: Optional[str] = None

    # Device/environment info
    signal_strength: str = "strong"
    is_abroad: bool = False
    network_accessible: bool = True
    authenticated: bool = True
    has_2fa_enabled: bool = True
    recognized_device: Optional[bool] = None  # Added for device recognition

    @classmethod
    def load(cls, path: Optional[str] = None) -> "BankingUserDB":
        if path is None:
            path = BANKING_DATA_DIR / "user_db.toml"
        return super().load(path)
