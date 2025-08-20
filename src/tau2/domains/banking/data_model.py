import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import Field
from tau2.domains.banking.utilts import BANKING_DB_PATH
from tau2.environment.db import DB
from tau2.utils.pydantic_utils import BaseModelNoExtra


DEFAULT_START_DATE = datetime.date(2025, 1, 1)


class Address(BaseModelNoExtra):
    street: str = Field(description="Street address")
    city: str = Field(description="City")
    state: str = Field(description="US state code, e.g., CA")
    postal_code: str = Field(description="ZIP or ZIP+4")


class Customer(BaseModelNoExtra):
    customer_id: str = Field(description="Unique customer identifier")
    full_name: str = Field(description="Full legal name")
    date_of_birth: str = Field(description="YYYY-MM-DD for identity checks")
    email: str = Field(description="Primary email")
    phone_number: str = Field(description="Primary phone")
    address: Address = Field(description="Mailing address")
    created_at: datetime.datetime = Field(
        DEFAULT_START_DATE, description="Account creation timestamp"
    )
    account_ids: List[str] = Field(default_factory=list, description="Owned accounts")
    card_ids: List[str] = Field(default_factory=list, description="Linked cards")
    statement_ids: List[str] = Field(
        default_factory=list, description="Recent statements"
    )
    payment_request_ids: List[str] = Field(
        default_factory=list, description="Bill pay requests"
    )
    dispute_ids: List[str] = Field(default_factory=list, description="Filed disputes")
    payee_ids: List[str] = Field(
        default_factory=list, description="Saved bill pay payees"
    )


class AccountType(str, Enum):
    CHECKING = "checking"
    SAVINGS = "savings"
    CREDIT_CARD = "credit_card"


class AccountStatus(str, Enum):
    ACTIVE = "Active"
    FROZEN = "Frozen"
    CLOSED = "Closed"


class Account(BaseModelNoExtra):
    account_id: str = Field(description="Unique account identifier")
    customer_id: str = Field(description="Owner customer id")
    type: AccountType = Field(description="Account category")
    masked_number: str = Field(description="Masked number, e.g., ••••6789")
    status: AccountStatus = Field(AccountStatus.ACTIVE, description="Lifecycle state")
    current_balance: float = Field(description="Ledger balance (USD)")
    available_balance: float = Field(description="Available balance (USD)")


class CardType(str, Enum):
    DEBIT = "debit"
    CREDIT = "credit"


class CardStatus(str, Enum):
    ACTIVE = "Active"
    LOCKED = "Locked"


class Card(BaseModelNoExtra):
    card_id: str = Field(description="Unique card identifier")
    account_id: str = Field(description="Linked account id")
    type: CardType = Field(description="Debit or credit")
    last4: str = Field(description="Last 4 digits")
    status: CardStatus = Field(CardStatus.ACTIVE, description="Card state")


class StatementStatus(str, Enum):
    ISSUED = "Issued"
    OVERDUE = "Overdue"
    PAID = "Paid"


class Statement(BaseModelNoExtra):
    statement_id: str = Field(description="Unique statement identifier")
    account_id: str = Field(description="Account id")
    period_start: datetime.date = Field(description="Start date YYYY-MM-DD")
    period_end: datetime.date = Field(description="End date YYYY-MM-DD")
    issue_date: datetime.date = Field(description="Issue date")
    total_due: float = Field(description="Total due for cycle (USD)")
    minimum_due: Optional[float] = Field(None, description="Minimum due if applicable")
    due_date: Optional[datetime.date] = Field(None, description="Payment due date")
    status: StatementStatus = Field(StatementStatus.ISSUED, description="State")


class TransactionType(str, Enum):
    DEPOSIT = "deposit"
    WITHDRAWAL = "withdrawal"
    TRANSFER_OUT = "transfer_out"
    BILLPAY = "billpay"
    CARD_PURCHASE = "card_purchase"
    ATM_WITHDRAWAL = "atm_withdrawal"
    ACH_DEBIT = "ach_debit"
    ACH_CREDIT = "ach_credit"


class TransactionStatus(str, Enum):
    PENDING = "Pending"
    POSTED = "Posted"
    REVERSED = "Reversed"


class Transaction(BaseModelNoExtra):
    tx_id: str = Field(description="Unique transaction id")
    account_id: str = Field(description="Related account id")
    timestamp: datetime.datetime = Field(description="Event time")
    type: TransactionType = Field(description="Transaction category")
    amount: float = Field(description="Signed amount in USD")
    merchant_or_payee: Optional[str] = Field(
        None, description="Merchant or payee label"
    )
    status: TransactionStatus = Field(
        TransactionStatus.POSTED, description="Lifecycle status"
    )
    reference: Optional[str] = Field(None, description="Optional reference")


class DeliverType(str, Enum):
    ELECTRONIC = "electronic"
    CHECK = "check"


class Payee(BaseModelNoExtra):
    payee_id: str = Field(description="Unique payee id")
    customer_id: str = Field(description="Owner customer id")
    name: str = Field(description="Biller name")
    deliver_type: DeliverType = Field(description="Electronic or paper check")
    verified: bool = Field(False, description="Whether payee is verified")


class PaymentRequestStatus(str, Enum):
    AWAITING_PAYMENT = "Awaiting Payment"
    AUTHORIZED = "Authorized"
    SETTLED = "Settled"
    CANCELED = "Canceled"
    EXPIRED = "Expired"
    FAILED = "Failed"


class PaymentRequest(BaseModelNoExtra):
    request_id: str = Field(description="Unique request id")
    origin: str = Field(description="agent|user|autopay")
    customer_id: str = Field(description="Customer id")
    from_account_id: str = Field(description="Source account id")
    to_payee_id: str = Field(description="Destination payee id")
    amount: float = Field(description="USD amount requested")
    status: PaymentRequestStatus = Field(
        PaymentRequestStatus.AWAITING_PAYMENT, description="Status"
    )
    created_at: datetime.datetime = Field(
        DEFAULT_START_DATE, description="Creation time"
    )
    expires_at: Optional[datetime.datetime] = Field(None, description="Expiry time")


class DisputeStatus(str, Enum):
    DRAFT = "Draft"
    SUBMITTED = "Submitted"
    UNDER_REVIEW = "Under Review"
    RESOLVED_CUSTOMER_FAVOR = "Resolved-Customer Favor"
    RESOLVED_MERCHANT_FAVOR = "Resolved-Merchant Favor"
    CLOSED = "Closed"


class Dispute(BaseModelNoExtra):
    dispute_id: str = Field(description="Unique dispute id")
    account_id: str = Field(description="Account id")
    tx_id: str = Field(description="Transaction being disputed")
    reason_code: str = Field(description="Reason code label")
    status: DisputeStatus = Field(DisputeStatus.DRAFT, description="Lifecycle state")
    opened_at: datetime.datetime = Field(DEFAULT_START_DATE, description="Open time")


class BankingDB(DB):
    """Database interface for the banking domain."""

    customers: List[Customer] = Field(default_factory=list, description="Customers")
    accounts: List[Account] = Field(default_factory=list, description="Accounts")
    cards: List[Card] = Field(default_factory=list, description="Cards")
    statements: List[Statement] = Field(default_factory=list, description="Statements")
    transactions: List[Transaction] = Field(
        default_factory=list, description="Transactions"
    )
    payees: List[Payee] = Field(default_factory=list, description="Bill pay payees")
    payment_requests: List[PaymentRequest] = Field(
        default_factory=list, description="Bill pay requests"
    )
    disputes: List[Dispute] = Field(default_factory=list, description="Disputes")

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "num_customers": len(self.customers),
            "num_accounts": len(self.accounts),
            "num_cards": len(self.cards),
            "num_statements": len(self.statements),
            "num_transactions": len(self.transactions),
            "num_payees": len(self.payees),
            "num_payment_requests": len(self.payment_requests),
            "num_disputes": len(self.disputes),
        }


def get_db() -> BankingDB:
    """Load the banking database from BANKING_DB_PATH."""
    return BankingDB.load(BANKING_DB_PATH)


if __name__ == "__main__":
    db = get_db()
    print(db.get_statistics())
