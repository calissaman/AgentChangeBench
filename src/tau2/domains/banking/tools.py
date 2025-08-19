import uuid
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from tau2.domains.banking.utilts import BANKING_DB_PATH
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool

from tau2.domains.banking.data_model import (
    BankingDB,
    Customer,
    Account,
    AccountStatus,
    Card,
    CardStatus,
    Statement,
    Transaction,
    TransactionType,
    TransactionStatus,
    Payee,
    DeliverType,
    PaymentRequest,
    PaymentRequestStatus,
    Dispute,
    DisputeStatus,
)


class IDGenerator:
    def __init__(self) -> None:
        self._ctr = defaultdict(int)

    def get_id(self, kind: str, prefix: Optional[str] = None) -> str:
        self._ctr[kind] += 1
        prefix = prefix or kind
        return f"{prefix}_{self._ctr[kind]}"

    @staticmethod
    def random_id(prefix: str) -> str:
        return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _now() -> datetime:
    # Use runner's clock if you later add a utils.get_now; for now naive UTC-like stamp.
    return datetime.utcnow()


class BankingTools(ToolKitBase):
    """
    Tools for the lean banking domain implementing functions described in the policy.
    Public tools are decorated with @is_tool and exposed to the model. Helpers are private.
    """

    db: BankingDB

    def __init__(self, db: BankingDB) -> None:
        super().__init__(db)
        self.idgen = IDGenerator()
        # Ephemeral stores for research instrumentation
        self._shift_events: List[Dict[str, Any]] = []
        self._parked_tasks: Dict[str, Dict[str, Any]] = {}

    def _get_customer_by_id(self, customer_id: str) -> Customer:
        for c in self.db.customers:
            if c.customer_id == customer_id:
                return c
        raise ValueError(f"Customer {customer_id} not found")

    def _get_customer_by_phone_exact(self, phone: str) -> Customer:
        for c in self.db.customers:
            if c.phone_number == phone:
                return c
        raise ValueError(f"Customer with phone {phone} not found")

    def _get_account(self, account_id: str) -> Account:
        for a in self.db.accounts:
            if a.account_id == account_id:
                return a
        raise ValueError(f"Account {account_id} not found")

    def _get_card(self, card_id: str) -> Card:
        for card in self.db.cards:
            if card.card_id == card_id:
                return card
        raise ValueError(f"Card {card_id} not found")

    def _get_statement(self, statement_id: str) -> Statement:
        for s in self.db.statements:
            if s.statement_id == statement_id:
                return s
        raise ValueError(f"Statement {statement_id} not found")

    def _get_payee(self, payee_id: str) -> Payee:
        for p in self.db.payees:
            if p.payee_id == payee_id:
                return p
        raise ValueError(f"Payee {payee_id} not found")

    def _get_request(self, request_id: str) -> PaymentRequest:
        for r in self.db.payment_requests:
            if r.request_id == request_id:
                return r
        raise ValueError(f"Payment request {request_id} not found")

    def _get_dispute(self, dispute_id: str) -> Dispute:
        for d in self.db.disputes:
            if d.dispute_id == dispute_id:
                return d
        raise ValueError(f"Dispute {dispute_id} not found")

    def _assert_customer_owns_account(self, customer_id: str, account_id: str) -> None:
        c = self._get_customer_by_id(customer_id)
        if account_id not in c.account_ids:
            raise ValueError(f"Account {account_id} is not owned by customer {customer_id}")

    def _assert_customer_owns_payee(self, customer_id: str, payee_id: str) -> None:
        c = self._get_customer_by_id(customer_id)
        if payee_id not in c.payee_ids:
            raise ValueError(f"Payee {payee_id} is not owned by customer {customer_id}")

    # ----------------------------
    # Lookup
    # ----------------------------

    @is_tool(ToolType.READ)
    def get_customer_by_id(self, customer_id: str) -> Customer:
        """
        Retrieve a customer by id.
        """
        return self._get_customer_by_id(customer_id)

    @is_tool(ToolType.READ)
    def get_customer_by_phone(self, phone_number: str) -> Customer:
        """
        Retrieve a customer by their primary phone number.
        """
        return self._get_customer_by_phone_exact(phone_number)

    @is_tool(ToolType.READ)
    def get_customer_by_name(self, full_name: str, dob: str) -> List[Customer]:
        """
        Search customers by exact full name and DOB (YYYY-MM-DD).
        """
        matches: List[Customer] = []
        for c in self.db.customers:
            if c.full_name.lower() == full_name.lower() and c.date_of_birth == dob:
                matches.append(c)
        return matches

    # ----------------------------
    # Accounts, statements, transactions
    # ----------------------------

    @is_tool(ToolType.READ)
    def get_accounts(self, customer_id: str) -> List[Account]:
        """
        List accounts owned by the customer.
        """
        c = self._get_customer_by_id(customer_id)
        return [self._get_account(aid) for aid in c.account_ids]

    @is_tool(ToolType.READ)
    def get_account(self, account_id: str) -> Account:
        """
        Get one account.
        """
        return self._get_account(account_id)

    @is_tool(ToolType.READ)
    def get_statements(self, account_id: str, limit: int = 12) -> List[Statement]:
        """
        Return recent statements for an account, newest first.
        """
        items = [s for s in self.db.statements if s.account_id == account_id]
        items.sort(key=lambda s: s.issue_date, reverse=True)
        return items[:limit]

    @is_tool(ToolType.READ)
    def get_transactions(
        self,
        account_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Transaction]:
        """
        Return recent transactions filtered by time window, newest first.
        """
        txs = [t for t in self.db.transactions if t.account_id == account_id]
        if start_time:
            txs = [t for t in txs if t.timestamp >= start_time]
        if end_time:
            txs = [t for t in txs if t.timestamp <= end_time]
        txs.sort(key=lambda t: t.timestamp, reverse=True)
        return txs[:limit]

    # ----------------------------
    # Payees & Bill Pay Requests
    # ----------------------------

    @is_tool(ToolType.WRITE)
    def add_payee(self, customer_id: str, name: str, deliver_type: str = "electronic") -> Dict[str, Any]:
        """
        Add a bill pay payee for the customer. New payees are considered verified for this simulator.
        """
        c = self._get_customer_by_id(customer_id)
        try:
            dt = DeliverType(deliver_type)
        except Exception:
            raise ValueError("deliver_type must be 'electronic' or 'check'")

        payee_id = IDGenerator.random_id("PY")
        p = Payee(
            payee_id=payee_id,
            customer_id=customer_id,
            name=name,
            deliver_type=dt,
            verified=True,
        )
        self.db.payees.append(p)
        c.payee_ids.append(payee_id)
        logger.info(f"Payee added: {payee_id} for customer {customer_id}")
        return {"payee_id": payee_id, "verified": True}

    @is_tool(ToolType.WRITE)
    def create_payment_request(
        self,
        customer_id: str,
        from_account_id: str,
        to_payee_id: str,
        amount: float,
        expires_at: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Create a bill pay payment request and set status to AWAITING_PAYMENT.
        Checks:
          - Customer owns the from_account and payee.
          - No other request in AWAITING_PAYMENT for this customer (policy parity with telecom).
          - Account is Active.
        """
        self._assert_customer_owns_account(customer_id, from_account_id)
        self._assert_customer_owns_payee(customer_id, to_payee_id)
        acct = self._get_account(from_account_id)
        if acct.status != AccountStatus.ACTIVE:
            raise ValueError("Source account must be Active")
        if amount <= 0:
            raise ValueError("Amount must be positive")

        # Enforce single awaiting-request per customer
        for r in self.db.payment_requests:
            if r.customer_id == customer_id and r.status == PaymentRequestStatus.AWAITING_PAYMENT:
                raise ValueError("Another payment request is already Awaiting Payment for this customer")

        rid = IDGenerator.random_id("PR")
        pr = PaymentRequest(
            request_id=rid,
            origin="agent",
            customer_id=customer_id,
            from_account_id=from_account_id,
            to_payee_id=to_payee_id,
            amount=amount,
            status=PaymentRequestStatus.AWAITING_PAYMENT,
            created_at=_now(),
            expires_at=expires_at,
        )
        self.db.payment_requests.append(pr)
        cust = self._get_customer_by_id(customer_id)
        cust.payment_request_ids.append(rid)
        logger.info(f"Payment request created {rid} amount ${amount:.2f}")
        return {"request_id": rid, "status": pr.status}

    @is_tool(ToolType.READ)
    def check_payment_request(self, request_id: str) -> PaymentRequest:
        """
        Return the current state of a payment request.
        """
        return self._get_request(request_id)

    @is_tool(ToolType.WRITE)
    def authorize_payment_request(self, request_id: str) -> Dict[str, Any]:
        """
        Mark a request as AUTHORIZED (e.g., user approved in-channel).
        """
        pr = self._get_request(request_id)
        if pr.status != PaymentRequestStatus.AWAITING_PAYMENT:
            raise ValueError(f"Request {request_id} is not Awaiting Payment")
        pr.status = PaymentRequestStatus.AUTHORIZED
        logger.info(f"Payment request {request_id} authorized")
        return {"request_id": request_id, "status": pr.status}

    @is_tool(ToolType.WRITE)
    def make_payment(self, request_id: str) -> Dict[str, Any]:
        """
        Settle an AUTHORIZED payment request:
          - Debit the source account available and current balances
          - Append a BILLPAY transaction
          - Set request status to SETTLED (or FAILED on insufficient funds)
        """
        pr = self._get_request(request_id)
        if pr.status != PaymentRequestStatus.AUTHORIZED:
            raise ValueError("Request must be AUTHORIZED before payment")

        acct = self._get_account(pr.from_account_id)

        if acct.available_balance < pr.amount:
            pr.status = PaymentRequestStatus.FAILED
            logger.warning(f"Payment {request_id} failed: insufficient funds")
            return {"request_id": request_id, "status": pr.status, "reason": "insufficient_funds"}

        # Debit account
        acct.available_balance -= pr.amount
        acct.current_balance -= pr.amount

        # Record transaction
        tx_id = IDGenerator.random_id("TX")
        tx = Transaction(
            tx_id=tx_id,
            account_id=acct.account_id,
            timestamp=_now(),
            type=TransactionType.BILLPAY,
            amount=-abs(pr.amount),
            merchant_or_payee=self._get_payee(pr.to_payee_id).name,
            status=TransactionStatus.POSTED,
            reference=request_id,
        )
        self.db.transactions.append(tx)

        # Close request
        pr.status = PaymentRequestStatus.SETTLED
        logger.info(f"Payment {request_id} settled, tx {tx_id}")
        return {"request_id": request_id, "status": pr.status, "tx_id": tx_id}

    @is_tool(ToolType.WRITE)
    def cancel_payment_request(self, request_id: str) -> Dict[str, Any]:
        """
        Cancel a payment request if not settled.
        """
        pr = self._get_request(request_id)
        if pr.status in (PaymentRequestStatus.SETTLED, PaymentRequestStatus.FAILED, PaymentRequestStatus.CANCELED):
            # Idempotent cancel; do nothing if already terminal (except Settled)
            if pr.status == PaymentRequestStatus.SETTLED:
                raise ValueError("Cannot cancel a settled payment")
            return {"request_id": request_id, "status": pr.status}
        pr.status = PaymentRequestStatus.CANCELED
        logger.info(f"Payment request {request_id} canceled")
        return {"request_id": request_id, "status": pr.status}


    @is_tool(ToolType.WRITE)
    def lock_card(self, card_id: str, reason: Optional[str] = None) -> Dict[str, Any]:
        """
        Lock a debit or credit card. Idempotent if already locked.
        """
        card = self._get_card(card_id)
        if card.status == CardStatus.LOCKED:
            return {"card_id": card_id, "status": card.status}
        card.status = CardStatus.LOCKED
        logger.info(f"Card {card_id} locked. Reason: {reason or 'unspecified'}")
        return {"card_id": card_id, "status": card.status}

    @is_tool(ToolType.WRITE)
    def unlock_card(self, card_id: str) -> Dict[str, Any]:
        """
        Unlock a debit or credit card. Requires policy confirmation outside this tool.
        """
        card = self._get_card(card_id)
        if card.status != CardStatus.LOCKED:
            return {"card_id": card_id, "status": card.status}
        card.status = CardStatus.ACTIVE
        logger.info(f"Card {card_id} unlocked")
        return {"card_id": card_id, "status": card.status}


    @is_tool(ToolType.WRITE)
    def file_dispute(self, account_id: str, tx_id: str, reason_code: str) -> Dict[str, Any]:
        """
        File a transaction dispute. Transaction is marked as Disputed.
        """
        # Validate account and tx
        _ = self._get_account(account_id)
        tx: Optional[Transaction] = None
        for t in self.db.transactions:
            if t.tx_id == tx_id and t.account_id == account_id:
                tx = t
                break
        if tx is None:
            raise ValueError(f"Transaction {tx_id} not found for account {account_id}")

        tx.status = TransactionStatus.PENDING if tx.status == TransactionStatus.PENDING else TransactionStatus.POSTED
        tx.status = TransactionStatus.DISPUTED

        dispute_id = IDGenerator.random_id("DP")
        d = Dispute(
            dispute_id=dispute_id,
            account_id=account_id,
            tx_id=tx_id,
            reason_code=reason_code,
            status=DisputeStatus.SUBMITTED,
            opened_at=_now(),
        )
        self.db.disputes.append(d)
        logger.info(f"Dispute filed {dispute_id} for tx {tx_id}")
        return {"dispute_id": dispute_id, "status": d.status}

    @is_tool(ToolType.READ)
    def get_dispute(self, dispute_id: str) -> Dispute:
        """
        Get dispute details.
        """
        return self._get_dispute(dispute_id)


    @is_tool(ToolType.GENERIC)
    def log_shift_event(
        self,
        turn_no: int,
        from_class: str,
        to_class: str,
        trigger_terms: List[str],
        requires_reauth: bool,
    ) -> Dict[str, Any]:
        """
        Record a goal-shift detection event for evaluation.
        """
        evt = {
            "ts": _now().isoformat(),
            "turn_no": turn_no,
            "from_class": from_class,
            "to_class": to_class,
            "trigger_terms": trigger_terms,
            "requires_reauth": requires_reauth,
        }
        self._shift_events.append(evt)
        logger.info(f"shift_event: {evt}")
        return {"logged": True, "count": len(self._shift_events)}

    @is_tool(ToolType.GENERIC)
    def park_task(self, current_task_id: str, resume_hint: Optional[str] = None) -> Dict[str, Any]:
        """
        Park the current task and return a parked_task_id that can be resumed later.
        """
        pid = IDGenerator.random_id("PT")
        self._parked_tasks[pid] = {
            "task_id": current_task_id,
            "resume_hint": resume_hint,
            "parked_at": _now().isoformat(),
        }
        logger.info(f"Task parked {pid}: {self._parked_tasks[pid]}")
        return {"parked_task_id": pid}

    @is_tool(ToolType.GENERIC)
    def resume_task(self, parked_task_id: str) -> Dict[str, Any]:
        """
        Resume a previously parked task. Returns its stored metadata.
        """
        meta = self._parked_tasks.get(parked_task_id)
        if not meta:
            raise ValueError(f"Parked task {parked_task_id} not found")
        logger.info(f"Task resumed {parked_task_id}")
        return {"status": "Resumed", "metadata": meta}


    @is_tool(ToolType.GENERIC)
    def transfer_to_human_agents(self, summary: str) -> str:
        """
        Transfer the user to a human agent with a summary. Policy decides when allowed.
        """
        logger.warning(f"Transfer to human requested: {summary}")
        return "Transfer successful"

    # Assertion methods for task evaluation
    def assert_any_payment_request_with_status(
        self, 
        customer_id: str, 
        expected_status: str, 
        min_amount: Optional[float] = None,
        from_account_id: Optional[str] = None
    ) -> bool:
        """Assert that a payment request exists with the given status and criteria."""
        for pr in self.db.payment_requests:
            if pr.customer_id == customer_id and pr.status == expected_status:
                if min_amount is not None and pr.amount < min_amount:
                    continue
                if from_account_id is not None and pr.from_account_id != from_account_id:
                    continue
                return True
        return False

    def assert_card_status(self, card_id: str, expected: str) -> bool:
        """Assert that a card has the expected status."""
        try:
            card = self._get_card(card_id)
            return card.status == expected
        except ValueError:
            return False

    def assert_shift_event_count_at_least(self, n: int) -> bool:
        """Assert that at least n shift events have been logged."""
        return len(self._shift_events) >= n

    def assert_parked_task_exists(self, parked_task_id: str) -> bool:
        """Assert that a parked task exists (use '*' to check if any exists)."""
        if parked_task_id == "*":
            return len(self._parked_tasks) > 0
        return parked_task_id in self._parked_tasks


if __name__ == "__main__":
    banking = BankingTools(BankingDB.load(BANKING_DB_PATH))
    print(banking.get_statistics())
