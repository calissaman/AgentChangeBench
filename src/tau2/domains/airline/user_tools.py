from tau2.environment.toolkit import ToolKitBase
from tau2.environment.toolkit import is_tool, ToolType
from tau2.domains.airline.user_data_model import AirlineUserDB


class AirlineUserTools(ToolKitBase):
    """
    Tools for interacting with the airline environment from user side.
    """

    db: AirlineUserDB

    def __init__(self, db: AirlineUserDB):
        super().__init__(db)

    def update_user(self, **kwargs):
        """Update user data with the provided key-value pairs."""
        for key, value in kwargs.items():
            if hasattr(self.db, key):
                setattr(self.db, key, value)

    @is_tool(ToolType.READ)
    def get_reservation_status(self) -> str:
        """Returns current reservation status."""
        if self.db.has_existing_reservation and self.db.reservation_id:
            status = self.db.reservation_status or "Active"
            return f"Reservation {self.db.reservation_id}: {status}"
        return "No existing reservation found."

    @is_tool(ToolType.READ)
    def check_baggage_allowance(self) -> str:
        """Returns baggage allowance information."""
        if self.db.baggage_count > 0:
            return f"Current baggage: {self.db.baggage_count} checked bag(s)"
        elif self.db.needs_checked_bags:
            return "Baggage needed but not yet added"
        return "No checked bags required."

    @is_tool(ToolType.READ)
    def get_travel_status(self) -> str:
        """Returns basic travel status information."""
        info = []
        if self.db.user_id:
            info.append(f"User ID: {self.db.user_id}")
        if self.db.membership_level:
            info.append(f"Membership: {self.db.membership_level}")
        if self.db.has_upcoming_flight:
            info.append("Has upcoming flight")
        if self.db.has_family:
            info.append("Traveling with family")
        return "\n".join(info) if info else "Travel status not available."

    @is_tool(ToolType.READ)
    def check_payment_status(self) -> str:
        """Returns payment status information."""
        if self.db.has_payment_issue:
            return "Payment issue detected - assistance needed"
        return "Payment status: OK"

    @is_tool(ToolType.READ)
    def get_customer_info(self) -> str:
        """Returns basic customer information."""
        info = []
        if self.db.customer_id:
            info.append(f"Customer ID: {self.db.customer_id}")
        if self.db.phone_number:
            info.append(f"Phone: {self.db.phone_number}")
        if self.db.first_time_flyer:
            info.append("First-time flyer")
        if self.db.budget_conscious:
            info.append("Budget-conscious traveler")
        return "\n".join(info) if info else "Customer information not available."
