from typing import Optional
from pydantic import BaseModel, Field


class AirlineUserDB(BaseModel):
    """Simple flat database for user-side airline information."""
    
    # User identification
    authenticated: bool = Field(default=False, description="Whether user is authenticated")
    user_id: Optional[str] = Field(default=None, description="User ID")
    phone_number: Optional[str] = Field(default=None, description="User phone number")
    
    # Account information  
    customer_id: Optional[str] = Field(default=None, description="Customer ID")
    membership_level: Optional[str] = Field(default="regular", description="Membership level")
    
    # Reservation information
    has_existing_reservation: bool = Field(default=False, description="Whether user has existing reservation")
    reservation_id: Optional[str] = Field(default=None, description="Reservation ID")
    reservation_status: Optional[str] = Field(default=None, description="Reservation status")
    
    # Travel preferences
    needs_checked_bags: bool = Field(default=False, description="Whether user needs checked bags")
    baggage_count: int = Field(default=0, description="Number of checked bags")
    
    # Issues and status
    has_payment_issue: bool = Field(default=False, description="Whether user has payment issues")
    complex_situation: bool = Field(default=False, description="Whether user is in complex situation")
    
    # Flight information
    has_upcoming_flight: bool = Field(default=False, description="Whether user has upcoming flight")
    first_time_flyer: bool = Field(default=False, description="Whether user is first-time flyer")
    family_emergency: bool = Field(default=False, description="Whether user has family emergency")
    
    # Budget and cost information
    budget_conscious: bool = Field(default=False, description="Whether user is budget-conscious")
    traveling_with_children: bool = Field(default=False, description="Whether traveling with children")
    has_family: bool = Field(default=False, description="Whether user has family")
    
    @classmethod
    def load(cls) -> "AirlineUserDB":
        """Load default user database."""
        return cls()
