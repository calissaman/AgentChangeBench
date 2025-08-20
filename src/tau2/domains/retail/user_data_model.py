"""Data model for the retail user simulator."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class RetailUserDB(BaseModel):
    """Database containing retail user simulator state and information."""
    
    # User authentication state
    authenticated: bool = Field(default=False, description="Whether user is authenticated")
    user_id: Optional[str] = Field(default=None, description="Authenticated user ID")
    
    # User profile information
    name: Optional[str] = Field(default=None, description="User's full name")
    email: Optional[str] = Field(default=None, description="User's email address")
    zip_code: Optional[str] = Field(default=None, description="User's zip code")
    phone_number: Optional[str] = Field(default=None, description="User's phone number")
    
    # Order and transaction context
    current_order_id: Optional[str] = Field(default=None, description="Current order being discussed")
    recent_orders: List[str] = Field(default_factory=list, description="List of recent order IDs")
    
    # Product context
    current_product_id: Optional[str] = Field(default=None, description="Current product being discussed")
    viewed_products: List[str] = Field(default_factory=list, description="List of recently viewed product IDs")
    
    # Payment and account context
    active_payment_methods: List[str] = Field(default_factory=list, description="User's active payment method IDs")
    gift_card_balance: Optional[float] = Field(default=None, description="Current gift card balance")
    
    # Conversation context
    conversation_topics: List[str] = Field(default_factory=list, description="Topics discussed in current conversation")
    pending_actions: List[str] = Field(default_factory=list, description="Actions user wants to complete")
    
    # Goal shift tracking
    initial_goal: Optional[str] = Field(default=None, description="User's initial conversation goal")
    current_goals: List[str] = Field(default_factory=list, description="All goals in current conversation")
    goal_shifts_made: int = Field(default=0, description="Number of goal shifts made")
    
    @classmethod
    def load(cls, file_path: Optional[str] = None) -> "RetailUserDB":
        """Load retail user database from file or create default."""
        if file_path:
            try:
                import json
                with open(file_path, 'r') as f:
                    data = json.load(f)
                return cls(**data)
            except (FileNotFoundError, json.JSONDecodeError):
                pass
        
        # Return default instance
        return cls()
    
    def update_db(self, **kwargs) -> None:
        """Update the database with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def authenticate_user(self, user_id: str, name: str = None, email: str = None, zip_code: str = None) -> None:
        """Authenticate user and store their information."""
        self.authenticated = True
        self.user_id = user_id
        if name:
            self.name = name
        if email:
            self.email = email
        if zip_code:
            self.zip_code = zip_code
    
    def add_conversation_topic(self, topic: str) -> None:
        """Add a topic to the conversation history."""
        if topic not in self.conversation_topics:
            self.conversation_topics.append(topic)
    
    def add_goal(self, goal: str) -> None:
        """Add a new goal to the conversation."""
        if not self.initial_goal:
            self.initial_goal = goal
        
        if goal not in self.current_goals:
            self.current_goals.append(goal)
            if len(self.current_goals) > 1:
                self.goal_shifts_made += 1
    
    def set_current_order(self, order_id: str) -> None:
        """Set the current order being discussed."""
        self.current_order_id = order_id
        if order_id not in self.recent_orders:
            self.recent_orders.append(order_id)
            # Keep only last 5 orders
            if len(self.recent_orders) > 5:
                self.recent_orders = self.recent_orders[-5:]
    
    def set_current_product(self, product_id: str) -> None:
        """Set the current product being discussed."""
        self.current_product_id = product_id
        if product_id not in self.viewed_products:
            self.viewed_products.append(product_id)
            # Keep only last 10 products
            if len(self.viewed_products) > 10:
                self.viewed_products = self.viewed_products[-10:]
    
    def reset(self) -> None:
        """Reset the database for a new conversation."""
        self.authenticated = False
        self.user_id = None
        self.current_order_id = None
        self.current_product_id = None
        self.conversation_topics = []
        self.pending_actions = []
        self.initial_goal = None
        self.current_goals = []
        self.goal_shifts_made = 0
