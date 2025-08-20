"""Tools for the retail user simulator."""

from typing import Optional
from tau2.domains.retail.user_data_model import RetailUserDB
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool


class RetailUserTools(ToolKitBase):
    """Tools available to the retail user simulator."""
    
    def __init__(self, user_db: RetailUserDB):
        self.user_db = user_db
        super().__init__(user_db)
    
    @is_tool(ToolType.WRITE)
    def update_user(self, **kwargs) -> str:
        """Update user information in the database.
        
        Args:
            **kwargs: Key-value pairs to update in the user database
            
        Returns:
            Confirmation message of the update
        """
        self.user_db.update_db(**kwargs)
        updated_fields = list(kwargs.keys())
        return f"Updated user fields: {', '.join(updated_fields)}"
    
    @is_tool(ToolType.WRITE)
    def authenticate_user(self, user_id: str, name: str = None, email: str = None, zip_code: str = None) -> str:
        """Authenticate the user and store their information.
        
        Args:
            user_id: The user's unique identifier
            name: The user's full name (optional)
            email: The user's email address (optional)
            zip_code: The user's zip code (optional)
            
        Returns:
            Confirmation message of authentication
        """
        self.user_db.authenticate_user(user_id, name, email, zip_code)
        return f"User {user_id} authenticated successfully"
    
    @is_tool(ToolType.WRITE)
    def add_conversation_topic(self, topic: str) -> str:
        """Add a topic to the current conversation context.
        
        Args:
            topic: The topic being discussed (e.g., "order_cancellation", "product_return")
            
        Returns:
            Confirmation message
        """
        self.user_db.add_conversation_topic(topic)
        return f"Added topic: {topic}"
    
    @is_tool(ToolType.WRITE)
    def add_goal(self, goal: str) -> str:
        """Add a goal to the user's conversation goals.
        
        Args:
            goal: The goal the user wants to achieve (e.g., "cancel_order", "return_product")
            
        Returns:
            Confirmation message and goal shift information
        """
        initial_goal_count = len(self.user_db.current_goals)
        self.user_db.add_goal(goal)
        
        if len(self.user_db.current_goals) > initial_goal_count:
            if initial_goal_count == 0:
                return f"Set initial goal: {goal}"
            else:
                return f"Added new goal: {goal} (Goal shift #{self.user_db.goal_shifts_made})"
        return f"Goal already exists: {goal}"
    
    @is_tool(ToolType.WRITE)
    def set_current_order(self, order_id: str) -> str:
        """Set the order currently being discussed.
        
        Args:
            order_id: The order ID (e.g., "#W1234567")
            
        Returns:
            Confirmation message
        """
        self.user_db.set_current_order(order_id)
        return f"Set current order: {order_id}"
    
    @is_tool(ToolType.WRITE)
    def set_current_product(self, product_id: str) -> str:
        """Set the product currently being discussed.
        
        Args:
            product_id: The product ID
            
        Returns:
            Confirmation message
        """
        self.user_db.set_current_product(product_id)
        return f"Set current product: {product_id}"
    
    @is_tool(ToolType.READ)
    def get_user_context(self) -> dict:
        """Get the current user context and conversation state.
        
        Returns:
            Dictionary containing user context information
        """
        return {
            "authenticated": self.user_db.authenticated,
            "user_id": self.user_db.user_id,
            "current_order": self.user_db.current_order_id,
            "current_product": self.user_db.current_product_id,
            "conversation_topics": self.user_db.conversation_topics,
            "current_goals": self.user_db.current_goals,
            "goal_shifts_made": self.user_db.goal_shifts_made,
            "initial_goal": self.user_db.initial_goal
        }
    
    @is_tool(ToolType.READ)
    def get_goal_shift_status(self) -> dict:
        """Get information about goal shifts in the conversation.
        
        Returns:
            Dictionary with goal shift information
        """
        return {
            "initial_goal": self.user_db.initial_goal,
            "current_goals": self.user_db.current_goals,
            "total_goals": len(self.user_db.current_goals),
            "goal_shifts_made": self.user_db.goal_shifts_made,
            "has_shifted": self.user_db.goal_shifts_made > 0
        }
    
    @is_tool(ToolType.WRITE)
    def reset_conversation(self) -> str:
        """Reset the user database for a new conversation.
        
        Returns:
            Confirmation message
        """
        self.user_db.reset()
        return "User conversation context reset"
