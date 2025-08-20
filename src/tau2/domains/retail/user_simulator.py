"""
Retail-specific user simulator that uses domain policy as system prompt.
"""
from typing import Optional, Tuple

from loguru import logger

from tau2.data_model.message import (
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolCall,
    UserMessage,
)
from tau2.data_model.tasks import UserInstructions
from tau2.environment.tool import Tool
from tau2.user.base import (
    OUT_OF_SCOPE,
    STOP,
    TRANSFER,
    BaseUser,
    UserState,
    ValidUserInputMessage,
    is_valid_user_history_message,
)
from tau2.domains.retail.utils import RETAIL_USER_POLICY_PATH
from tau2.utils.llm_utils import generate
from tau2.utils import load_file


RETAIL_USER_SYSTEM_PROMPT = """
{retail_user_policy}

<scenario>
{instructions}
</scenario>
""".strip()


class RetailUserSimulator(BaseUser):
    """Retail-specific user simulator that uses retail user policy as system prompt."""

    def __init__(
        self,
        tools: Optional[list[Tool]] = None,
        instructions: Optional[UserInstructions] = None,
        llm: Optional[str] = None,
        llm_args: Optional[dict] = None,
    ):
        super().__init__(instructions=instructions, llm=llm, llm_args=llm_args)
        self.tools = tools

    @property
    def retail_user_policy(self) -> str:
        """
        Load the retail user policy from file.
        The policy includes tool handling and goal shift capabilities.
        """
        return load_file(RETAIL_USER_POLICY_PATH)

    @property
    def system_prompt(self) -> str:
        """
        The system prompt for the retail user simulator.
        Uses retail user policy which includes tool handling and goal shift capabilities.
        """
        if self.instructions is None:
            logger.warning("No instructions provided for retail user simulator")

        system_prompt = RETAIL_USER_SYSTEM_PROMPT.format(
            retail_user_policy=self.retail_user_policy,
            instructions=self.instructions,
        )
        return system_prompt

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> UserState:
        """
        Get the initial state of the retail user simulator.
        
        Args:
            message_history: Optional message history to initialize with
            
        Returns:
            UserState with system prompt and message history
        """
        # Initialize message history with system prompt
        if message_history is None:
            message_history = []

        # Add system message if not already present
        if not message_history or not isinstance(message_history[0], SystemMessage):
            system_message = SystemMessage(content=self.system_prompt)
            message_history = [system_message] + message_history

        return UserState(
            message_history=message_history,
            active=True,
            should_stop=False,
            tools=self.tools or [],
        )

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> UserState:
        """
        Get the initial state of the retail user simulator.
        """
        if message_history is None:
            message_history = []
        assert all(is_valid_user_history_message(m) for m in message_history), (
            "Invalid user message history. User messages must be of type UserMessage, AssistantMessage, or ToolMessage to User."
        )

        user_state = UserState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=message_history,
        )
        return user_state

    @classmethod
    def is_stop(cls, message: UserMessage) -> bool:
        """
        Check if the message is a stop message.
        """
        if message.is_tool_call():
            return False
        assert message.content is not None
        return (
            STOP in message.content
            or TRANSFER in message.content
            or OUT_OF_SCOPE in message.content
        )

    def generate_next_message(
        self, message: ValidUserInputMessage, state: UserState
    ) -> Tuple[UserMessage, UserState]:
        return self._generate_next_message(message, state)

    def _generate_next_message(
        self, message: ValidUserInputMessage, state: UserState
    ) -> Tuple[UserMessage, UserState]:
        """Get the response from the retail user simulator.

        Args:
            message: The assistant or tool message.
            state: The current user state.

        Returns:
            The user response and updated state.
        """
        # Generate response using LLM
        try:
            response = generate(
                model=self.llm,
                messages=state.system_messages + state.messages + [message],
                tools=self.tools,
                **(self.llm_args or {})
            )
            
            # Handle different response types
            if isinstance(response, str):
                # Simple text response
                user_message = UserMessage(content=response)
                
            elif hasattr(response, 'tool_calls') and response.tool_calls:
                # Response with tool calls
                tool_calls = [
                    ToolCall(
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                        id=tc.id
                    ) for tc in response.tool_calls
                ]
                user_message = MultiToolMessage(
                    content=response.content or "",
                    tool_calls=tool_calls
                )
                
            else:
                # Standard message response
                user_message = UserMessage(content=response.content or "")

            # Create updated state
            updated_state = UserState(
                system_messages=state.system_messages,
                messages=state.messages + [message, user_message],
            )

            return user_message, updated_state

        except Exception as e:
            logger.error(f"Error generating user response: {e}")
            # Fallback response
            user_message = UserMessage(content="I'm having trouble responding. Could you please repeat that?")
            updated_state = UserState(
                system_messages=state.system_messages,
                messages=state.messages + [message, user_message],
            )
            return user_message, updated_state
