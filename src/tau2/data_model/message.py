import json
import re
from typing import Literal, Optional

from pydantic import BaseModel, Field

from tau2.utils.utils import get_now

SystemRole = Literal["system"]
UserRole = Literal["user"]
AssistantRole = Literal["assistant"]
ToolRole = Literal["tool"]
ToolRequestor = Literal["user", "assistant"]


class SystemMessage(BaseModel):
    """
    A system message.
    """

    role: SystemRole = Field(description="The role of the message sender.")
    content: Optional[str] = Field(
        description="The content of the message.", default=None
    )
    turn_idx: Optional[int] = Field(
        description="The index of the turn in the conversation.", default=None
    )
    timestamp: Optional[str] = Field(
        description="The timestamp of the message.", default_factory=get_now
    )

    def __str__(self) -> str:
        lines = [
            "SystemMessage",
        ]
        if self.turn_idx is not None:
            lines.append(f"turn_idx: {self.turn_idx}")
        if self.timestamp is not None:
            lines.append(f"timestamp: {self.timestamp}")
        if self.content is not None:
            lines.append(f"content: {self.content}")
        return "\n".join(lines)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SystemMessage):
            return False
        return self.role == other.role and self.content == other.content


class ToolCall(BaseModel):
    """
    A tool call.
    """

    id: str = Field(default="", description="The unique identifier for the tool call.")
    name: str = Field(description="The name of the tool.")
    arguments: dict = Field(description="The arguments of the tool.")
    requestor: ToolRequestor = Field(
        "assistant",
        description="The requestor of the tool call.",
    )

    def __str__(self) -> str:
        lines = [f"ToolCall (from {self.requestor})"]
        if self.id:
            lines.append(f"id: {self.id}")
        lines.append(f"name: {self.name}")
        lines.append(f"arguments:\n{json.dumps(self.arguments, indent=2)}")
        return "\n".join(lines)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ToolCall):
            return False
        return (
            self.id == other.id
            and self.name == other.name
            and self.arguments == other.arguments
            and self.requestor == other.requestor
        )


class ParticipantMessageBase(BaseModel):
    """
    A message from a participant in the conversation.
    if content is None, then tool_calls must be provided
    if tool_calls is None, then content must be provided
    """

    role: str = Field(description="The role of the message sender.")

    content: Optional[str] = Field(
        description="The content of the message.", default=None
    )
    tool_calls: Optional[list[ToolCall]] = Field(
        description="The tool calls made in the message.", default=None
    )
    meta: Optional[dict] = Field(
        description="Legacy metadata for the message (deprecated - use meta_event).", default=None
    )
    meta_event: Optional["MetaEvent"] = Field(  # Forward reference
        description="Structured meta event from meta-tags v2 system.", default=None
    )
    meta_error: Optional[str] = Field(
        description="Error message if meta tag parsing failed.", default=None
    )
    display_content: Optional[str] = Field(
        description="Filtered content for display purposes (without meta tags).", default=None
    )
    original_content: Optional[str] = Field(
        description="Original content with meta tags preserved for simulation analysis.", default=None
    )
    turn_idx: Optional[int] = Field(
        description="The index of the turn in the conversation.", default=None
    )
    timestamp: Optional[str] = Field(
        description="The timestamp of the message.", default_factory=get_now
    )
    cost: Optional[float] = Field(description="The cost of the message.", default=None)

    usage: Optional[dict] = Field(
        description="The token usage of the message.", default=None
    )
    raw_data: Optional[dict] = Field(
        description="The raw data of the message.", default=None
    )

    def validate(self):  # NOTE: It would be better to do this in the Pydantic model
        """
        Validate the message.
        """
        if not (self.has_text_content() or self.is_tool_call()):
            raise ValueError(
                f"AssistantMessage must have either content or tool calls. Got {self}"
            )

    def has_text_content(self) -> bool:
        """
        Check if the message has text content.
        """
        if self.content is None:
            return False
        if isinstance(self.content, str) and self.content.strip() == "":
            return False
        return True

    def is_tool_call(self) -> bool:
        """
        Check if the message is a tool call.
        """
        return self.tool_calls is not None

    def get_filtered_content_for_other_participant(self) -> Optional[str]:
        """
        Get the content with meta tags removed for sending to other participants.
        This removes <meta></meta> tags from the content.
        """
        if self.content is None:
            return None
        
        # Remove meta tags and their content
        filtered_content = re.sub(r'<meta>.*?</meta>', '', self.content, flags=re.DOTALL)
        # Clean up any extra whitespace left behind
        filtered_content = re.sub(r'\n\s*\n', '\n', filtered_content).strip()
        
        return filtered_content if filtered_content else None

    def extract_meta_from_content(self) -> None:
        """
        Extract meta information using the new meta-tags v2 system.
        
        This method:
        1. Parses the first line for meta tags using the new grammar
        2. Falls back to legacy migration if needed
        3. Sets meta_event, meta_error, and filters content appropriately
        4. Preserves original_content for analysis
        """
        if self.content is None:
            return
            
        # Import here to avoid circular imports
        from tau2.meta.grammar import parse_meta_line
        from tau2.meta.migrations import migrate_legacy_meta, is_legacy_meta_line
        from tau2.meta.schema import MetaEvent
        
        # Store original content
        if self.original_content is None:
            self.original_content = self.content
            
        lines = self.content.splitlines()
        if not lines:
            return
            
        first_line = lines[0].strip()
        
        # Try parsing with new grammar first
        meta_dict, error = parse_meta_line(first_line)
        
        # If new parsing failed but this looks like a legacy meta tag, try migration
        if meta_dict is None and is_legacy_meta_line(first_line):
            meta_dict, error = migrate_legacy_meta(first_line)
            
        if meta_dict is not None:
            # Create MetaEvent from parsed data
            try:
                meta_event = MetaEvent(raw=first_line, **meta_dict)
                self.meta_event = meta_event
                
                # Check if valid and set error if not
                if not meta_event.is_valid():
                    self.meta_error = error or "INVALID_META_FIELDS"
                    
            except Exception as e:
                self.meta_error = f"META_VALIDATION_ERROR:{str(e)}"
                # Create a minimal MetaEvent for analysis with safe values
                try:
                    safe_dict = {"event": meta_dict.get("event", "UNKNOWN")}
                    self.meta_event = MetaEvent(raw=first_line, error=str(e), **safe_dict)
                except Exception:
                    # If even that fails, just store the error
                    self.meta_event = None
            
            # Remove the first line (meta tag) from content
            body = "\n".join(lines[1:]) if len(lines) > 1 else ""
        else:
            # No meta tag on first line
            body = self.content
            if error and error != "NO_META_OR_BAD_TAG":
                self.meta_error = error
        
        # Remove any stray meta tags elsewhere in the content (legacy behavior)
        # Use a more precise regex that preserves spacing
        body = re.sub(r'<meta>.*?</meta>', '', body, flags=re.DOTALL)
        
        # Clean up excessive whitespace but preserve single line breaks
        body = re.sub(r'\n\s*\n\s*\n', '\n\n', body)  # Multiple empty lines -> double
        body = body.strip()
        
        # Set filtered content
        self.display_content = body
        self.content = body

    # Keep legacy method for backward compatibility
    def extract_meta_from_content_legacy(self) -> Optional[dict]:
        """
        Legacy meta extraction method for backward compatibility.
        """
        if self.content is None:
            return None
            
        meta_match = re.search(r'<meta>(.*?)</meta>', self.content, flags=re.DOTALL)
        if meta_match:
            meta_content = meta_match.group(1).strip()
            # Try to parse as structured data, otherwise just store as text
            try:
                # If it looks like a simple key:value or just value, parse accordingly
                if ':' in meta_content:
                    parts = meta_content.split(':', 1)
                    return {parts[0].strip(): parts[1].strip()}
                else:
                    return {"type": meta_content}
            except:
                return {"content": meta_content}
        return None

    def __str__(self) -> str:
        lines = [f"{self.role.capitalize()}Message"]
        if self.turn_idx is not None:
            lines.append(f"turn_idx: {self.turn_idx}")
        if self.timestamp is not None:
            lines.append(f"timestamp: {self.timestamp}")
        if self.content is not None:
            lines.append(f"content: {self.content}")
        if self.tool_calls is not None:
            lines.append("ToolCalls")
            lines.extend([str(tool_call) for tool_call in self.tool_calls])
        if self.meta is not None:
            lines.append(f"meta: {self.meta}")
        if self.meta_event is not None:
            lines.append(f"meta_event: {self.meta_event}")
        if self.meta_error is not None:
            lines.append(f"meta_error: {self.meta_error}")
        if self.cost is not None:
            lines.append(f"cost: {self.cost}")
        return "\n".join(lines)

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return False
        return (
            self.role == other.role
            and self.content == other.content
            and self.tool_calls == other.tool_calls
            and self.meta == other.meta
            and self.meta_event == other.meta_event
        )


class AssistantMessage(ParticipantMessageBase):
    """
    A message from the assistant
    """

    role: AssistantRole = Field(description="The role of the message sender.")


class UserMessage(ParticipantMessageBase):
    """
    A message from the user.
    """

    role: UserRole = Field(description="The role of the message sender.")


class ToolMessage(BaseModel):
    """
    A message from the tool.
    """

    id: str = Field(description="The unique identifier for the tool call.")
    role: ToolRole = Field(description="The role of the message sender.")
    content: Optional[str] = Field(description="The output of the tool.", default=None)
    requestor: Literal["user", "assistant"] = Field(
        "assistant",
        description="The requestor of the tool call.",
    )
    error: bool = Field(description="Whether the tool call failed.", default=False)
    turn_idx: Optional[int] = Field(
        description="The index of the turn in the conversation.", default=None
    )
    timestamp: Optional[str] = Field(
        description="The timestamp of the message.", default_factory=get_now
    )

    def __str__(self) -> str:
        lines = [f"ToolMessage (responding to {self.requestor})"]
        if self.turn_idx is not None:
            lines.append(f"turn_idx: {self.turn_idx}")
        if self.timestamp is not None:
            lines.append(f"timestamp: {self.timestamp}")
        if self.content is not None:
            lines.append(f"content: {self.content}")
        if self.error:
            lines.append("Error")
        return "\n".join(lines)

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return False
        return (
            self.id == other.id
            and self.role == other.role
            and self.content == other.content
            and self.requestor == other.requestor
            and self.error == other.error
        )


class MultiToolMessage(BaseModel):
    """
    Encapsulates multiple tool messages.
    """

    role: ToolRole = Field(description="The role of the message sender.")
    tool_messages: list[ToolMessage] = Field(description="The tool messages.")


APICompatibleMessage = SystemMessage | AssistantMessage | UserMessage | ToolMessage
Message = (
    SystemMessage | AssistantMessage | UserMessage | ToolMessage | MultiToolMessage
)

# Forward reference resolution for MetaEvent
from tau2.meta.schema import MetaEvent
ParticipantMessageBase.model_rebuild()
