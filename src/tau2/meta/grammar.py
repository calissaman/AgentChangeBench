import re
from typing import Dict, Optional, Tuple

# Strict regex for meta tags - must be first line, max 256 chars
# Updated to allow empty body for proper error handling
META_RE = re.compile(r"^\s*<meta>(?P<body>[^<]{0,256})</meta>\s*$", re.ASCII)


def parse_meta_line(line: str) -> Tuple[Optional[Dict], Optional[str]]:
    r"""
    Parse a meta tag line according to the deterministic grammar.

    Grammar:
    <MetaLine>  ::= "^<meta>" <Event> (WS <KVPair>)* "</meta>$"
    <Event>     ::= "GOAL_SHIFT" | "PARK" | "RESUME"
    <KVPair>    ::= <Key> "=" <Value>
    <Key>       ::= "seq" | "from" | "to" | "reason" | "task" | "note"
    <Value>     ::= <Token> | "\"" <Quoted> "\""
    <Token>     ::= [A-Za-z0-9_\-.:/]+
    <Quoted>    ::= any chars except quote and newline

    Args:
        line: The line to parse (should be first line of message)

    Returns:
        Tuple of (meta_dict, error) where meta_dict is a structured dict
        or None on failure. Error is a string describing the failure.
        Enforces first-line-only at call site.
    """
    line = line.strip()
    if not line:
        return None, "NO_META_OR_BAD_TAG"

    # Check if line matches meta tag pattern
    m = META_RE.match(line)
    if not m:
        return None, "NO_META_OR_BAD_TAG"

    body = m.group("body").strip()
    if not body:
        return None, "EMPTY_META_BODY"

    # Parse the body using a more sophisticated approach to handle quoted values
    # Split by spaces but keep quoted strings intact
    parts = []
    current_part = ""
    in_quotes = False
    i = 0

    while i < len(body):
        char = body[i]
        if char == '"' and not in_quotes:
            in_quotes = True
            current_part += char
        elif char == '"' and in_quotes:
            in_quotes = False
            current_part += char
        elif char == " " and not in_quotes:
            if current_part:
                parts.append(current_part)
                current_part = ""
        else:
            current_part += char
        i += 1

    if current_part:
        parts.append(current_part)

    if not parts:
        return None, "EMPTY_META_BODY"

    event = parts[0]

    # Validate event type
    if event not in ["GOAL_SHIFT", "PARK", "RESUME"]:
        return None, f"INVALID_EVENT:{event}"

    # Parse key-value pairs
    kvs = {}
    for p in parts[1:]:
        if "=" not in p:
            return None, f"BAD_KV:{p}"

        k, v = p.split("=", 1)

        # Handle quoted values
        if v.startswith('"') and v.endswith('"') and len(v) >= 2:
            v = v[1:-1]  # Remove quotes
        # Validate token format for unquoted values
        elif not re.match(r"^[A-Za-z0-9_\-.:/]+$", v):
            return None, f"INVALID_TOKEN_VALUE:{v}"

        # Validate key
        if k not in ["seq", "from", "to", "reason", "task", "note"]:
            return None, f"INVALID_KEY:{k}"

        kvs[k] = v

    return {"event": event, **kvs}, None


def validate_max_length(line: str) -> bool:
    """Check if meta line exceeds maximum allowed length."""
    return len(line.strip()) <= 256


def is_meta_line(line: str) -> bool:
    """Quick check if a line looks like a meta tag."""
    line = line.strip()
    return line.startswith("<meta>") and line.endswith("</meta>")
