# Actions Guide

## Overview

Actions define the tool usage expectations for agents in a task. They specify which tools the agent should use and with what parameters to successfully complete the task objectives.

## What's Changing and Why

### Old Format Problems
The previous action format was rigid and brittle:

```json
{
  "action_id": "get_user_details_1",
  "requestor": "assistant",
  "name": "get_user_details",
  "arguments": {"user_id": "sophia_silva_7557"},
  "compare_args": ["user_id"]
}
```

Issues: path-dependent, brittle parameter matching, single solution bias, poor failure signal.

### New Format Benefits
The new format is flexible and path-agnostic:

```json
{
  "action_id": "lookup_customer",
  "allowed_tools": [
    {
      "function_name": "get_user_details",
      "params": {"user_id": "sophia_silva_7557"}
    },
    {
      "function_name": "get_customer_by_phone",
      "params": {"phone_number": "+1234567890"}
    }
  ]
}
```

Benefits: multiple valid approaches, partial credit scoring, realistic flexibility, clearer failure analysis.

## Metrics That Use Actions

### TSR v2 (Task Success Rate)
```
TSR_action = tasks_with_correct_tool_usage / total_tasks
```

### TUE v2 (Tool Usage Efficiency) 
```
TUE = 0.6 × T_correct + 0.4 × P_params
T_correct = tools_matching_allowed_tools / total_tool_calls
P_params = calls_with_correct_parameters / total_tool_calls
```

### ACTION Component (Reward Calculation)
```
per_action_score = tool_score + param_score
tool_score = 0.5 if agent_used_any_allowed_tool else 0.0
param_score = 0.5 if tool_called_with_correct_params else 0.0
ACTION_reward = sum(per_action_scores) / total_action_ids
```

## How to Write Actions

### Basic Structure
```json
{
  "action_id": "descriptive_step_name",
  "allowed_tools": [
    {
      "function_name": "tool_name",
      "params": {"required_param": "expected_value"}
    }
  ]
}
```

### Parameter Handling
**Only include required parameters in the params object.** Optional parameters are omitted from scoring entirely - agents can include them or not without penalty.

```json
{
  "function_name": "search_transactions",
  "params": {
    "account_id": "acc_123",
    "start_date": "2024-01-01"
  }
  // Optional params like "limit", "sort_order" are omitted
  // Agent can use them without affecting scoring
}
```

## Writing Checklist

### Action ID Naming
- ✅ Use logical goal names: `"lookup_customer"`, `"process_payment"`
- ❌ Avoid implementation details: `"get_customer_by_id_1"`, `"call_api"`

### Allowed Tools Strategy  
- ✅ Include all valid approaches to achieve the goal
- ✅ Use single tool when only one approach makes sense
- ✅ Use multiple tools when different methods are equally valid
- ❌ Don't include tools that don't achieve the action's goal

### Parameter Guidelines
- ✅ Include only required parameters that must match exactly
- ✅ Use exact values when specified in task scenario
- ✅ Use descriptive names when values come from conversation
- ❌ Don't include optional parameters (they're ignored in scoring)
- ❌ Don't include parameters that aren't essential to the action

### Action Granularity
- ✅ One action per logical step or goal
- ✅ Group related tool calls into single actions
- ❌ Don't create separate actions for each tool call
- ❌ Don't create actions for trivial intermediate steps

## Examples

### Single Tool Action
```json
{
  "action_id": "get_account_balance",
  "allowed_tools": [
    {
      "function_name": "get_account_balance",
      "params": {"account_id": "acc_12345"}
    }
  ]
}
```

### Multiple Valid Approaches
```json
{
  "action_id": "lookup_customer",
  "allowed_tools": [
    {
      "function_name": "get_customer_by_id",
      "params": {"customer_id": "cust_789"}
    },
    {
      "function_name": "get_customer_by_phone",
      "params": {"phone_number": "+1234567890"}
    }
  ]
}
```

### Complete Task Example
Banking dispute task:

```json
"actions": [
  {
    "action_id": "identify_customer",
    "allowed_tools": [
      {
        "function_name": "get_customer_by_phone",
        "params": {"phone_number": "+1555123456"}
      },
      {
        "function_name": "get_customer_by_id",
        "params": {"customer_id": "cust_789"}
      }
    ]
  },
  {
    "action_id": "locate_disputed_transaction",
    "allowed_tools": [
      {
        "function_name": "get_transaction_by_id",
        "params": {"transaction_id": "tx_12345"}
      }
    ]
  },
  {
    "action_id": "create_dispute_case",
    "allowed_tools": [
      {
        "function_name": "file_dispute",
        "params": {"transaction_id": "tx_12345", "reason": "unauthorized"}
      }
    ]
  }
]
``` 