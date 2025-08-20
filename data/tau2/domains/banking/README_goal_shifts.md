# Banking Goal Shift System

This document explains how to use the declarative goal shift system for banking domain tasks.

## Overview

The banking domain supports declarative goal shifts that can be specified per-task using a structured `goal_shifts` field in the user scenario. The banking user simulator executes natural goal transitions based on conversation flow and mandatory progression rules.

**Note**: Banking users do not have tools. All user information (account balances, card status, transaction details, etc.) is provided through the `known_info` section of the user scenario. Users share this information naturally when asked by the agent.

## Components

### 1. User Personas (`user_personas.json`)
- Contains 5 personas: `EASY_1`, `EASY_2`, `MEDIUM_1`, `HARD_1`, `MEDIUM_2`
- Each persona is automatically injected into tasks during runtime
- Personas include personality traits, expertise levels, and behavioral patterns

### 2. Banking User Policy (`user_policy.md`)  
- Acts as the system prompt for the banking user simulator
- Focuses on information sharing rather than tool usage
- Includes banking-specific context and behavior guidelines
- Contains "Internal Goal Sequence System" with mandatory progression rules

### 3. Banking User Simulator (`user_simulator.py`)
- Domain-specific user simulator that uses the banking user policy
- Registered as `banking_user_simulator` in the registry
- Handles goal shifts naturally through conversation without tools

## Creating Goal Shift Tasks

### 1. Task Structure

```json
{
  "id": "goalshift_banking_001",
  "user_scenario": {
    "persona": "EASY_1",
    "instructions": {
      "domain": "banking",
      "reason_for_call": "Issues logging in, suspicious card activity",
      "known_info": "You are Alex Morgan with phone number 555-123-7890, date of birth 1980-04-15, and email alex.morgan@email.com. You have a checking account (account #12345678) with a current balance of $2,847.32. You have an active debit card ending in 4567 that expires 12/2025. You remember seeing a recent charge for $89.99 from 'UNKNOWN MERCHANT XYZ' that you don't recognize from 3 days ago. Your account has 2FA enabled and you usually receive codes via SMS.",
      "unknown_info": "Login reason failure, latest transactions, how to dispute",
      "task_instructions": "Assist the user through login issues, then surface recent transactions and resolve a disputed charge."
    },
    "goal_shifts": {
      "required_shifts": 2,
      "goals": ["authentication", "transactions", "dispute"]
    }
  }
}
```

### 2. Information Handling

**Known Information**: Include all relevant details in `known_info`:
- Personal details (name, phone, email, DOB)
- Account information (numbers, balances, types)
- Card details (numbers, expiration, status)
- Recent transaction information
- Authentication details (2FA status, PIN)
- Specific details relevant to the task goals

**Unknown Information**: List in `unknown_info` what the user doesn't know or remember

### 3. Evaluation Criteria

Focus on **agent actions** rather than user tool calls:
```json
"evaluation_criteria": {
  "actions": [
    {
      "action_id": "act_1",
      "requestor": "assistant",
      "name": "get_customer_by_phone", 
      "arguments": {"phone_number": "555-123-7890"},
      "info": "Agent authenticates user using phone number lookup"
    }
  ]
}
```

### 2. Goal Shifts Configuration

The `goal_shifts` field defines the goal sequence:

```json
"goal_shifts": {
  "required_shifts": 2,
  "goals": ["authentication", "transactions", "dispute"]
}
```

- **`required_shifts`**: Number of goal transitions required (always `goals.length - 1`)
- **`goals`**: Array of goal names in sequence order
- **Common goal types**: `authentication`, `transactions`, `dispute`, `account_info`, `payments`, `cards`

### 3. Important: Customer Data in `known_info`

**Critical**: Always include the customer's actual data in `known_info` so the user simulator can provide correct authentication details:

```json
"known_info": "You are Alex Morgan with phone number 555-123-7890, date of birth 1980-04-15, and email alex.morgan@email.com. You have a checking account and an active debit card."
```

This data must match what's in the banking database (`db.toml`).

### 4. Initialization Setup

Include both `initialization_data` and `initialization_actions` to set up the user tools database:

```json
"initial_state": {
  "initialization_data": {
    "agent_data": null,
    "user_data": {
      "authenticated": false,
      "has_2fa_enabled": true,
      "phone_number": "555-123-7890",
      "customer_id": "cust_001"
    }
  },
  "initialization_actions": [
    {
      "env_type": "user",
      "func_name": "update_user",
      "arguments": {
        "authenticated": false,
        "phone_number": "555-123-7890",
        "customer_id": "cust_001"
      }
    }
  ]
}
```

## How Goal Shifts Work

### 1. Natural Transition System

The user simulator follows these progression rules:

- **Maximum 4 exchanges per goal** - prevents getting stuck in authentication loops
- **Forced progression triggers** - moves to next goal when agent offers transfer, asks for alternative info, or says they can't help
- **Natural transition phrases** - uses phrases like "Before we transfer, I also wanted to ask about..."

### 2. Goal Sequence Example

With `goals: ["authentication", "transactions", "dispute"]`:

1. **GOAL 1 (authentication)**: User starts with login issues, goes through verification
2. **GOAL 2 (transactions)**: User shifts to "I also wanted to check recent transactions"  
3. **GOAL 3 (dispute)**: User shifts to "I noticed a charge I don't recognize"

### 3. Transition Triggers

Shifts happen naturally when:
- Agent provides solution steps (even if not fully resolved)
- Agent offers transfer or asks "anything else?"
- User receives helpful information
- After 4 user messages on same goal (forced progression)

## Running Goal Shift Tasks

```bash
# Use the banking-specific user simulator
uv run tau2 run --domain banking --agent llm_agent --user banking_user_simulator --num-tasks 1

# View results
uv run tau2 view <simulation_file>
```

## Example Task Templates

### Authentication → Transactions
```json
{
  "goal_shifts": {
    "required_shifts": 1,
    "goals": ["authentication", "transactions"]
  },
  "task_instructions": "Help user log in, then check recent account activity."
}
```

### Full Banking Flow
```json
{
  "goal_shifts": {
    "required_shifts": 2, 
    "goals": ["authentication", "transactions", "dispute"]
  },
  "task_instructions": "Assist with login issues, then review transactions and resolve disputed charge."
}
```

### Account Services
```json
{
  "goal_shifts": {
    "required_shifts": 1,
    "goals": ["account_info", "cards"]
  },
  "task_instructions": "Help user check account details, then assist with card issues."
}
```

## Key Design Principles

### ✅ **No Information Leakage**
- Agent never sees goal markers or shift indicators
- All transitions appear natural from agent's perspective
- Tests agent's genuine adaptability to changing user needs

### ✅ **Declarative Configuration**
- Simple `goal_shifts` object defines behavior
- No complex turn counting or deterministic triggers
- Easy to modify and maintain

### ✅ **Natural Conversation Flow**
- Uses realistic transition phrases
- Follows banking conversation patterns
- Respects user personas and expertise levels

### ✅ **Robust Progression** 
- Mandatory rules prevent infinite loops
- Handles agent transfer scenarios gracefully
- Ensures all goals get addressed

## Evaluation Considerations

Goal shift tasks test the agent's ability to:
- Handle multiple customer needs in one conversation
- Adapt tool usage as goals change (authentication → transaction lookup → dispute filing)
- Maintain context across goal transitions
- Provide appropriate solutions for each goal category

The system ensures realistic, evaluable goal shifts while maintaining natural conversation flow for robust agent testing. 