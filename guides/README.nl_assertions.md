# NL Assertions Guide

## Overview

NL assertions are behavioral expectations evaluated by an LLM judge that reviews the conversation transcript. They focus on task-specific behaviors, compliance requirements, and process adherence rather than generic politeness or factual accuracy.


## Writing Guidelines

Writing good NL assertions is about capturing the specific behaviors that matter for your task, not generic politeness checks. Think about what the agent should actually do in this specific situation, not how they should generally behave.

The key is to focus on task-specific behaviors and compliance requirements. If your task involves a goal shift, you probably want to check that the agent didn't just give up and transfer to a human. If it's a payment task, you want to verify they confirmed details before processing. If it's a dispute, you want to see that they followed the proper escalation process.

Start with behaviors that could go wrong in your specific scenario. What are the common failure modes? What shortcuts might an agent take that would be problematic? What compliance requirements exist for this type of interaction?

### Universal Assertion for Goal Shift Tasks

For any task with goal shifts, include this assertion in every case:

```json
"nl_assertions": [
  "Agent did not transfer the customer to a human agent when the goal changed"
]
```

### Banking Domain Examples

For banking tasks involving sensitive information, focus on verification and security behaviors:

```json
"nl_assertions": [
  "Agent verified customer identity before accessing account information",
  "Agent confirmed payment details before processing the transaction",
  "Agent explained any fees or charges before the customer agreed to proceed"
]
```

For dispute or problem resolution tasks, focus on process compliance and proper handling:

```json
"nl_assertions": [
  "Agent gathered all necessary details about the disputed transaction before proceeding",
  "Agent explained the dispute resolution process and timeline to the customer",
  "Agent created a proper case reference number and provided it to the customer"
]
```

For account changes or service modifications, focus on proper authorization and confirmation:

```json
"nl_assertions": [
  "Agent confirmed the customer's identity before making any account changes",
  "Agent clearly explained what changes would be made and when they would take effect",
  "Agent obtained explicit customer consent before implementing the changes"
]
```

### Multi-Step Process Assertions

If your task involves multiple steps or a complex process, write assertions that check each critical step was handled properly:

```json
"nl_assertions": [
  "Agent verified the customer's eligibility for the requested service before proceeding",
  "Agent explained all terms and conditions relevant to the new service",
  "Agent confirmed the customer understood the billing implications before activation"
]
```

## What to Avoid

The best NL assertions are specific to what could go wrong in your exact scenario. Generic assertions like "agent was professional" don't help you understand where your agent is actually failing. Instead, think about the specific professional behaviors required for this task.

Avoid assertions that are really about factual accuracy or tool usage. Those belong in completions and actions respectively. NL assertions should be about process, compliance, verification, and task-specific behavioral requirements.

Remember that these assertions will be evaluated by an LLM judge looking at the conversation transcript. Make sure your assertions describe behaviors that can actually be detected from reading the conversation, not internal mental states or intentions.

## Best Practices

- Focus on specific, observable behaviors
- Write assertions that can be clearly detected from conversation transcript
- Target common failure modes for your specific task type
- Include compliance and verification requirements
- Avoid generic politeness or professionalism checks
- Make assertions measurable and specific to the task context


## Judge Prompt

### System Prompt
```
TASK
- You will be given a list of expected outcomes and a conversation that was collected during a test case run.
- The conversation is between an agent and a customer.
- Your job is to evaluate whether the agent satisfies each of the expected outcomes.
- Grade each expected outcome individually.

FORMAT
- Your response should be a JSON object with the following fields:
- `reasoning`: a short explanation for your classification
- `metExpectation`: `true` if the agent satisfies the expected outcomes, `false` otherwise
- `expectedOutcome`: repeat the expectation from the input that you are grading

Example response structure:
{
    "results": [
        {
            "expectedOutcome": "<one of the expected outcomes from the input>",
            "reasoning": "<reasoning trace>",
            "metExpectation": <false or true>,
        }
    ]
}
```

### User Prompt Format
```
conversation:
user: I need my account balance
assistant: I'd be happy to help! Can you verify your identity with your customer ID?
user: It's 12345
assistant: [calls get_customer_by_id(customer_id="12345")]
tool: Customer found: John Doe
assistant: Thank you John! Your current balance is $2,847.32. Is there anything else I can help you with?

nl_assertions:
["Agent verified customer identity before sharing account balance information", "Agent asked if the customer needed help with anything else"]
```