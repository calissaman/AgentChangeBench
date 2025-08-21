# Communicate Info Guide

## Overview

Communicate info defines the specific strings or information that the agent must say in their responses. These are exact text matches that validate factual accuracy and information delivery. Usually, this is used to check that the agent has correctly executed a tool call. 

Therefore, we can say that communicate_info is a list of strings that are expected to be found in the agent's response.

## How to Write

Include only the exact numbers, amounts, or specific values the agent should communicate:

```json
"communicate_info": [
  "$2,847.32",
  "Premium", 
  "TX789123"
]
```

## Examples

```json
"communicate_info": [
  "$150.00",
  "January 15th",
  "10GB"
]
```

## Scoring

```
COMMUNICATE_reward = matched_strings / total_strings
```

Each string is worth equal points. Agent gets partial credit for saying some but not all required values. 