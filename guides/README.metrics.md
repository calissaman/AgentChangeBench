# Metrics v2 Guide

## Overview

The evaluation system has been overhauled from binary, brittle metrics to partial credit, multi-channel assessment. This provides better signal about agent capabilities and enables more nuanced analysis of performance patterns.

## Key Changes

### From Binary to Partial Scoring
Previous system used all-or-nothing evaluation where one failure meant total task failure. New system provides partial credit across all components.

### From Multiplicative to Weighted Rewards
Old reward calculation multiplied components together, causing cascade failures. New system uses weighted averages for balanced assessment.

### Enhanced Granularity
Metrics now break down by component, task, and failure type to identify specific improvement areas.

## Metrics Breakdown

### TSR v2 (Task Success Rate)
Multi-channel success assessment replacing simple binary thresholding.

```
TSR_overall = weighted_average(TSR_communicate_info, TSR_action, TSR_nl)
```

```
TSR_communicate_info = tasks_where_agent_outputted_required_information / total_tasks
```
Measures whether agents said the specific numbers or values from communicate_info. This comes from `evaluation_criteria.communicate_info`.

```
TSR_action = tasks_where_agent_used_correct_tools / total_tasks
```
Measures whether agents called the appropriate tools with correct parameters. This comes from `evaluation_criteria.actions`.

```
TSR_nl = tasks_where_agent_behaved_appropriately / total_tasks  
```
Measures whether agents met the behavioral expectations from nl_assertions. This comes from `evaluation_criteria.nl_assertions`.

```
OSR = tasks_where_environment_reached_expected_state / total_tasks
```
Environment-only success rate for cross-domain comparison, ignores agent communication. This comes from `evaluation_criteria.env_assertions`.

**Interpretation**: Overall success rate shows percentage of tasks that achieved acceptable outcomes. OSR provides cross-domain comparison using only environment outcomes.

### TUE v2 (Tool Usage Efficiency)
Simplified performance measurement focusing on tool selection and parameter accuracy.

```
TUE = 0.6 × T_correct + 0.4 × P_params
T_correct = correct_tool_calls / total_tool_calls
P_params = valid_parameter_calls / total_tool_calls
```

Latency component removed since tools are in-memory operations. Cost efficiency placeholder for future real-cost scenarios.

**Interpretation**: Tool correctness shows agent's ability to select appropriate tools from allowed options. Parameter accuracy shows agent's precision in providing correct parameters. Redundancy rate shows how often agents repeat recent tool calls unnecessarily.

### TCRR v2 (Tool-Call Redundancy Ratio)
Enhanced window-based redundancy detection with batch inefficiency detection.

```
TCRR = redundant_calls_in_window / total_calls
redundant_call = (same_tool_and_params within last 3 turns) OR 
                 (excess_calls_to_same_function > batch_threshold)
```

**Recent Enhancement**: Added intra-turn batch redundancy detection to catch inefficient patterns like making 5+ calls to the same function simultaneously. The batch threshold (default: 2) flags excess calls beyond reasonable limits.

**Example**: Agent makes 5 `get_reservation_details` calls in one turn → 3 flagged as redundant (5 - 2 threshold = 3).

**Interpretation**: Redundancy rate shows both traditional duplicate calls and modern batch inefficiency patterns. This catches real-world agent inefficiencies that the original metric missed.

### GSRT v2 (Goal Shift Recovery Time)
Multi-variant recovery assessment with task-level breakdown.

```
GSRT_ack = turns_until_agent_mentions_new_goal
GSRT_tool = turns_until_agent_uses_relevant_tool  
GSRT_outcome = turns_until_new_goal_achieved
Recovery_rate = successful_recoveries / total_shifts
```

**Recent Fix**: Corrected recovery rate calculation to exclude transfers-to-human from successful recoveries. Previously, transfers were incorrectly counted as agent successes, inflating recovery rates.

**Key Logic**: `recovery_successful = acknowledged AND not transferred_to_human`

Three recovery measurements capture different aspects of goal shift handling. Transfer-to-human is not considered acknowledgment.

**Interpretation**: Acknowledgment median shows how quickly agents recognize goal changes. Tool usage median shows how quickly agents start using relevant tools for new goals. Outcome success median shows how quickly agents actually achieve new goals. Transfer-to-human rate shows how often agents give up and transfer instead of adapting.

### Reward System v2
Weighted average replacing broken multiplicative combination.

```
final_reward = (0.5 × COMMUNICATE_INFO) + (0.3 × ACTION) + (0.2 × NL_ASSERTION)

COMMUNICATE_INFO = matched_communicate_info_strings / total_communicate_info_strings
ACTION = (0.5 × tool_correctness + 0.5 × param_correctness) per action_id
NL_ASSERTION = met_assertions / total_assertions
```

COMMUNICATE_INFO component validates factual accuracy. All components use partial scoring.

**Interpretation**: Communicate_info scores show how well agents communicate required information. Action scores show how well agents select and use appropriate tools. NL assertion scores show how well agents follow behavioral expectations.

## Recent Improvements & Bug Fixes

### Enhanced TCRR Batch Detection
- **Problem**: Original TCRR missed inefficient batch patterns (e.g., 5 consecutive calls to same function)
- **Solution**: Added intra-turn batch redundancy detection with configurable threshold
- **Configuration**: `TCRR_BATCH_THRESHOLD = 2` in metrics config

### GSRT Recovery Rate Correction  
- **Problem**: Transfers-to-human were incorrectly counted as successful agent recoveries
- **Solution**: Modified logic to exclude transfers from recovery rate calculation
- **Impact**: More accurate measurement of actual agent adaptability vs. giving up

### TSR Proportional Rebalancing Verification
- **Issue**: Cached results sometimes show incorrect 50/50 weight distributions
- **Status**: Core proportional rebalancing logic confirmed working correctly
- **Note**: Cached simulation results may show outdated weights; recomputation resolves this

### Action Scoring Accuracy Confirmation
- **Issue**: Some cached results showed 0.75 scores for perfect tool usage
- **Status**: Current action scoring logic confirmed working correctly 
- **Note**: Modern scoring properly handles multiple tool calls and finds best matches

## Complete Metrics Output

```json
{
  "evaluation_summary": {
    "overall_success_rate": 0.78,
    "avg_reward": 0.82,
    "total_tasks": 50,
    "total_simulations": 150,
    "evaluation_date": "2025-01-20T14:30:00Z",
    "evaluator_version": "2.1.0"
  },

  "tsr_v2": {
    "overall": 0.78,
    "by_channel": {
      "communicate_info": 0.85,
      "action": 0.72,
      "nl_assertion": 0.77
    },
    "osr": 0.81,
    "partial_credit_rates": {
      "communicate_info_pass_rate": 0.89,
      "action_pass_rate": 0.76,
      "nl_pass_rate": 0.82
    }
  },

  "tue_v2": {
    "overall": 0.74,
    "components": {
      "tool_correctness": 0.79,
      "parameter_accuracy": 0.68
    },
    "coverage": {
      "tool_calls_analyzed": 1247,
      "param_validation_rate": 0.94
    }
  },

  "tcrr_v2": {
    "overall": 0.05,
    "redundant_calls": 62,
    "total_calls": 1247,
    "window_size": 3,
    "batch_threshold": 2,
    "redundancy_breakdown": {
      "cross_turn_duplicates": 0.02,
      "intra_turn_batch": 0.03,
      "total_redundancy": 0.05
    }
  },

  "gsrt_v2": {
    "aggregate": {
      "median_recovery_time": 4.2,
      "recovery_rate": 0.73,
      "never_recovered_rate": 0.27,
      "transfer_to_human_rate": 0.18
    },
    "by_variant": {
      "acknowledgment": {
        "median": 2.8,
        "ci_95": [2.1, 3.5]
      },
      "tool_usage": {
        "median": 4.2,
        "ci_95": [3.4, 5.1]
      },
      "outcome_success": {
        "median": 7.8,
        "ci_95": [6.2, 9.4]
      }
    },
    "by_task": {
      "banking_001": {
        "shifts": 3,
        "median_gsrt": 3.5,
        "recovery_rate": 1.0
      },
      "banking_002": {
        "shifts": 2,
        "median_gsrt": 6.0,
        "recovery_rate": 0.5
      }
    }
  },

  "component_breakdown": {
    "communicate_info_metrics": {
      "avg_score": 0.85,
      "exact_matches": 0.78,
      "total_communicate_info_checks": 127
    },
    "action_metrics": {
      "avg_score": 0.72,
      "tool_correctness": 0.79,
      "param_correctness": 0.65,
      "total_action_checks": 94
    },
    "nl_assertion_metrics": {
      "avg_score": 0.77,
      "total_assertions": 203
    }
  },

  "task_level_breakdown": {
    "by_task_performance": [
      {
        "task_id": "banking_001",
        "success_rate": 0.90,
        "avg_reward": 0.88,
        "communicate_info_score": 0.95,
        "action_score": 0.85,
        "nl_score": 0.82
      },
      {
        "task_id": "banking_002",
        "success_rate": 0.65,
        "avg_reward": 0.71,
        "communicate_info_score": 0.78,
        "action_score": 0.60,
        "nl_score": 0.75
      }
    ]
  },

  "cross_cutting_analysis": {
    "reward_weights_used": {
      "COMMUNICATE_INFO": 0.5,
      "ACTION": 0.3,
      "NL_ASSERTION": 0.2
    },
    "partial_scoring_impact": {
      "tasks_benefiting_from_partial": 34,
      "avg_reward_increase": 0.18
    },
    "coverage_statistics": {
      "tasks_with_communicate_info": 47,
      "tasks_with_actions": 50,
      "tasks_with_nl_assertions": 48,
      "tasks_with_goal_shifts": 23
    }
  },

  "raw_statistics": {
    "total_tool_calls": 1247,
    "total_messages": 3421,
    "avg_conversation_length": 22.8,
    "total_evaluation_time": "14m 32s"
  }
}
```

**Task-Level Patterns**: Individual task performance breakdown enables identification of consistently difficult tasks and common failure patterns across the domain.

This metrics system provides actionable insights for improving agent performance by identifying specific areas of weakness while giving credit for partial progress.
