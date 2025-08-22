import math
import re
import statistics
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel

from tau2.data_model.message import Message, ToolMessage, AssistantMessage, UserMessage
from tau2.data_model.simulation import Results, SimulationRun


def is_successful(reward: float) -> bool:
    """
    Check if the reward is successful.
    """
    return (1 - 1e-6) <= reward <= (1 + 1e-6)


def safe_float(val, default: float = 0.0) -> float:
    """Convert value safely to float."""
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def normalized_params(params: dict) -> tuple:
    """Recursively normalize parameters for consistency."""
    if not isinstance(params, dict):
        return str(params)

    normalized_items = []
    for k, v in sorted(params.items()):
        if isinstance(v, dict):
            normalized_items.append((k, normalized_params(v)))
        elif isinstance(v, list):
            try:
                sorted_v = tuple(sorted(v))
            except TypeError:
                sorted_v = tuple(str(item) for item in v)
            normalized_items.append((k, sorted_v))
        else:
            normalized_items.append((k, v))
    return tuple(normalized_items)


def extract_tool_calls_from_messages(messages: list[Message]) -> list[dict]:
    """
    Extract tool call metrics from tau2 message objects.

    Determines correctness and parameter validity according to AgentChangeBench specification:
    - T_correct: Tool executed without errors
    - P_params: Tool called with valid & optimal parameters
    """
    tool_calls = []

    # Create mapping of tool call IDs to their results
    tool_results = {}
    for msg in messages:
        if isinstance(msg, ToolMessage):
            # Tool message contains the result of a tool call
            tool_results[msg.id] = {
                "success": not msg.error,
                "error": msg.error,
                "content": msg.content,
            }

    for msg in messages:
        # Look for tool calls in AssistantMessage and UserMessage objects
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tool_call in msg.tool_calls:
                # Determine correctness based on whether the tool executed without errors
                correct = True
                if hasattr(tool_call, "id") and tool_call.id in tool_results:
                    correct = tool_results[tool_call.id]["success"]

                # Determine parameter validity
                # For now, assume parameters are valid if:
                # 1. The tool call was attempted (it exists)
                # 2. The tool didn't fail due to parameter issues
                params_valid = True
                if hasattr(tool_call, "id") and tool_call.id in tool_results:
                    result = tool_results[tool_call.id]
                    if result["error"] and result.get("content"):
                        # Check if error message indicates parameter issues
                        error_content = str(result["content"]).lower()
                        if any(
                            keyword in error_content
                            for keyword in [
                                "invalid parameter",
                                "missing parameter",
                                "parameter error",
                                "bad parameter",
                                "invalid argument",
                                "missing argument",
                            ]
                        ):
                            params_valid = False

                # Calculate latency if timestamps are available
                latency = 0.0
                if (
                    hasattr(msg, "timestamp")
                    and hasattr(tool_call, "id")
                    and tool_call.id in tool_results
                ):
                    # Could implement timestamp-based latency calculation here
                    pass

                tool_call_metric = {
                    "name": tool_call.name,
                    "params": tool_call.arguments,  # ToolCall uses 'arguments' not 'params'
                    "cost": safe_float(msg.cost) if hasattr(msg, "cost") else 0.0,
                    "latency": latency,
                    "correct": correct,
                    "params_valid": params_valid,
                }
                tool_calls.append(tool_call_metric)

    return tool_calls


def extract_domain_from_task_id(task_id: str) -> str:
    """Extract domain name from task_id."""
    parts = task_id.split("_")
    if len(parts) > 0:
        return parts[0]
    return "unknown"


class AgentMetrics(BaseModel):
    avg_reward: float
    pass_hat_ks: dict[int, float]
    avg_agent_cost: float
    # New AgentChangeBench metrics
    tsr: float  # Task Success Rate
    tue: float  # Tool Usage Efficiency
    tcrr: float  # Tool-Call Redundancy Ratio
    num_tool_calls: int
    # Enhanced TSR metrics (multi-channel breakdown)
    tsr_communicate_info: float = 0.0
    tsr_action: float = 0.0
    tsr_nl_assertion: float = 0.0
    tsr_weights: dict = {}  # Weights used in TSR calculation
    tsr_by_task: dict = {}  # Task-level TSR breakdown
    # Enhanced TUE metrics
    tue_tool_correctness: float = 0.0
    tue_param_accuracy: float = 0.0
    tue_correct_calls: int = 0
    tue_valid_param_calls: int = 0
    tue_by_task: dict = {}  # Task-level TUE breakdown
    # Enhanced TCRR metrics
    tcrr_window_size: int = 3
    tcrr_total_calls: int = 0
    tcrr_redundant_calls: int = 0
    tcrr_by_task: dict = {}  # Task-level TCRR breakdown
    # Enhanced GSRT metrics
    gsrt_median_ack: Optional[float] = None
    gsrt_median_tool: Optional[float] = None
    gsrt_median_outcome: Optional[float] = None
    gsrt_num_shifts: int = 0
    gsrt_recovery_rate: float = 0.0
    gsrt_transfer_rate: float = 0.0
    gsrt_never_recovered_rate: float = 0.0
    gsrt_by_task: dict = {}  # Task-level breakdown

    # Coverage Statistics
    tasks_with_communicate_info: int = 0
    tasks_with_actions: int = 0
    tasks_with_nl_assertions: int = 0
    tasks_with_goal_shifts: int = 0

    # Component Breakdown
    communicate_info_avg_score: Optional[float] = None
    communicate_info_exact_matches: Optional[float] = None
    total_communicate_info_checks: int = 0

    action_avg_score: Optional[float] = None
    action_tool_correctness: Optional[float] = None
    action_param_correctness: Optional[float] = None
    total_action_checks: int = 0

    nl_assertion_avg_score: Optional[float] = None
    total_nl_assertions: int = 0

    # Partial Scoring Impact
    tasks_benefiting_from_partial: int = 0
    avg_reward_increase: float = 0.0

    def as_dict(self) -> dict:
        data = {
            "avg_reward": self.avg_reward,
            "avg_agent_cost": self.avg_agent_cost,
            "tsr": self.tsr,
            "tue": self.tue,
            "tcrr": self.tcrr,
            "num_tool_calls": self.num_tool_calls,
            "tsr_communicate_info": self.tsr_communicate_info,
            "tsr_action": self.tsr_action,
            "tsr_nl_assertion": self.tsr_nl_assertion,
            "tsr_weights": self.tsr_weights,
            "tsr_by_task": self.tsr_by_task,
            "tue_tool_correctness": self.tue_tool_correctness,
            "tue_param_accuracy": self.tue_param_accuracy,
            "tue_correct_calls": self.tue_correct_calls,
            "tue_valid_param_calls": self.tue_valid_param_calls,
            "tue_by_task": self.tue_by_task,
            "tcrr_window_size": self.tcrr_window_size,
            "tcrr_total_calls": self.tcrr_total_calls,
            "tcrr_redundant_calls": self.tcrr_redundant_calls,
            "tcrr_by_task": self.tcrr_by_task,
            "gsrt_median_ack": self.gsrt_median_ack,
            "gsrt_median_tool": self.gsrt_median_tool,
            "gsrt_median_outcome": self.gsrt_median_outcome,
            "gsrt_num_shifts": self.gsrt_num_shifts,
            "gsrt_recovery_rate": self.gsrt_recovery_rate,
            "gsrt_transfer_rate": self.gsrt_transfer_rate,
            "gsrt_never_recovered_rate": self.gsrt_never_recovered_rate,
            "gsrt_by_task": self.gsrt_by_task,
        }
        for k, v in self.pass_hat_ks.items():
            data[f"pass_hat_{k}"] = v
        return data


def pass_hat_k(num_trials: int, success_count: int, k: int) -> float:
    """
    Compute the pass^k metric for the given number of trials, success count, and k.
    from https://arxiv.org/pdf/2406.12045
    Args:
        num_trials: The number of trials.
        success_count: The number of successful trials.
        k: The number of trials to consider.
    Returns:
        The pass^k metric.
    """
    if num_trials < k:
        raise ValueError(f"Number of trials {num_trials} is less than k {k}.")
    return math.comb(success_count, k) / math.comb(num_trials, k)


def get_metrics_df(results: Results) -> tuple[pd.DataFrame, int]:
    """
    Convert the results to a dataframe and add a column for success.
    Checks that all simulations have the same number of trials.
    Returns the maximum number of trials that can be used for pass^k metrics.
    """
    df = results.to_df()
    df["success"] = df.reward.apply(is_successful)
    if len(df.info_num_trials.unique()) > 1:
        logger.warning(
            f"All simulations must have the same number of trials. Found {df.info_num_trials.unique()}"
        )
    max_k = df.info_num_trials.max()

    task_ids_counts = [(tid, count) for tid, count in df.task_id.value_counts().items()]
    task_ids_counts.sort(key=lambda x: x[1])
    min_k = task_ids_counts[0][1]
    if min_k < max_k:
        logger.warning(
            f"The minimum number of trials for a task is {min_k}, which is less than the expected number of trials {max_k}. Setting max k to {min_k}."
        )
        max_k = min_k
    return df, max_k


def compute_metrics_simple(results: Results) -> AgentMetrics:
    """
    Compute metrics for the agent with simpler approach that doesn't rely on task evaluation criteria.
    - average reward
    - TSR (Task Success Rate)
    - TUE (Tool Usage Efficiency)
    - TCRR (Tool-Call Redundancy Ratio)
    - GSRT (Goal Shift Recovery Time)
    """
    # Simple reward calculation
    rewards = [
        sim.reward_info.reward if sim.reward_info else 0.0
        for sim in results.simulations
    ]
    avg_reward = np.mean(rewards) if rewards else 0.0

    # Simple cost calculation
    agent_costs = [
        sim.agent_cost if sim.agent_cost else 0.0 for sim in results.simulations
    ]
    avg_agent_cost = np.mean(agent_costs) if agent_costs else 0.0

    # Pass^k metrics - simplified
    pass_hat_ks = {}
    if results.simulations:
        # Group by task_id and compute success rates
        task_groups = {}
        for sim in results.simulations:
            task_id = sim.task_id
            if task_id not in task_groups:
                task_groups[task_id] = []
            task_groups[task_id].append(
                is_successful(sim.reward_info.reward if sim.reward_info else 0.0)
            )

        # Compute pass^k for k=1 (basic success rate)
        if task_groups:
            task_success_rates = []
            for task_id, successes in task_groups.items():
                success_rate = sum(successes) / len(successes)
                task_success_rates.append(success_rate)
            pass_hat_ks[1] = np.mean(task_success_rates)

    # Compute enhanced TSR metrics with multi-channel breakdown
    try:
        from tau2.metrics.tsr import compute_tsr_enhanced

        tsr_result = compute_tsr_enhanced(results.tasks, results.simulations)
        tsr = tsr_result.overall_tsr
        tsr_communicate_info = tsr_result.communicate_info.success_rate
        tsr_action = tsr_result.action.success_rate
        tsr_nl_assertion = tsr_result.nl_assertion.success_rate
        tsr_weights = tsr_result.weights
        tsr_by_task = tsr_result.by_task

    except Exception as e:
        logger.warning(f"Enhanced TSR computation failed, falling back to simple: {e}")
        # Fallback to simple TSR calculation
        success_count = sum(
            1
            for sim in results.simulations
            if sim.reward_info and safe_float(sim.reward_info.reward) > 0
        )
        tsr = success_count / len(results.simulations) if results.simulations else 0.0
        tsr_communicate_info = 0.0
        tsr_action = 0.0
        tsr_nl_assertion = 0.0
        tsr_weights = {"communicate_info": 0.5, "action": 0.3, "nl_assertion": 0.2}
        tsr_by_task = {}

    # Extract all tool calls
    all_tool_calls = []
    for run in results.simulations:
        tool_calls = extract_tool_calls_from_messages(run.messages)
        all_tool_calls.extend(tool_calls)

    num_tool_calls = len(all_tool_calls)

    if all_tool_calls:
        # Calculate enhanced TCRR with window-based approach
        try:
            from tau2.metrics.tcrr import compute_tcrr_enhanced, compute_tcrr_by_task

            tcrr_result = compute_tcrr_enhanced(results.simulations, window_size=3)
            tcrr = tcrr_result.redundancy_ratio
            tcrr_window_size = tcrr_result.window_size
            tcrr_total_calls = tcrr_result.total_calls
            tcrr_redundant_calls = tcrr_result.redundant_calls

            # Get task-level breakdown
            tcrr_by_task_results = compute_tcrr_by_task(
                results.simulations, window_size=3
            )
            tcrr_by_task = {
                task_id: {
                    "redundancy_ratio": result.redundancy_ratio,
                    "total_calls": result.total_calls,
                    "redundant_calls": result.redundant_calls,
                }
                for task_id, result in tcrr_by_task_results.items()
            }
        except Exception as e:
            logger.warning(
                f"Enhanced TCRR computation failed, falling back to simple: {e}"
            )
            # Fallback to simple TCRR calculation
            seen_calls = set()
            duplicate_count = 0
            for call in all_tool_calls:
                try:
                    params_norm = normalized_params(call["params"])
                    identity = (call["name"], params_norm)
                    if identity in seen_calls:
                        duplicate_count += 1
                    else:
                        seen_calls.add(identity)
                except Exception:
                    identity = (call["name"], str(call["params"]))
                    if identity in seen_calls:
                        duplicate_count += 1
                    else:
                        seen_calls.add(identity)
            tcrr = duplicate_count / len(all_tool_calls) if all_tool_calls else 0.0
            tcrr_window_size = 3
            tcrr_total_calls = len(all_tool_calls)
            tcrr_redundant_calls = int(tcrr * len(all_tool_calls))
            tcrr_by_task = {}

        # Calculate enhanced TUE without latency component
        try:
            from tau2.metrics.tue import (
                compute_tue_enhanced_for_simulations,
                compute_tue_by_task,
            )

            tue_result = compute_tue_enhanced_for_simulations(results.simulations)
            tue = tue_result.overall_tue
            tue_tool_correctness = tue_result.tool_correctness
            tue_param_accuracy = tue_result.param_accuracy
            tue_correct_calls = tue_result.correct_calls
            tue_valid_param_calls = tue_result.valid_param_calls

            # Get task-level breakdown
            tue_by_task_results = compute_tue_by_task(results.simulations)
            tue_by_task = {
                task_id: {
                    "overall_tue": result.overall_tue,
                    "tool_correctness": result.tool_correctness,
                    "param_accuracy": result.param_accuracy,
                    "total_calls": result.total_calls,
                    "correct_calls": result.correct_calls,
                    "valid_param_calls": result.valid_param_calls,
                }
                for task_id, result in tue_by_task_results.items()
            }
        except Exception as e:
            logger.warning(
                f"Enhanced TUE computation failed, falling back to simple: {e}"
            )
            # Fallback to original TUE calculation
            all_costs = [call["cost"] for call in all_tool_calls if call["cost"] > 0]
            all_latencies = [
                call["latency"] for call in all_tool_calls if call["latency"] > 0
            ]
            cost_cap = np.percentile(all_costs, 95) if all_costs else 1.0
            latency_cap = np.percentile(all_latencies, 95) if all_latencies else 1.0
            cost_cap = max(cost_cap, 0.001)
            latency_cap = max(latency_cap, 0.001)
            # Fallback to simple TUE calculation
            num_total = len(all_tool_calls)
            num_correct = sum(
                1 for call in all_tool_calls if call.get("correct", False)
            )
            num_valid_params = sum(
                1 for call in all_tool_calls if call.get("params_valid", False)
            )
            T_correct = num_correct / num_total
            P_params = num_valid_params / num_total
            total_cost = sum(call.get("cost", 0.0) for call in all_tool_calls)
            C_cost = (
                max(0.0, min(1.0, 1.0 - (total_cost / cost_cap)))
                if cost_cap > 0
                else 1.0
            )
            valid_latencies = [
                call.get("latency", 0.0)
                for call in all_tool_calls
                if call.get("latency", 0.0) > 0
            ]
            if valid_latencies and latency_cap > 0:
                avg_latency = sum(valid_latencies) / len(valid_latencies)
                capped_latency = min(avg_latency, latency_cap)
                L_latency = max(0.0, min(1.0, 1.0 - (capped_latency / latency_cap)))
            else:
                L_latency = 1.0
            tue = 0.4 * T_correct + 0.25 * P_params + 0.2 * C_cost + 0.15 * L_latency

            # Extract basic metrics for fallback
            tue_correct_calls = sum(
                1 for call in all_tool_calls if call.get("correct", False)
            )
            tue_valid_param_calls = sum(
                1 for call in all_tool_calls if call.get("params_valid", False)
            )
            tue_tool_correctness = (
                tue_correct_calls / len(all_tool_calls) if all_tool_calls else 0.0
            )
            tue_param_accuracy = (
                tue_valid_param_calls / len(all_tool_calls) if all_tool_calls else 0.0
            )
            tue_by_task = {}
    else:
        tcrr = 0.0
        tue = 0.0
        tcrr_window_size = 3
        tcrr_total_calls = 0
        tcrr_redundant_calls = 0
        tcrr_by_task = {}
        tue_tool_correctness = 0.0
        tue_param_accuracy = 0.0
        tue_correct_calls = 0
        tue_valid_param_calls = 0
        tue_by_task = {}

    # Compute enhanced GSRT metrics
    try:
        from tau2.metrics.gsrt import compute_gsrt_enhanced_metrics

        judge_model = getattr(results.info, "gsrt_judge_llm", None) or "gpt-4o-mini"
        judge_args = getattr(results.info, "gsrt_judge_llm_args", None) or {
            "temperature": 0.0
        }

        gsrt_aggregate = compute_gsrt_enhanced_metrics(
            results.tasks, results.simulations, judge_model, judge_args
        )

        gsrt_median_ack = gsrt_aggregate.aggregate_median_ack
        gsrt_median_tool = gsrt_aggregate.aggregate_median_tool
        gsrt_median_outcome = gsrt_aggregate.aggregate_median_outcome
        gsrt_num_shifts = gsrt_aggregate.total_shifts
        gsrt_recovery_rate = gsrt_aggregate.overall_recovery_rate
        gsrt_transfer_rate = gsrt_aggregate.overall_transfer_rate
        gsrt_never_recovered_rate = gsrt_aggregate.never_recovered_rate
        gsrt_by_task = {
            task_id: {
                "median_ack": task_result.median_gsrt_ack,
                "median_tool": task_result.median_gsrt_tool,
                "median_outcome": task_result.median_gsrt_outcome,
                "total_shifts": task_result.total_shifts,
                "recovery_rate": task_result.recovery_rate,
                "transfer_rate": task_result.transfer_rate,
            }
            for task_id, task_result in gsrt_aggregate.by_task.items()
        }
    except Exception as e:
        logger.warning(f"Enhanced GSRT computation failed: {e}")
        gsrt_median_ack, gsrt_median_tool, gsrt_median_outcome = None, None, None
        gsrt_num_shifts = 0
        gsrt_recovery_rate, gsrt_transfer_rate, gsrt_never_recovered_rate = (
            0.0,
            0.0,
            0.0,
        )
        gsrt_by_task = {}

    return AgentMetrics(
        avg_reward=avg_reward,
        pass_hat_ks=pass_hat_ks,
        avg_agent_cost=avg_agent_cost,
        tsr=tsr,
        tue=tue,
        tcrr=tcrr,
        num_tool_calls=num_tool_calls,
        tsr_communicate_info=tsr_communicate_info,
        tsr_action=tsr_action,
        tsr_nl_assertion=tsr_nl_assertion,
        tsr_weights=tsr_weights,
        tsr_by_task=tsr_by_task,
        tcrr_window_size=tcrr_window_size,
        tcrr_total_calls=tcrr_total_calls,
        tcrr_redundant_calls=tcrr_redundant_calls,
        tcrr_by_task=tcrr_by_task,
        tue_tool_correctness=tue_tool_correctness,
        tue_param_accuracy=tue_param_accuracy,
        tue_correct_calls=tue_correct_calls,
        tue_valid_param_calls=tue_valid_param_calls,
        tue_by_task=tue_by_task,
        gsrt_median_ack=gsrt_median_ack,
        gsrt_median_tool=gsrt_median_tool,
        gsrt_median_outcome=gsrt_median_outcome,
        gsrt_num_shifts=gsrt_num_shifts,
        gsrt_recovery_rate=gsrt_recovery_rate,
        gsrt_transfer_rate=gsrt_transfer_rate,
        gsrt_never_recovered_rate=gsrt_never_recovered_rate,
        gsrt_by_task=gsrt_by_task,
        # Coverage Statistics - basic computation
        tasks_with_communicate_info=0,  # Simple fallback doesn't compute this
        tasks_with_actions=0,
        tasks_with_nl_assertions=0,
        tasks_with_goal_shifts=0,
        # Component Breakdown - basic values
        communicate_info_avg_score=None,
        communicate_info_exact_matches=None,
        total_communicate_info_checks=0,
        action_avg_score=None,
        action_tool_correctness=None,
        action_param_correctness=None,
        total_action_checks=0,
        nl_assertion_avg_score=None,
        total_nl_assertions=0,
        # Partial Scoring Impact
        tasks_benefiting_from_partial=0,
        avg_reward_increase=0.0,
    )


def _compute_basic_metrics(results: Results) -> tuple[float, dict, float]:
    """Compute basic metrics: average reward, pass^k, and average cost."""
    df, max_k = get_metrics_df(results)
    avg_reward = df.reward.mean()

    # Get pass^k metrics separately
    df_pass_hat_k = get_tasks_pass_hat_k(results)
    pass_hat_ks = {}
    for column in df_pass_hat_k.columns:
        if match := re.match(r"pass\^(\d+)", column):
            k = int(match.group(1))
            pass_hat_ks[k] = df_pass_hat_k[column].mean()

    avg_agent_cost = df.agent_cost.mean()
    return avg_reward, pass_hat_ks, avg_agent_cost


def _compute_tsr_metrics(
    results: Results,
) -> tuple[float, float, float, float, dict, dict]:
    """Compute TSR metrics with multi-channel breakdown."""
    try:
        from tau2.metrics.tsr import compute_tsr_enhanced

        tsr_result = compute_tsr_enhanced(results.tasks, results.simulations)
        tsr = tsr_result.overall_tsr
        tsr_communicate_info = tsr_result.communicate_info.success_rate
        tsr_action = tsr_result.action.success_rate
        tsr_nl_assertion = tsr_result.nl_assertion.success_rate
        tsr_weights = tsr_result.weights
        tsr_by_task = tsr_result.by_task

    except Exception as e:
        logger.warning(f"Enhanced TSR computation failed, falling back to simple: {e}")
        # Fallback to simple TSR calculation
        success_count = sum(
            1
            for sim in results.simulations
            if sim.reward_info and safe_float(sim.reward_info.reward) > 0
        )
        tsr = success_count / len(results.simulations) if results.simulations else 0.0
        tsr_communicate_info = 0.0
        tsr_action = 0.0
        tsr_nl_assertion = 0.0
        tsr_weights = {"communicate_info": 0.5, "action": 0.3, "nl_assertion": 0.2}
        tsr_by_task = {}

    return (
        tsr,
        tsr_communicate_info,
        tsr_action,
        tsr_nl_assertion,
        tsr_weights,
        tsr_by_task,
    )


def _compute_tool_calls(results: Results) -> list:
    """Extract all tool calls from simulation results."""
    all_tool_calls = []
    for run in results.simulations:
        tool_calls = extract_tool_calls_from_messages(run.messages)
        all_tool_calls.extend(tool_calls)
    return all_tool_calls


def _compute_tcrr_metrics(
    results: Results, all_tool_calls: list
) -> tuple[float, int, int, int, dict]:
    """Compute TCRR metrics with enhanced window-based detection."""
    try:
        from tau2.metrics.tcrr import compute_tcrr_enhanced, compute_tcrr_by_task

        tcrr_result = compute_tcrr_enhanced(results.simulations)
        tcrr = tcrr_result.redundancy_ratio
        tcrr_window_size = tcrr_result.window_size
        tcrr_total_calls = tcrr_result.total_calls
        tcrr_redundant_calls = tcrr_result.redundant_calls

        # Get task-level breakdown
        tcrr_by_task_results = compute_tcrr_by_task(results.simulations)
        tcrr_by_task = {
            task_id: {
                "redundancy_ratio": result.redundancy_ratio,
                "total_calls": result.total_calls,
                "redundant_calls": result.redundant_calls,
            }
            for task_id, result in tcrr_by_task_results.items()
        }

    except Exception as e:
        logger.warning(f"Enhanced TCRR computation failed, falling back to simple: {e}")
        # Fallback to simple TCRR calculation
        tcrr = 0.0
        tcrr_window_size = 3
        tcrr_total_calls = len(all_tool_calls)
        tcrr_redundant_calls = 0
        tcrr_by_task = {}

    return tcrr, tcrr_window_size, tcrr_total_calls, tcrr_redundant_calls, tcrr_by_task


def _compute_tue_metrics(
    results: Results, all_tool_calls: list
) -> tuple[float, float, float, int, int, dict]:
    """Compute TUE metrics with enhanced tool correctness and parameter accuracy."""
    try:
        from tau2.metrics.tue import (
            compute_tue_enhanced_for_simulations,
            compute_tue_by_task,
        )

        tue_result = compute_tue_enhanced_for_simulations(results.simulations)
        tue = tue_result.overall_tue
        tue_tool_correctness = tue_result.tool_correctness
        tue_param_accuracy = tue_result.param_accuracy
        tue_correct_calls = tue_result.correct_calls
        tue_valid_param_calls = tue_result.valid_param_calls

        # Get task-level breakdown
        tue_by_task_results = compute_tue_by_task(results.simulations)
        tue_by_task = {
            task_id: {
                "overall_tue": result.overall_tue,
                "tool_correctness": result.tool_correctness,
                "param_accuracy": result.param_accuracy,
                "correct_calls": result.correct_calls,
                "valid_param_calls": result.valid_param_calls,
            }
            for task_id, result in tue_by_task_results.items()
        }

    except Exception as e:
        logger.warning(f"Enhanced TUE computation failed, falling back to simple: {e}")
        # Fallback to simple TUE calculation
        tue = 0.0
        tue_tool_correctness = 0.0
        tue_param_accuracy = 0.0
        tue_correct_calls = 0
        tue_valid_param_calls = 0
        tue_by_task = {}

    return (
        tue,
        tue_tool_correctness,
        tue_param_accuracy,
        tue_correct_calls,
        tue_valid_param_calls,
        tue_by_task,
    )


def _compute_gsrt_metrics(
    results: Results,
) -> tuple[
    Optional[float], Optional[float], Optional[float], int, float, float, float, dict
]:
    """Compute GSRT metrics with enhanced multi-variant recovery detection."""
    try:
        from tau2.metrics.gsrt import compute_gsrt_enhanced_metrics

        judge_model = getattr(results.info, "gsrt_judge_llm", None) or "gpt-4o-mini"
        judge_args = getattr(results.info, "gsrt_judge_llm_args", None) or {
            "temperature": 0.0
        }

        gsrt_aggregate = compute_gsrt_enhanced_metrics(
            results.tasks, results.simulations, judge_model, judge_args
        )

        gsrt_median_ack = gsrt_aggregate.aggregate_median_ack
        gsrt_median_tool = gsrt_aggregate.aggregate_median_tool
        gsrt_median_outcome = gsrt_aggregate.aggregate_median_outcome
        gsrt_num_shifts = gsrt_aggregate.total_shifts
        gsrt_recovery_rate = gsrt_aggregate.overall_recovery_rate
        gsrt_transfer_rate = gsrt_aggregate.overall_transfer_rate
        gsrt_never_recovered_rate = gsrt_aggregate.never_recovered_rate
        gsrt_by_task = {
            task_id: {
                "shifts": task_result.total_shifts,
                "median_ack": task_result.median_gsrt_ack,
                "median_tool": task_result.median_gsrt_tool,
                "median_outcome": task_result.median_gsrt_outcome,
                "recovery_rate": task_result.recovery_rate,
                "transfer_rate": task_result.transfer_rate,
            }
            for task_id, task_result in gsrt_aggregate.by_task.items()
        }

    except Exception as e:
        logger.warning(f"Enhanced GSRT computation failed, falling back to simple: {e}")
        gsrt_median_ack, gsrt_median_tool, gsrt_median_outcome = None, None, None
        gsrt_num_shifts = 0
        gsrt_recovery_rate, gsrt_transfer_rate, gsrt_never_recovered_rate = (
            0.0,
            0.0,
            0.0,
        )
        gsrt_by_task = {}

    return (
        gsrt_median_ack,
        gsrt_median_tool,
        gsrt_median_outcome,
        gsrt_num_shifts,
        gsrt_recovery_rate,
        gsrt_transfer_rate,
        gsrt_never_recovered_rate,
        gsrt_by_task,
    )


def _compute_coverage_statistics(
    results: Results, gsrt_num_shifts: int
) -> tuple[int, int, int, int]:
    """Compute coverage statistics for different evaluation criteria."""
    tasks_with_communicate_info = sum(
        1
        for task in results.tasks
        if task.evaluation_criteria and task.evaluation_criteria.communicate_info
    )
    tasks_with_actions = sum(
        1
        for task in results.tasks
        if task.evaluation_criteria
        and (task.evaluation_criteria.actions or task.evaluation_criteria.action_sets)
    )
    tasks_with_nl_assertions = sum(
        1
        for task in results.tasks
        if task.evaluation_criteria and task.evaluation_criteria.nl_assertions
    )
    tasks_with_goal_shifts = gsrt_num_shifts

    return (
        tasks_with_communicate_info,
        tasks_with_actions,
        tasks_with_nl_assertions,
        tasks_with_goal_shifts,
    )


def _compute_component_breakdown(
    results: Results,
    tsr_communicate_info: float,
    tsr_action: float,
    tsr_nl_assertion: float,
    tue_tool_correctness: float,
    tue_param_accuracy: float,
) -> tuple[dict, dict, dict]:
    """Compute component breakdown metrics."""
    # Communicate Info Metrics
    communicate_info_metrics = {
        "avg_score": tsr_communicate_info if tsr_communicate_info > 0 else None,
        "exact_matches": tsr_communicate_info if tsr_communicate_info > 0 else None,
        "total_checks": sum(
            len(task.evaluation_criteria.communicate_info or [])
            for task in results.tasks
            if task.evaluation_criteria and task.evaluation_criteria.communicate_info
        ),
    }

    # Action Metrics
    action_metrics = {
        "avg_score": tsr_action if tsr_action > 0 else None,
        "tool_correctness": tue_tool_correctness if tue_tool_correctness > 0 else None,
        "param_correctness": tue_param_accuracy if tue_param_accuracy > 0 else None,
        "total_checks": sum(
            len(task.evaluation_criteria.actions or [])
            + len(task.evaluation_criteria.action_sets or [])
            for task in results.tasks
            if task.evaluation_criteria
        ),
    }

    # NL Assertion Metrics
    nl_assertion_metrics = {
        "avg_score": tsr_nl_assertion if tsr_nl_assertion > 0 else None,
        "total_assertions": sum(
            len(task.evaluation_criteria.nl_assertions or [])
            for task in results.tasks
            if task.evaluation_criteria and task.evaluation_criteria.nl_assertions
        ),
    }

    return communicate_info_metrics, action_metrics, nl_assertion_metrics


def compute_metrics(results: Results) -> AgentMetrics:
    """
    Compute comprehensive metrics for the agent with modular approach.

    Falls back to simpler computation if enhanced metrics fail.
    """
    try:
        avg_reward, pass_hat_ks, avg_agent_cost = _compute_basic_metrics(results)
        (
            tsr,
            tsr_communicate_info,
            tsr_action,
            tsr_nl_assertion,
            tsr_weights,
            tsr_by_task,
        ) = _compute_tsr_metrics(results)
        all_tool_calls = _compute_tool_calls(results)
        tcrr, tcrr_window_size, tcrr_total_calls, tcrr_redundant_calls, tcrr_by_task = (
            _compute_tcrr_metrics(results, all_tool_calls)
        )
        (
            tue,
            tue_tool_correctness,
            tue_param_accuracy,
            tue_correct_calls,
            tue_valid_param_calls,
            tue_by_task,
        ) = _compute_tue_metrics(results, all_tool_calls)
        (
            gsrt_median_ack,
            gsrt_median_tool,
            gsrt_median_outcome,
            gsrt_num_shifts,
            gsrt_recovery_rate,
            gsrt_transfer_rate,
            gsrt_never_recovered_rate,
            gsrt_by_task,
        ) = _compute_gsrt_metrics(results)
        (
            tasks_with_communicate_info,
            tasks_with_actions,
            tasks_with_nl_assertions,
            tasks_with_goal_shifts,
        ) = _compute_coverage_statistics(results, gsrt_num_shifts)
        communicate_info_metrics, action_metrics, nl_assertion_metrics = (
            _compute_component_breakdown(
                results,
                tsr_communicate_info,
                tsr_action,
                tsr_nl_assertion,
                tue_tool_correctness,
                tue_param_accuracy,
            )
        )
        tasks_benefiting_from_partial = len(
            [
                sim
                for sim in results.simulations
                if sim.reward_info and 0 < sim.reward_info.reward < 1
            ]
        )

        return AgentMetrics(
            # Basic metrics
            avg_reward=avg_reward,
            pass_hat_ks=pass_hat_ks,
            avg_agent_cost=avg_agent_cost,
            num_tool_calls=len(all_tool_calls),
            # Core metrics
            tsr=tsr,
            tue=tue,
            tcrr=tcrr,
            # TSR breakdown
            tsr_communicate_info=tsr_communicate_info,
            tsr_action=tsr_action,
            tsr_nl_assertion=tsr_nl_assertion,
            tsr_weights=tsr_weights,
            tsr_by_task=tsr_by_task,
            # TCRR breakdown
            tcrr_window_size=tcrr_window_size,
            tcrr_total_calls=tcrr_total_calls,
            tcrr_redundant_calls=tcrr_redundant_calls,
            tcrr_by_task=tcrr_by_task,
            # TUE breakdown
            tue_tool_correctness=tue_tool_correctness,
            tue_param_accuracy=tue_param_accuracy,
            tue_correct_calls=tue_correct_calls,
            tue_valid_param_calls=tue_valid_param_calls,
            tue_by_task=tue_by_task,
            # GSRT breakdown
            gsrt_median_ack=gsrt_median_ack,
            gsrt_median_tool=gsrt_median_tool,
            gsrt_median_outcome=gsrt_median_outcome,
            gsrt_num_shifts=gsrt_num_shifts,
            gsrt_recovery_rate=gsrt_recovery_rate,
            gsrt_transfer_rate=gsrt_transfer_rate,
            gsrt_never_recovered_rate=gsrt_never_recovered_rate,
            gsrt_by_task=gsrt_by_task,
            # Coverage statistics
            tasks_with_communicate_info=tasks_with_communicate_info,
            tasks_with_actions=tasks_with_actions,
            tasks_with_nl_assertions=tasks_with_nl_assertions,
            tasks_with_goal_shifts=tasks_with_goal_shifts,
            # Component breakdown
            communicate_info_avg_score=communicate_info_metrics["avg_score"],
            communicate_info_exact_matches=communicate_info_metrics["exact_matches"],
            total_communicate_info_checks=communicate_info_metrics["total_checks"],
            action_avg_score=action_metrics["avg_score"],
            action_tool_correctness=action_metrics["tool_correctness"],
            action_param_correctness=action_metrics["param_correctness"],
            total_action_checks=action_metrics["total_checks"],
            nl_assertion_avg_score=nl_assertion_metrics["avg_score"],
            total_nl_assertions=nl_assertion_metrics["total_assertions"],
            # Partial scoring impact
            tasks_benefiting_from_partial=tasks_benefiting_from_partial,
            avg_reward_increase=0.0,  # Would need old vs new comparison
        )

    except Exception as e:
        logger.warning(
            f"Full metrics computation failed: {e}. Falling back to simple computation."
        )
        return compute_metrics_simple(results)


def get_tasks_pass_hat_k(results: Results) -> pd.DataFrame:
    """
    Compute the pass^k for each k from 1 to the maximum number of trials.
    """
    df, max_k = get_metrics_df(results)
    pass_hat_k_data = []
    for task_id in df.task_id.unique():
        task_df = df[df.task_id == task_id]
        num_trials = len(task_df)
        success_count = task_df.success.sum()

        task_row = {"task_id": task_id}
        for k in range(1, min(max_k + 1, num_trials + 1)):
            task_row[f"pass^{k}"] = pass_hat_k(num_trials, success_count, k)
        pass_hat_k_data.append(task_row)

    df_pass_hat_k = pd.DataFrame(pass_hat_k_data)
    df_pass_hat_k["num_actions"] = df.groupby("task_id").first()["task_num_actions"]
    df_pass_hat_k = df_pass_hat_k.sort_values(by="num_actions")
    return df_pass_hat_k


def prepare_dfs(results: Results) -> tuple[pd.DataFrame, pd.DataFrame]:
    df, max_k = get_metrics_df(results)
    df_pass_hat_k = get_tasks_pass_hat_k(results)
    df_pass_hat_k["num_actions"] = df.groupby("task_id").first()["task_num_actions"]
    df_pass_hat_k = df_pass_hat_k.sort_values(by="num_actions")
    return df, df_pass_hat_k


def display_metrics(metrics: AgentMetrics) -> None:
    print(f"🏆 Average reward: {metrics.avg_reward}")
    print("📈 Pass^k")
    for k, pass_hat_k in metrics.pass_hat_ks.items():
        print(f"  k={k}: {pass_hat_k}")
    print(f"💰 Average agent cost: {metrics.avg_agent_cost}")
    print()
    print("🎯 AgentChangeBench Metrics:")
    print(f"  📊 TSR (Task Success Rate): {metrics.tsr:.2%}")
    print(
        f"    💬 Communicate Info: {metrics.tsr_communicate_info:.2%} (weight: {metrics.tsr_weights.get('communicate_info', 0.5):.0%})"
    )
    print(
        f"    ⚙️  Actions: {metrics.tsr_action:.2%} (weight: {metrics.tsr_weights.get('action', 0.3):.0%})"
    )
    print(
        f"    📝 NL Assertions: {metrics.tsr_nl_assertion:.2%} (weight: {metrics.tsr_weights.get('nl_assertion', 0.2):.0%})"
    )
    if metrics.tsr_by_task:
        print(f"    📋 Task-Level Breakdown: {len(metrics.tsr_by_task)} tasks")
    print(f"  ⚙️  TUE (Tool Usage Efficiency): {metrics.tue:.2%}")
    print(
        f"    🎯 Tool Correctness: {metrics.tue_tool_correctness:.2%} ({metrics.tue_correct_calls}/{metrics.num_tool_calls})"
    )
    print(
        f"    📝 Parameter Accuracy: {metrics.tue_param_accuracy:.2%} ({metrics.tue_valid_param_calls}/{metrics.num_tool_calls})"
    )
    if metrics.tue_by_task:
        print(f"    📋 Task-Level Breakdown: {len(metrics.tue_by_task)} tasks")
    print(f"  🔄 TCRR (Tool-Call Redundancy Ratio): {metrics.tcrr:.2%}")
    print(
        f"    📊 Redundant Calls: {metrics.tcrr_redundant_calls}/{metrics.tcrr_total_calls}"
    )
    print(f"    🪟 Window Size: {metrics.tcrr_window_size} assistant turns")
    if metrics.tcrr_by_task:
        print(f"    📋 Task-Level Breakdown: {len(metrics.tcrr_by_task)} tasks")
    print(f"  🛠️  Total Tool Calls: {metrics.num_tool_calls}")
    print(f"  🔀 GSRT (Goal Shift Recovery Time):")
    if metrics.gsrt_num_shifts > 0:
        print(f"    📊 Goal Shifts Detected: {metrics.gsrt_num_shifts}")
        print(f"    📈 Recovery Rate: {metrics.gsrt_recovery_rate:.1%}")
        print(f"    🔄 Transfer Rate: {metrics.gsrt_transfer_rate:.1%}")
        print(f"    ❌ Never Recovered Rate: {metrics.gsrt_never_recovered_rate:.1%}")
        print(f"    Recovery Times by Variant:")
        if metrics.gsrt_median_ack is not None:
            print(
                f"      🎯 Acknowledgment: {metrics.gsrt_median_ack:.1f} turns (median)"
            )
        if metrics.gsrt_median_tool is not None:
            print(f"      🛠️  Tool Usage: {metrics.gsrt_median_tool:.1f} turns (median)")
        if metrics.gsrt_median_outcome is not None:
            print(
                f"      ✅ Outcome Success: {metrics.gsrt_median_outcome:.1f} turns (median)"
            )
        if metrics.gsrt_by_task:
            print(
                f"    📋 Task-Level Breakdown: {len(metrics.gsrt_by_task)} tasks with shifts"
            )
    else:
        print(f"    ❌ No goal shifts detected")


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True)
    args = parser.parse_args()
    results = Results.load(Path(args.results))
    metrics = compute_metrics(results)
    display_metrics(metrics)
