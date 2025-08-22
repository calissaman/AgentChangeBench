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

    tsr: float  # Task Success Rate
    tue: float  # Tool Usage Efficiency
    tcrr: float  # Tool-Call Redundancy Ratio
    num_tool_calls: int

    tsr_communicate_info: float = 0.0
    tsr_action: float = 0.0
    tsr_nl_assertion: float = 0.0
    tsr_weights: dict = {}  # Weights used in TSR calculation
    tsr_by_task: dict = {}  # Task-level TSR breakdown

    tue_tool_correctness: float = 0.0
    tue_param_accuracy: float = 0.0
    tue_correct_calls: int = 0
    tue_valid_param_calls: int = 0
    tue_by_task: dict = {}  # Task-level TUE breakdown

    tcrr_window_size: int = 3
    tcrr_total_calls: int = 0
    tcrr_redundant_calls: int = 0
    tcrr_by_task: dict = {}  # Task-level TCRR breakdown

    gsrt_median_ack: Optional[float] = None
    gsrt_median_tool: Optional[float] = None
    gsrt_median_outcome: Optional[float] = None
    gsrt_num_shifts: int = 0
    gsrt_recovery_rate: float = 0.0
    gsrt_transfer_rate: float = 0.0
    gsrt_never_recovered_rate: float = 0.0
    gsrt_by_task: dict = {}  # Task-level breakdown

    tasks_with_communicate_info: int = 0
    tasks_with_actions: int = 0
    tasks_with_nl_assertions: int = 0
    tasks_with_goal_shifts: int = 0

    communicate_info_avg_score: Optional[float] = None
    communicate_info_exact_matches: Optional[float] = None
    total_communicate_info_checks: int = 0

    action_avg_score: Optional[float] = None
    action_tool_correctness: Optional[float] = None
    action_param_correctness: Optional[float] = None
    total_action_checks: int = 0

    nl_assertion_avg_score: Optional[float] = None
    total_nl_assertions: int = 0

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


def _compute_basic_metrics(results: Results) -> tuple[float, dict, float]:
    """Compute basic metrics: average reward, pass^k, and average cost."""
    try:
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
    except Exception as e:
        logger.error(f"Basic metrics computation failed: {e}")
        return -1.0, {1: -1.0}, -1.0


def _compute_tsr_metrics(
    results: Results,
) -> tuple[float, float, float, float, dict, dict]:
    """Compute TSR metrics using cached results from simulation runs."""
    try:
        # Check if we can use cached TSR results from simulation runs
        cached_tsr_results = []
        has_cached_data = True

        for sim in results.simulations:
            if (
                sim.reward_info
                and sim.reward_info.info
                and isinstance(sim.reward_info.info, dict)
                and "tsr_details" in sim.reward_info.info
            ):
                cached_tsr_results.append(sim.reward_info.info["tsr_details"])
            else:
                has_cached_data = False
                break

        if has_cached_data and cached_tsr_results:
            total_tsr = sum(
                result.get("overall_tsr", 0) for result in cached_tsr_results
            )
            avg_tsr = total_tsr / len(cached_tsr_results)

            comm_scores = [
                r.get("communicate_info")
                for r in cached_tsr_results
                if r.get("communicate_info") is not None
            ]
            action_scores = [
                r.get("action")
                for r in cached_tsr_results
                if r.get("action") is not None
            ]
            nl_scores = [
                r.get("nl_assertion")
                for r in cached_tsr_results
                if r.get("nl_assertion") is not None
            ]

            tsr_communicate_info = (
                sum(comm_scores) / len(comm_scores) if comm_scores else 0.0
            )
            tsr_action = (
                sum(action_scores) / len(action_scores) if action_scores else 0.0
            )
            tsr_nl_assertion = sum(nl_scores) / len(nl_scores) if nl_scores else 0.0

            tsr_weights = cached_tsr_results[0].get(
                "weights_used",
                {"communicate_info": 0.5, "action": 0.3, "nl_assertion": 0.2},
            )

            tsr_by_task = {}
            for sim, tsr_result in zip(results.simulations, cached_tsr_results):
                tsr_by_task[sim.task_id] = {
                    "overall_tsr": tsr_result.get("overall_tsr", 0),
                    "communicate_info": tsr_result.get("communicate_info"),
                    "action": tsr_result.get("action"),
                    "nl_assertion": tsr_result.get("nl_assertion"),
                }

            logger.info(
                "Using cached TSR results from simulation runs - no recomputation needed!"
            )
            return (
                avg_tsr,
                tsr_communicate_info,
                tsr_action,
                tsr_nl_assertion,
                tsr_weights,
                tsr_by_task,
            )

        else:
            # This should not happen in normal operation
            logger.error(
                "No cached TSR data found in simulation runs! TSR should be computed during tau2 run, not during metrics aggregation."
            )
            raise ValueError(
                "Missing cached TSR data - ensure simulations were run with TSR computation enabled"
            )

    except Exception as e:
        logger.error(f"TSR metrics computation failed: {e}")
        return -1.0, -1.0, -1.0, -1.0, {}, {}


def _compute_tool_calls(results: Results) -> list:
    """Extract all tool calls from simulation results."""
    try:
        all_tool_calls = []
        for run in results.simulations:
            tool_calls = extract_tool_calls_from_messages(run.messages)
            all_tool_calls.extend(tool_calls)
        return all_tool_calls
    except Exception as e:
        logger.error(f"Tool calls extraction failed: {e}")
        return []


def _compute_tcrr_metrics(
    results: Results, all_tool_calls: list
) -> tuple[float, int, int, int, dict]:
    """Compute TCRR metrics with enhanced window-based detection."""
    try:
        from tau2.metrics.tcrr import compute_tcrr

        tcrr_result, tcrr_by_task_results = compute_tcrr(results.simulations)
        tcrr = tcrr_result.redundancy_ratio
        tcrr_window_size = tcrr_result.window_size
        tcrr_total_calls = tcrr_result.total_calls
        tcrr_redundant_calls = tcrr_result.redundant_calls

        tcrr_by_task = {
            task_id: {
                "redundancy_ratio": result.redundancy_ratio,
                "total_calls": result.total_calls,
                "redundant_calls": result.redundant_calls,
            }
            for task_id, result in tcrr_by_task_results.items()
        }

    except Exception as e:
        logger.error(f"TCRR metrics computation failed: {e}")
        return -1.0, -1, -1, -1, {}

    return tcrr, tcrr_window_size, tcrr_total_calls, tcrr_redundant_calls, tcrr_by_task


def _compute_tue_metrics(
    results: Results, all_tool_calls: list
) -> tuple[float, float, float, int, int, dict]:
    """Compute TUE metrics with enhanced tool correctness and parameter accuracy."""
    try:
        from tau2.metrics.tue import compute_tue

        tue_result, tue_by_task_results = compute_tue(results.simulations)
        tue = tue_result.overall_tue
        tue_tool_correctness = tue_result.tool_correctness
        tue_param_accuracy = tue_result.param_accuracy
        tue_correct_calls = tue_result.correct_calls
        tue_valid_param_calls = tue_result.valid_param_calls

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
        logger.error(f"TUE metrics computation failed: {e}")
        return -1.0, -1.0, -1.0, -1, -1, {}

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
    """Compute GSRT metrics using cached results from simulation runs."""
    try:
        cached_gsrt_results = []
        has_cached_data = True

        for sim in results.simulations:
            if (
                sim.reward_info
                and sim.reward_info.info
                and isinstance(sim.reward_info.info, dict)
                and (
                    "gsrt_v2" in sim.reward_info.info
                    or "gsrt_enhanced" in sim.reward_info.info
                )
            ):
                gsrt_data = sim.reward_info.info.get(
                    "gsrt_v2"
                ) or sim.reward_info.info.get("gsrt_enhanced")
                cached_gsrt_results.append((sim.task_id, gsrt_data))
            else:
                has_cached_data = False
                break

        if has_cached_data and cached_gsrt_results:
            total_shifts = 0
            all_ack_times = []
            all_tool_times = []
            all_outcome_times = []
            recovered_count = 0
            transfer_count = 0
            never_recovered_count = 0
            gsrt_by_task = {}

            for task_id, gsrt_data in cached_gsrt_results:
                if not gsrt_data or not isinstance(gsrt_data, dict):
                    continue

                goal_shifts = gsrt_data.get("user_goal_shifts", [])
                task_shifts = len(goal_shifts)
                total_shifts += task_shifts

                task_ack_times = []
                task_tool_times = []
                task_outcome_times = []
                task_recovered = 0
                task_transfers = 0

                for shift in goal_shifts:
                    if isinstance(shift, dict):
                        if "agent_responses" in shift:
                            responses = shift["agent_responses"]
                            if responses:
                                ack_time = min(
                                    r.get("turn", float("inf"))
                                    for r in responses
                                    if isinstance(r, dict)
                                )
                                if ack_time != float("inf"):
                                    task_ack_times.append(ack_time)

                        if shift.get("transferred_to_human"):
                            task_transfers += 1

                        ack_turn = shift.get("acknowledgment_turn")
                        shift_turn = shift.get("turn", 0)
                        transferred = shift.get("transferred_to_human", False)
                        
                        if ack_turn is not None and ack_turn > shift_turn and not transferred:
                            task_recovered += 1

                all_ack_times.extend(task_ack_times)
                all_tool_times.extend(task_tool_times)
                all_outcome_times.extend(task_outcome_times)

                recovered_count += task_recovered
                transfer_count += task_transfers

                if task_shifts > 0 and task_recovered == 0:
                    never_recovered_count += 1

                gsrt_by_task[task_id] = {
                    "shifts": task_shifts,
                    "median_ack": statistics.median(task_ack_times)
                    if task_ack_times
                    else None,
                    "median_tool": statistics.median(task_tool_times)
                    if task_tool_times
                    else None,
                    "median_outcome": statistics.median(task_outcome_times)
                    if task_outcome_times
                    else None,
                    "recovery_rate": task_recovered / task_shifts
                    if task_shifts > 0
                    else 0.0,
                    "transfer_rate": task_transfers / task_shifts
                    if task_shifts > 0
                    else 0.0,
                }

            gsrt_median_ack = (
                statistics.median(all_ack_times) if all_ack_times else None
            )
            gsrt_median_tool = (
                statistics.median(all_tool_times) if all_tool_times else None
            )
            gsrt_median_outcome = (
                statistics.median(all_outcome_times) if all_outcome_times else None
            )
            gsrt_recovery_rate = (
                recovered_count / total_shifts if total_shifts > 0 else 0.0
            )
            gsrt_transfer_rate = (
                transfer_count / total_shifts if total_shifts > 0 else 0.0
            )
            gsrt_never_recovered_rate = (
                never_recovered_count / len(cached_gsrt_results)
                if cached_gsrt_results
                else 0.0
            )

            logger.info(
                "Using cached GSRT results from simulation runs - no LLM judge recomputation needed!"
            )
            return (
                gsrt_median_ack,
                gsrt_median_tool,
                gsrt_median_outcome,
                total_shifts,
                gsrt_recovery_rate,
                gsrt_transfer_rate,
                gsrt_never_recovered_rate,
                gsrt_by_task,
            )

        else:
            # No cached GSRT data found - this should not happen in normal operation
            logger.error(
                "No cached GSRT data found in simulation runs! GSRT should be computed during tau2 run, not during metrics aggregation."
            )
            raise ValueError(
                "Missing cached GSRT data - ensure simulations were run with GSRT computation enabled"
            )

    except Exception as e:
        logger.error(f"GSRT metrics computation failed: {e}")
        return None, None, None, -1, -1.0, -1.0, -1.0, {}


def _compute_coverage_statistics(
    results: Results, gsrt_num_shifts: int
) -> tuple[int, int, int, int]:
    """Compute coverage statistics for different evaluation criteria."""
    try:
        tasks_with_communicate_info = sum(
            1
            for task in results.tasks
            if task.evaluation_criteria and task.evaluation_criteria.communicate_info
        )
        tasks_with_actions = sum(
            1
            for task in results.tasks
            if task.evaluation_criteria
            and (
                task.evaluation_criteria.actions or task.evaluation_criteria.action_sets
            )
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
    except Exception as e:
        logger.error(f"Coverage statistics computation failed: {e}")
        return -1, -1, -1, -1


def _compute_component_breakdown(
    results: Results,
    tsr_communicate_info: float,
    tsr_action: float,
    tsr_nl_assertion: float,
    tue_tool_correctness: float,
    tue_param_accuracy: float,
) -> tuple[dict, dict, dict]:
    """Compute component breakdown metrics."""
    try:
        communicate_info_metrics = {
            "avg_score": tsr_communicate_info if tsr_communicate_info > 0 else None,
            "exact_matches": tsr_communicate_info if tsr_communicate_info > 0 else None,
            "total_checks": sum(
                len(task.evaluation_criteria.communicate_info or [])
                for task in results.tasks
                if task.evaluation_criteria
                and task.evaluation_criteria.communicate_info
            ),
        }

        action_metrics = {
            "avg_score": tsr_action if tsr_action > 0 else None,
            "tool_correctness": tue_tool_correctness
            if tue_tool_correctness > 0
            else None,
            "param_correctness": tue_param_accuracy if tue_param_accuracy > 0 else None,
            "total_checks": sum(
                len(task.evaluation_criteria.actions or [])
                + len(task.evaluation_criteria.action_sets or [])
                for task in results.tasks
                if task.evaluation_criteria
            ),
        }

        nl_assertion_metrics = {
            "avg_score": tsr_nl_assertion if tsr_nl_assertion > 0 else None,
            "total_assertions": sum(
                len(task.evaluation_criteria.nl_assertions or [])
                for task in results.tasks
                if task.evaluation_criteria and task.evaluation_criteria.nl_assertions
            ),
        }

        return communicate_info_metrics, action_metrics, nl_assertion_metrics
    except Exception as e:
        logger.error(f"Component breakdown computation failed: {e}")
        return (
            {"avg_score": -1, "exact_matches": -1, "total_checks": -1},
            {
                "avg_score": -1,
                "tool_correctness": -1,
                "param_correctness": -1,
                "total_checks": -1,
            },
            {"avg_score": -1, "total_assertions": -1},
        )


def compute_metrics(results: Results) -> AgentMetrics:
    """
    Compute comprehensive metrics for the agent with modular approach.

    Each helper function handles its own failures and returns -1 on error.
    """
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
        avg_reward=avg_reward,
        pass_hat_ks=pass_hat_ks,
        avg_agent_cost=avg_agent_cost,
        num_tool_calls=len(all_tool_calls),
        tsr=tsr,
        tue=tue,
        tcrr=tcrr,
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
        tasks_with_communicate_info=tasks_with_communicate_info,
        tasks_with_actions=tasks_with_actions,
        tasks_with_nl_assertions=tasks_with_nl_assertions,
        tasks_with_goal_shifts=tasks_with_goal_shifts,
        communicate_info_avg_score=communicate_info_metrics["avg_score"],
        communicate_info_exact_matches=communicate_info_metrics["exact_matches"],
        total_communicate_info_checks=communicate_info_metrics["total_checks"],
        action_avg_score=action_metrics["avg_score"],
        action_tool_correctness=action_metrics["tool_correctness"],
        action_param_correctness=action_metrics["param_correctness"],
        total_action_checks=action_metrics["total_checks"],
        nl_assertion_avg_score=nl_assertion_metrics["avg_score"],
        total_nl_assertions=nl_assertion_metrics["total_assertions"],
        tasks_benefiting_from_partial=tasks_benefiting_from_partial,
        avg_reward_increase=0.0,  # Would need old vs new comparison
    )


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
