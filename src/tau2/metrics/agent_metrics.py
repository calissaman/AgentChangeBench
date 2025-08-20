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
                'success': not msg.error,
                'error': msg.error,
                'content': msg.content
            }
    
    for msg in messages:
        # Look for tool calls in AssistantMessage and UserMessage objects
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tool_call in msg.tool_calls:
                # Determine correctness based on whether the tool executed without errors
                correct = True
                if hasattr(tool_call, 'id') and tool_call.id in tool_results:
                    correct = tool_results[tool_call.id]['success']
                
                # Determine parameter validity
                # For now, assume parameters are valid if:
                # 1. The tool call was attempted (it exists)
                # 2. The tool didn't fail due to parameter issues
                params_valid = True
                if hasattr(tool_call, 'id') and tool_call.id in tool_results:
                    result = tool_results[tool_call.id]
                    if result['error'] and result.get('content'):
                        # Check if error message indicates parameter issues
                        error_content = str(result['content']).lower()
                        if any(keyword in error_content for keyword in [
                            'invalid parameter', 'missing parameter', 'parameter error',
                            'bad parameter', 'invalid argument', 'missing argument'
                        ]):
                            params_valid = False
                
                # Calculate latency if timestamps are available
                latency = 0.0
                if hasattr(msg, 'timestamp') and hasattr(tool_call, 'id') and tool_call.id in tool_results:
                    # Could implement timestamp-based latency calculation here
                    pass
                
                tool_call_metric = {
                    'name': tool_call.name,
                    'params': tool_call.arguments,  # ToolCall uses 'arguments' not 'params'
                    'cost': safe_float(msg.cost) if hasattr(msg, 'cost') else 0.0,
                    'latency': latency,
                    'correct': correct,
                    'params_valid': params_valid
                }
                tool_calls.append(tool_call_metric)
    
    return tool_calls


def compute_tsr(sim_runs: list[SimulationRun]) -> float:
    """Calculate Task Success Rate from simulation runs."""
    if not sim_runs:
        return 0.0
    
    success_count = 0
    for run in sim_runs:
        if run.reward_info and safe_float(run.reward_info.reward) > 0:
            success_count += 1
    
    return success_count / len(sim_runs)


def compute_tcrr(tool_calls: list[dict]) -> float:
    """Compute Tool-Call Redundancy Ratio."""
    if not tool_calls:
        return 0.0
    
    seen_calls = set()
    duplicate_count = 0
    
    for call in tool_calls:
        try:
            params_norm = normalized_params(call['params'])
            identity = (call['name'], params_norm)
            if identity in seen_calls:
                duplicate_count += 1
            else:
                seen_calls.add(identity)
        except Exception as e:
            logger.warning(f"Error normalizing params for TCRR: {e}")
            # Fallback: string representation
            identity = (call['name'], str(call['params']))
            if identity in seen_calls:
                duplicate_count += 1
            else:
                seen_calls.add(identity)
    
    return duplicate_count / len(tool_calls)


def compute_tue(tool_calls: list[dict], cost_cap: float, latency_cap: float) -> float:
    """
    Compute Tool Usage Efficiency according to AgentChangeBench specification.
    
    TUE = wT·Tcorrect + wP·Pparams + wC·Ccost + wL·Llatency
    Weights: wT=0.4, wP=0.25, wC=0.2, wL=0.15
    
    Args:
        tool_calls: List of tool call dictionaries with keys: correct, params_valid, cost, latency
        cost_cap: 95th percentile cost cap for normalization
        latency_cap: 95th percentile latency cap for normalization
    """
    if not tool_calls:
        return 0.0
    
    num_total = len(tool_calls)
    
    # T_correct: Fraction of correct tool calls
    num_correct = sum(1 for call in tool_calls if call.get('correct', False))
    T_correct = num_correct / num_total
    
    # P_params: Fraction of tool calls with valid parameters
    num_valid_params = sum(1 for call in tool_calls if call.get('params_valid', False))
    P_params = num_valid_params / num_total
    
    # C_cost: Cost efficiency = 1 - actual_cost/cost_cap
    total_cost = sum(call.get('cost', 0.0) for call in tool_calls)
    if cost_cap > 0:
        C_cost = max(0.0, min(1.0, 1.0 - (total_cost / cost_cap)))
    else:
        C_cost = 1.0  # If no cost cap, assume perfect efficiency
    
    # L_latency: Latency efficiency = 1 - min(avg_latency, latency_cap)/latency_cap
    valid_latencies = [call.get('latency', 0.0) for call in tool_calls if call.get('latency', 0.0) > 0]
    if valid_latencies and latency_cap > 0:
        avg_latency = sum(valid_latencies) / len(valid_latencies)
        # Use min(avg_latency, latency_cap) as per spec
        capped_latency = min(avg_latency, latency_cap)
        L_latency = max(0.0, min(1.0, 1.0 - (capped_latency / latency_cap)))
    else:
        L_latency = 1.0  # If no latency data, assume perfect efficiency
    
    # Weighted TUE formula with exact AgentChangeBench weights
    tue = 0.4 * T_correct + 0.25 * P_params + 0.2 * C_cost + 0.15 * L_latency
    
    return tue


def extract_domain_from_task_id(task_id: str) -> str:
    """Extract domain name from task_id."""
    parts = task_id.split('_')
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
    # GSRT metrics
    gsrt_median: Optional[float] = None
    gsrt_worst_case: Optional[int] = None  
    gsrt_num_shifts: int = 0

    def as_dict(self) -> dict:
        data = {
            "avg_reward": self.avg_reward,
            "avg_agent_cost": self.avg_agent_cost,
            "tsr": self.tsr,
            "tue": self.tue,
            "tcrr": self.tcrr,
            "num_tool_calls": self.num_tool_calls,
            "gsrt_median": self.gsrt_median,
            "gsrt_worst_case": self.gsrt_worst_case,
            "gsrt_num_shifts": self.gsrt_num_shifts,
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
    rewards = [sim.reward_info.reward if sim.reward_info else 0.0 for sim in results.simulations]
    avg_reward = np.mean(rewards) if rewards else 0.0
    
    # Simple cost calculation
    agent_costs = [sim.agent_cost if sim.agent_cost else 0.0 for sim in results.simulations]
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
            task_groups[task_id].append(is_successful(sim.reward_info.reward if sim.reward_info else 0.0))
        
        # Compute pass^k for k=1 (basic success rate)
        if task_groups:
            task_success_rates = []
            for task_id, successes in task_groups.items():
                success_rate = sum(successes) / len(successes)
                task_success_rates.append(success_rate)
            pass_hat_ks[1] = np.mean(task_success_rates)
    
    # Compute AgentChangeBench metrics
    # Calculate TSR
    tsr = compute_tsr(results.simulations)
    
    # Extract all tool calls
    all_tool_calls = []
    for run in results.simulations:
        tool_calls = extract_tool_calls_from_messages(run.messages)
        all_tool_calls.extend(tool_calls)
    
    num_tool_calls = len(all_tool_calls)
    
    if all_tool_calls:
        # Calculate TCRR
        tcrr = compute_tcrr(all_tool_calls)
        
        # Calculate TUE (need cost and latency caps)
        all_costs = [call['cost'] for call in all_tool_calls if call['cost'] > 0]
        all_latencies = [call['latency'] for call in all_tool_calls if call['latency'] > 0]
        
        cost_cap = np.percentile(all_costs, 95) if all_costs else 1.0
        latency_cap = np.percentile(all_latencies, 95) if all_latencies else 1.0
        cost_cap = max(cost_cap, 0.001)  # prevent division by zero
        latency_cap = max(latency_cap, 0.001)
        
        tue = compute_tue(all_tool_calls, cost_cap, latency_cap)
    else:
        tcrr = 0.0
        tue = 0.0
    
    # Compute GSRT using v2 judge unconditionally
    try:
        from tau2.metrics.gsrt_v2 import detect_gsrt_v2
        gsrt_counts = 0
        recovery_turns: list[int] = []
        judge_model = getattr(results.info, "gsrt_judge_llm", None) or "gpt-5"
        judge_args = getattr(results.info, "gsrt_judge_llm_args", None) or {"temperature": 0.0}
        for sim in results.simulations:
            task = None
            for t in results.tasks:
                if t.id == sim.task_id:
                    task = t
                    break
            if not task:
                continue
            # Try to reuse cached judge output if present
            res = None
            if sim.reward_info and sim.reward_info.info and isinstance(sim.reward_info.info, dict):
                res = sim.reward_info.info.get("gsrt_v2")
            if not res:
                res = detect_gsrt_v2(task, sim, model=judge_model, llm_args=judge_args)
                # persist back into reward_info.info
                try:
                    if sim.reward_info is not None:
                        if sim.reward_info.info is None:
                            sim.reward_info.info = {}
                        if isinstance(sim.reward_info.info, dict):
                            sim.reward_info.info["gsrt_v2"] = res
                except Exception:
                    pass
            shifts = res.get("user_goal_shifts", [])
            gsrt_counts += len(shifts)
            for s in shifts:
                agent_turn = s.get("agent_turn")
                if isinstance(agent_turn, int):
                    dt = max(0, agent_turn - int(s.get("turn", 0)))
                    recovery_turns.append(dt)
        gsrt_median = (statistics.median(recovery_turns) if recovery_turns else None)
        gsrt_worst_case = (max(recovery_turns) if recovery_turns else None)
        gsrt_num_shifts = gsrt_counts
    except Exception as e:
        logger.warning(f"GSRT computation failed: {e}")
        gsrt_median, gsrt_worst_case, gsrt_num_shifts = None, None, 0
    
    return AgentMetrics(
        avg_reward=avg_reward,
        pass_hat_ks=pass_hat_ks,
        avg_agent_cost=avg_agent_cost,
        tsr=tsr,
        tue=tue,
        tcrr=tcrr,
        num_tool_calls=num_tool_calls,
        gsrt_median=gsrt_median,
        gsrt_worst_case=gsrt_worst_case,
        gsrt_num_shifts=gsrt_num_shifts,
    )


def compute_metrics(results: Results) -> AgentMetrics:
    """
    Compute metrics for the agent.
    - average reward
    - pass^k
    - TSR (Task Success Rate)
    - TUE (Tool Usage Efficiency)
    - TCRR (Tool-Call Redundancy Ratio)
    - GSRT (Goal Shift Recovery Time via LLM-judged v2)
    
    Falls back to simpler computation if task evaluation criteria are missing.
    """
    try:
        # Try the full computation first
        df, df_pass_hat_k = get_metrics_df(results)
        avg_reward = df.reward.mean()
        pass_hat_ks = {}
        for column in df_pass_hat_k.columns:
            if match := re.match(r"pass\^(\d+)", column):
                k = int(match.group(1))
                pass_hat_ks[k] = df_pass_hat_k[column].mean()
        avg_agent_cost = df.agent_cost.mean()
        
        # Compute AgentChangeBench metrics
        tsr = compute_tsr(results.simulations)
        
        # Extract all tool calls
        all_tool_calls = []
        for run in results.simulations:
            tool_calls = extract_tool_calls_from_messages(run.messages)
            all_tool_calls.extend(tool_calls)
        
        num_tool_calls = len(all_tool_calls)
        
        if all_tool_calls:
            tcrr = compute_tcrr(all_tool_calls)
            
            all_costs = [call['cost'] for call in all_tool_calls if call['cost'] > 0]
            all_latencies = [call['latency'] for call in all_tool_calls if call['latency'] > 0]
            
            cost_cap = np.percentile(all_costs, 95) if all_costs else 1.0
            latency_cap = np.percentile(all_latencies, 95) if all_latencies else 1.0
            cost_cap = max(cost_cap, 0.001)
            latency_cap = max(latency_cap, 0.001)
            
            tue = compute_tue(all_tool_calls, cost_cap, latency_cap)
        else:
            tcrr = 0.0
            tue = 0.0
        
        # Compute GSRT using v2 judge unconditionally
        try:
            from tau2.metrics.gsrt_v2 import detect_gsrt_v2
            gsrt_counts = 0
            recovery_turns: list[int] = []
            judge_model = getattr(results.info, "gsrt_judge_llm", None) or "gpt-5"
            judge_args = getattr(results.info, "gsrt_judge_llm_args", None) or {"temperature": 0.0}
            for sim in results.simulations:
                task = next((t for t in results.tasks if t.id == sim.task_id), None)
                if not task:
                    continue
                # Try to reuse cached judge output if present
                res = None
                if sim.reward_info and sim.reward_info.info and isinstance(sim.reward_info.info, dict):
                    res = sim.reward_info.info.get("gsrt_v2")
                if not res:
                    res = detect_gsrt_v2(task, sim, model=judge_model, llm_args=judge_args)
                    # persist back into reward_info.info
                    try:
                        if sim.reward_info is not None:
                            if sim.reward_info.info is None:
                                sim.reward_info.info = {}
                            if isinstance(sim.reward_info.info, dict):
                                sim.reward_info.info["gsrt_v2"] = res
                    except Exception:
                        pass
                shifts = res.get("user_goal_shifts", [])
                gsrt_counts += len(shifts)
                for s in shifts:
                    agent_turn = s.get("agent_turn")
                    if isinstance(agent_turn, int):
                        dt = max(0, agent_turn - int(s.get("turn", 0)))
                        recovery_turns.append(dt)
            gsrt_median = (statistics.median(recovery_turns) if recovery_turns else None)
            gsrt_worst_case = (max(recovery_turns) if recovery_turns else None)
            gsrt_num_shifts = gsrt_counts
        except Exception as e:
            logger.warning(f"GSRT computation failed: {e}")
            gsrt_median, gsrt_worst_case, gsrt_num_shifts = None, None, 0
        
        return AgentMetrics(
            avg_reward=avg_reward,
            pass_hat_ks=pass_hat_ks,
            avg_agent_cost=avg_agent_cost,
            tsr=tsr,
            tue=tue,
            tcrr=tcrr,
            num_tool_calls=num_tool_calls,
            gsrt_median=gsrt_median,
            gsrt_worst_case=gsrt_worst_case,
            gsrt_num_shifts=gsrt_num_shifts,
        )
    except Exception as e:
        logger.warning(f"Full metrics computation failed: {e}. Falling back to simple computation.")
        return compute_metrics_simple(results)


def get_tasks_pass_hat_k(results: Results) -> pd.DataFrame:
    """
    Compute the pass^k for each k from 1 to the maximum number of trials.
    """
    df, max_k = get_metrics_df(results)
    df_pass_hat_k = get_tasks_pass_hat_k(results)
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
    print(f"  ⚙️  TUE (Tool Usage Efficiency): {metrics.tue:.2%}")
    print(f"  🔄 TCRR (Tool-Call Redundancy Ratio): {metrics.tcrr:.2%}")
    print(f"  🛠️  Total Tool Calls: {metrics.num_tool_calls}")
    print(f"  🔀 GSRT (Goal Shift Recovery Time):")
    if metrics.gsrt_num_shifts > 0:
        print(f"    📊 Goal Shifts Detected: {metrics.gsrt_num_shifts}")
        if metrics.gsrt_median is not None:
            print(f"    📈 Median Recovery Time: {metrics.gsrt_median:.1f} turns")
        if metrics.gsrt_worst_case is not None:
            print(f"    📉 Worst Case Recovery: {metrics.gsrt_worst_case} turns")
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
