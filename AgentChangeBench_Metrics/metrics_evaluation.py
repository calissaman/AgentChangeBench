import os
import json
import logging
import numpy as np


# Configure logging to save all logs to a file for debugging and audit
logging.basicConfig(
    filename='metrics_evaluation.log',
    filemode='w',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def load_data(file_path):
    """
    Load JSON data from a file path.
    Handles file not found and JSON decoding errors gracefully,
    and logs errors for debugging.
    Returns an empty dict on failure.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load JSON file {file_path}: {e}")
        return {}


def validate_tool_params(tool_call):
    """
    Validate the structure of a tool call dict.
    Ensures 'name' key is present and
    'params' or 'arguments' key is a dict.
    Logs warnings on invalid entries.
    Returns True if valid, False otherwise.
    """
    if not isinstance(tool_call, dict):
        return False
    if 'name' not in tool_call:
        logging.warning(f"Tool call missing 'name': {tool_call}")
        return False
    params_key = None
    if 'params' in tool_call:
        params_key = 'params'
    elif 'arguments' in tool_call:
        params_key = 'arguments'
    if params_key is None or not isinstance(tool_call.get(params_key), dict):
        logging.warning(f"Tool call with invalid or missing params/arguments: {tool_call}")
        return False
    return True


def extract_tool_calls(data):
    """
    Extract all valid tool calls from simulation data.
    Handles single or multiple simulations within data.
    Parses and validates tool call entries only from messages with role "tool".
    Ignores empty or non-JSON tool call contents.
    Returns a list of normalized tool call dicts with keys:
        - name
        - params (dict)
        - cost (float, default 0)
        - latency (float, default 0)
        - correct (bool, default True)
        - params_valid (bool, default True)
    """
    tool_calls = []
    simulations = data.get("simulations", [data])
    if not isinstance(simulations, list):
        simulations = [simulations]
    for sim in simulations:
        messages = sim.get("messages", [])
        for msg in messages:
            if not isinstance(msg, dict) or msg.get("role") != "tool":
                continue
            content_raw = msg.get("content", "")
            if not content_raw or (isinstance(content_raw, str) and content_raw.strip() == ""):
                logging.debug(f"Skipping empty tool call content in message: {msg}")
                continue
            try:
                # Parse content: can be JSON string or dict
                content = json.loads(content_raw) if isinstance(content_raw, str) else content_raw
            except Exception:
                logging.debug(f"Ignoring non-JSON tool call content: {content_raw}")
                continue
            # Normalize to list for consistent iteration
            calls = content if isinstance(content, list) else [content]
            for call in calls:
                if not isinstance(call, dict):
                    continue
                params_key = None
                if 'params' in call:
                    params_key = 'params'
                elif 'arguments' in call:
                    params_key = 'arguments'
                if 'name' in call and params_key and isinstance(call.get(params_key), dict):
                    tool_call_entry = {
                        'name': call['name'],
                        'params': call[params_key],
                        'cost': safe_float(call.get('cost')),
                        'latency': safe_float(call.get('latency')),
                        'correct': call.get('correct', True),
                        'params_valid': call.get('params_valid', True)
                    }
                    tool_calls.append(tool_call_entry)
                else:
                    logging.debug(f"Ignored invalid tool call entry: {call}")
    return tool_calls


def safe_float(val, default=0.0):
    """
    Convert value safely to float.
    Return default if conversion fails or value is None.
    """
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def calculate_tsr(data):
    """
    Calculate Task Success Rate (TSR) from simulation data.
    TSR is the fraction of runs with reward > 0.
    Supports both single simulation (dict with reward_info) and multiple simulations.
    Returns TSR value between 0 and 1.
    """
    try:
        if isinstance(data, dict) and 'reward_info' in data:
            reward = data.get('reward_info', {}).get('reward', 0)
            return 1.0 if safe_float(reward) > 0 else 0.0
        elif isinstance(data, dict) and 'simulations' in data:
            runs = data['simulations']
            if not isinstance(runs, list) or len(runs) == 0:
                logging.warning("Empty or invalid simulations list for TSR calculation.")
                return 0.0
            success_count = 0
            for run in runs:
                if isinstance(run, dict):
                    reward = run.get('reward_info', {}).get('reward', 0)
                    if safe_float(reward) > 0:
                        success_count += 1
            return success_count / len(runs)
        else:
            logging.warning("No valid reward_info found for calculating TSR.")
            return 0.0
    except Exception as e:
        logging.error(f"Error calculating TSR: {e}")
        return 0.0


def compute_tue_for_domain(sim_dir, domain_files):
    """
    Compute Tool Usage Efficiency (TUE) aggregated over simulations in a domain.
    - Collects all tool call costs and latencies to compute 95th percentile caps.
    - Calculates weighted TUE per file using:
        TUE = 0.4 * correctness + 0.25 * params_valid + 0.2 * cost_efficiency + 0.15 * latency_efficiency
    - Prints and logs per-file and aggregate TUE.
    """
    if not domain_files:
        logging.warning("No domain files provided for TUE computation")
        return 0.0
    tue_scores = []
    all_costs = []
    all_latencies = []
    # First pass: gather costs and latencies for caps
    for json_file in domain_files:
        filepath = os.path.join(sim_dir, json_file)
        if not os.path.exists(filepath):
            logging.warning(f"File not found: {filepath}")
            continue
        data = load_data(filepath)
        if not data:
            continue
        tool_calls = extract_tool_calls(data)
        for call in tool_calls:
            cost = call.get('cost')
            latency = call.get('latency')
            if cost is not None and cost > 0:
                all_costs.append(cost)
            if latency is not None and latency > 0:
                all_latencies.append(latency)
    cost_cap = np.percentile(all_costs, 95) if all_costs else 1
    latency_cap = np.percentile(all_latencies, 95) if all_latencies else 1
    cost_cap = max(cost_cap, 0.001)  # prevent division by zero
    latency_cap = max(latency_cap, 0.001)

    # Second pass: compute TUE per file
    for json_file in domain_files:
        filepath = os.path.join(sim_dir, json_file)
        if not os.path.exists(filepath):
            continue
        data = load_data(filepath)
        if not data:
            continue
        tool_calls = extract_tool_calls(data)
        if not tool_calls:
            logging.info(f"{json_file} - No valid tool calls found, skipping TUE.")
            print(f"{json_file} - No valid tool calls found, skipping TUE.")
            continue

        num_total = len(tool_calls)
        num_correct = sum(1 for call in tool_calls if call.get('correct', True))
        num_valid = sum(1 for call in tool_calls if call.get('params_valid', True))

        act_cost = sum(safe_float(call.get('cost')) for call in tool_calls)
        latencies = [safe_float(call.get('latency')) for call in tool_calls 
                     if call.get('latency') is not None and safe_float(call.get('latency')) > 0]
        act_latency = np.mean(latencies) if latencies else 0

        T = num_correct / num_total if num_total > 0 else 0
        P = num_valid / num_total if num_total > 0 else 0
        C = max(0, min(1, 1 - (act_cost / cost_cap)))
        L = max(0, min(1, 1 - (act_latency / latency_cap)))

        tue = 0.4 * T + 0.25 * P + 0.2 * C + 0.15 * L
        tue_scores.append(tue)

        logging.info(f"{json_file} - TUE: {tue:.2%} (T:{T:.2f}, P:{P:.2f}, C:{C:.2f}, L:{L:.2f})")
        print(f"{json_file:<70} TUE: {tue:.2%} (T:{T:.2f}, P:{P:.2f}, C:{C:.2f}, L:{L:.2f})")

    avg_tue = np.mean(tue_scores) if tue_scores else 0
    logging.info(f"Domain Average TUE: {avg_tue:.2%}")
    print(f"\nDomain Average TUE: {avg_tue:.2%}")
    return avg_tue


def normalized_params(params):
    """
    Recursively normalize parameters for consistency,
    to enable comparison and duplication detection regardless of order.
    """
    if not isinstance(params, dict):
        return str(params)  # convert non-dict to string for comparison

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


def compute_tcrr(tool_calls):
    """
    Compute Tool-Call Redundancy Ratio (TCRR):
    fraction of tool calls that are duplicates (same 'name' and identical parameters).
    """
    if not tool_calls:
        return 0.0
    seen_calls = set()
    duplicate_count = 0
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        name = call.get('name', '')
        params = call.get('params', {})
        try:
            params_norm = normalized_params(params)
            identity = (name, params_norm)
            if identity in seen_calls:
                duplicate_count += 1
            else:
                seen_calls.add(identity)
        except Exception as e:
            logging.warning(f"Error normalizing params for TCRR: {e}")
            # Fallback: string representation
            identity = (name, str(params))
            if identity in seen_calls:
                duplicate_count += 1
            else:
                seen_calls.add(identity)
    return duplicate_count / len(tool_calls) if tool_calls else 0.0


def compute_tcrr_for_domain(sim_dir, domain_files):
    """
    Compute average TCRR over all JSON files in a domain.
    """
    if not domain_files:
        logging.warning("No domain files provided for TCRR computation")
        return 0.0
    tcrr_scores = []
    for json_file in domain_files:
        filepath = os.path.join(sim_dir, json_file)
        if not os.path.exists(filepath):
            logging.warning(f"File not found: {filepath}")
            continue
        data = load_data(filepath)
        if not data:
            continue
        tool_calls = extract_tool_calls(data)
        tcrr = compute_tcrr(tool_calls)
        logging.info(f"{json_file} - TCRR: {tcrr:.2%}")
        print(f"{json_file:<70} TCRR: {tcrr:.2%}")
        tcrr_scores.append(tcrr)
    avg_tcrr = np.mean(tcrr_scores) if tcrr_scores else 0
    logging.info(f"Average TCRR for domain: {avg_tcrr:.2%}")
    print(f"\nAverage TCRR for domain: {avg_tcrr:.2%}")
    return avg_tcrr


def main():
    """
    Main driver function:
    - Iterates over configured domains.
    - Lists domain files.
    - For each domain, computes and prints TSR, TUE, and TCRR metrics,
      with readable formatting and error handling.
    """
    sim_dir = '../data/simulations'  # Adjust to your simulations directory
    domains = ['retail', 'telecom', 'banking', 'airline', 'mock']

    for domain in domains:
        print("\n" + "="*80)
        print(f"Processing domain: {domain.upper()}")
        print("="*80 + "\n")

        try:
            all_files = os.listdir(sim_dir)
            domain_files = [f for f in all_files if domain in f and f.endswith('.json')]
        except Exception as e:
            print(f"Error accessing directory {sim_dir}: {e}")
            logging.error(f"Error accessing directory {sim_dir}: {e}")
            continue

        if not domain_files:
            print(f"No JSON files found for domain: {domain} in {sim_dir}\n")
            continue

        print(f"Found {len(domain_files)} JSON file(s) for domain '{domain}':")
        for f in domain_files:
            print(f"  - {f}")
        print()

        # Calculate and print TSR
        print("== Task Success Rate (TSR) ==")
        tsr_values = []
        for json_file in domain_files:
            filepath = os.path.join(sim_dir, json_file)
            data = load_data(filepath)
            if not data:
                print(f"  [!] Failed to load data: {json_file}")
                continue
            tsr = calculate_tsr(data)
            tsr_values.append(tsr)
            print(f"  {json_file:<70} TSR = {tsr:.2%}")
        if tsr_values:
            avg_tsr = np.mean(tsr_values)
            print(f"  --> Average TSR for {domain}: {avg_tsr:.2%} ({len(tsr_values)} runs)\n")
        else:
            print(f"  No valid TSR values computed for {domain}.\n")

        # Calculate and print TUE
        print("== Tool Usage Efficiency (TUE) ==")
        try:
            avg_tue = compute_tue_for_domain(sim_dir, domain_files)
            print(f"  --> Average TUE for {domain}: {avg_tue:.2%}\n")
        except Exception as e:
            print(f"  [!] Error computing TUE for domain {domain}: {e}\n")
            logging.error(f"Error computing TUE for domain {domain}: {e}")

        # Calculate and print TCRR
        print("== Tool-Call Redundancy Ratio (TCRR) ==")
        try:
            avg_tcrr = compute_tcrr_for_domain(sim_dir, domain_files)
            print(f"  --> Average TCRR for {domain}: {avg_tcrr:.2%}\n")
        except Exception as e:
            print(f"  [!] Error computing TCRR for domain {domain}: {e}\n")
            logging.error(f"Error computing TCRR for domain {domain}: {e}")

        print("-"*80)

    print("\nEvaluation complete. See 'metrics_evaluation.log' for details.")


if __name__ == "__main__":
    main()
