#!/usr/bin/env python3
"""
Generate systematic retail task matrix.
"""

import json
import random
from typing import Dict, List, Any

# Retail use cases with scenarios
RETAIL_USE_CASES = {
    "order_management": {
        "single": "Track order status and get delivery updates",
        "soft": "Check order status, then modify delivery address", 
        "hard": "Track delayed order, escalate to complaint about service"
    },
    "product_return": {
        "single": "Return defective product and get refund",
        "soft": "Return wrong size item, then ask about exchange policy",
        "hard": "Return multiple defective items, demand compensation"
    },
    "product_exchange": {
        "single": "Exchange shirt for different size",
        "soft": "Exchange product, then inquire about return policy",
        "hard": "Exchange defective item, escalate to warranty claim"
    },
    "account_management": {
        "single": "Update delivery address for future orders",
        "soft": "Change address, then verify payment method details",
        "hard": "Update account after identity theft, need security review"
    },
    "payment_issues": {
        "single": "Resolve declined payment on recent order",
        "soft": "Fix payment issue, then check order status",
        "hard": "Dispute unauthorized charge, demand investigation"
    },
    "product_inquiry": {
        "single": "Check availability of specific product model",
        "soft": "Ask about product features, then inquire about warranty",
        "hard": "Complain about product quality, demand explanation"
    },
    "shipping_delivery": {
        "single": "Check delivery status for pending order",
        "soft": "Track shipment, then modify delivery instructions",
        "hard": "Report lost package, demand replacement and compensation"
    },
    "customer_support": {
        "single": "Get general information about return policy",
        "soft": "Ask about policies, then request specific account help",
        "hard": "File formal complaint about service quality"
    }
}

# Persona customer data
PERSONA_DATA = {
    "EASY_1": {"name": "Emma Johnson", "zip": "90210", "customer_id": "easy1_customer"},
    "EASY_2": {"name": "Mike Chen", "zip": "10001", "customer_id": "easy2_customer"},
    "MEDIUM_1": {"name": "Sarah Martinez", "zip": "60601", "customer_id": "medium1_customer"},
    "MEDIUM_2": {"name": "Lisa Wang", "zip": "94102", "customer_id": "medium2_customer"},
    "HARD_1": {"name": "David Thompson", "zip": "33101", "customer_id": "hard1_customer"}
}

def generate_goal_shifts(use_case: str, pattern: str) -> Dict[str, Any]:
    """Generate goal shifts based on pattern."""
    if pattern == "single":
        return {"required_shifts": 0, "goals": [use_case]}
    
    # Goal shift patterns
    shift_map = {
        "order_management": ["order_tracking", "account_update"] if pattern == "soft" else ["order_tracking", "customer_complaint"],
        "product_return": ["product_return", "product_exchange"] if pattern == "soft" else ["product_return", "payment_dispute"],
        "product_exchange": ["product_exchange", "product_inquiry"] if pattern == "soft" else ["product_exchange", "warranty_claim"],
        "account_management": ["account_update", "order_inquiry"] if pattern == "soft" else ["account_update", "security_issue"],
        "payment_issues": ["payment_issue", "order_status"] if pattern == "soft" else ["payment_issue", "fraud_report"],
        "product_inquiry": ["product_inquiry", "order_placement"] if pattern == "soft" else ["product_inquiry", "quality_complaint"],
        "shipping_delivery": ["delivery_tracking", "address_change"] if pattern == "soft" else ["delivery_tracking", "service_complaint"],
        "customer_support": ["general_inquiry", "specific_request"] if pattern == "soft" else ["general_inquiry", "escalation"]
    }
    
    goals = shift_map.get(use_case, [use_case, "general_inquiry"])
    return {"required_shifts": 1, "goals": goals}

def generate_task(use_case: str, persona: str, pattern: str, task_number: int) -> Dict[str, Any]:
    """Generate a single systematic task."""
    task_id = f"retail_{use_case}_{persona.lower()}_{pattern}_{task_number:03d}_systematic"
    
    # Get scenario and data
    scenario = RETAIL_USE_CASES[use_case][pattern]
    persona_data = PERSONA_DATA[persona]
    goal_shifts = generate_goal_shifts(use_case, pattern)
    
    # Generate known info
    known_info = f"You are {persona_data['name']} in zip code {persona_data['zip']}."
    if use_case in ["order_management", "product_return", "product_exchange", "shipping_delivery"]:
        known_info += f" You have order #R{random.randint(1000000, 9999999)}."
    
    task = {
        "id": task_id,
        "description": {
            "purpose": f"Systematic retail task: {use_case} with {pattern} goal pattern",
            "relevant_policies": "Retail customer service policies",
            "notes": f"Use case: {use_case}, Persona: {persona}, Pattern: {pattern}"
        },
        "user_scenario": {
            "persona": persona,
            "instructions": {
                "domain": "retail",
                "reason_for_call": scenario,
                "known_info": known_info,
                "unknown_info": "Specific product IDs or order details not mentioned",
                "task_instructions": f"Act according to {persona} persona. Complete your {use_case} request."
            },
            "goal_shifts": goal_shifts
        },
        "initial_state": {
            "initialization_data": {
                "agent_data": None,
                "user_data": {
                    "authenticated": False,
                    "user_id": None,
                    "name": persona_data["name"],
                    "zip_code": persona_data["zip"],
                    "customer_id": persona_data["customer_id"]
                }
            },
            "initialization_actions": [
                {
                    "env_type": "user",
                    "func_name": "update_user",
                    "arguments": {
                        "authenticated": False,
                        "name": persona_data["name"],
                        "zip_code": persona_data["zip"],
                        "customer_id": persona_data["customer_id"]
                    }
                }
            ],
            "message_history": []
        },
        "evaluation_criteria": {
            "actions": [],
            "nl_assertions": [
                f"User successfully completes their {use_case} request",
                "Agent properly handles user authentication",
                "Conversation follows retail customer service policies"
            ]
        }
    }
    
    if goal_shifts["required_shifts"] > 0:
        task["evaluation_criteria"]["nl_assertions"].append(
            f"User successfully transitions from {goal_shifts['goals'][0]} to {goal_shifts['goals'][1]}"
        )
    
    return task

def main():
    """Generate systematic retail task matrix."""
    print("🔄 Generating systematic retail task matrix...")
    
    use_cases = list(RETAIL_USE_CASES.keys())
    personas = list(PERSONA_DATA.keys()) 
    patterns = ["single", "soft", "hard"]
    
    print(f"📊 Matrix: {len(use_cases)} use cases × {len(personas)} personas × {len(patterns)} patterns = {len(use_cases) * len(personas) * len(patterns)} tasks")
    
    tasks = []
    task_counter = 1
    
    for use_case in use_cases:
        for persona in personas:
            for pattern in patterns:
                task = generate_task(use_case, persona, pattern, task_counter)
                tasks.append(task)
                task_counter += 1
    
    print(f"✅ Generated {len(tasks)} systematic tasks")
    
    # Load existing tasks
    with open('data/tau2/domains/retail/tasks.json', 'r') as f:
        existing_tasks = json.load(f)
    
    print(f"📊 Found {len(existing_tasks)} existing tasks")
    
    # Combine
    all_tasks = existing_tasks + tasks
    print(f"🔗 Combined total: {len(all_tasks)} tasks")
    
    # Save
    with open('data/tau2/domains/retail/tasks.json', 'w') as f:
        json.dump(all_tasks, f, indent=2)
    
    print(f"✅ Saved {len(all_tasks)} total retail tasks")
    print("🎯 Retail domain matrix generation complete!")

if __name__ == "__main__":
    main()
