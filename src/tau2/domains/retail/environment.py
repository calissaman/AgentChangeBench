# Copyright Sierra
import json
from typing import Optional

from tau2.data_model.tasks import Task
from tau2.domains.retail.data_model import RetailDB
from tau2.domains.retail.tools import RetailTools
from tau2.domains.retail.utils import (
    RETAIL_DB_PATH,
    RETAIL_POLICY_PATH,
    RETAIL_TASK_SET_PATH,
    RETAIL_USER_POLICY_PATH,
)
from tau2.environment.environment import Environment
from tau2.utils import load_file, DATA_DIR


def get_environment(
    db: Optional[RetailDB] = None,
    solo_mode: bool = False,
) -> Environment:
    if solo_mode:
        raise ValueError("Retail domain does not support solo mode")
    if db is None:
        db = RetailDB.load(RETAIL_DB_PATH)
    tools = RetailTools(db)

    if solo_mode:
        policy = load_file(RETAIL_USER_POLICY_PATH)
    else:
        policy = load_file(RETAIL_POLICY_PATH)

    return Environment(
        domain_name="retail",
        policy=policy,
        tools=tools,
    )


def load_personas() -> dict:
    """Load user personas from the retail domain personas file"""
    try:
        personas_path = DATA_DIR / "tau2" / "domains" / "retail" / "user_personas.json"
        with open(personas_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def inject_persona_data(task_data: dict, personas: dict) -> dict:
    """Inject persona data into task based on persona key"""
    if "user_scenario" in task_data and "persona" in task_data["user_scenario"]:
        persona_key = task_data["user_scenario"]["persona"]
        if persona_key in personas:
            task_data["user_scenario"]["persona"] = personas[persona_key]
    return task_data


def get_tasks() -> list[Task]:
    with open(RETAIL_TASK_SET_PATH, "r") as fp:
        tasks = json.load(fp)

    personas = load_personas()

    processed_tasks = []
    for task in tasks:
        task_with_persona = inject_persona_data(task, personas)
        processed_tasks.append(Task.model_validate(task_with_persona))

    return processed_tasks
