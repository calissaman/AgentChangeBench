from functools import partial
from typing import Optional
import json

from tau2.data_model.tasks import Task
from tau2.domains.banking.data_model import BankingDB
from tau2.domains.banking.tools import BankingTools

from tau2.domains.banking.utilts import (
    BANKING_DB_PATH,
    BANKING_POLICY_PATH,
    BANKING_USER_POLICY_PATH,
    BANKING_TASK_SET_PATH,
    BANKING_DATA_DIR,
)
from tau2.environment.environment import Environment
from tau2.environment.toolkit import ToolKitBase
from tau2.utils import load_file


def get_environment(
    db: Optional[BankingDB] = None,
    solo_mode: bool = False,
) -> Environment:
    if db is None:
        db = BankingDB.load(BANKING_DB_PATH)
    tools = BankingTools(db)

    if solo_mode:
        policy = load_file(BANKING_USER_POLICY_PATH)
    else:
        policy = load_file(BANKING_POLICY_PATH)

    return Environment(
        domain_name="banking",
        policy=policy,
        tools=tools,
    )


def load_personas() -> dict:
    """Load user personas from the banking domain personas file"""
    try:
        personas_path = BANKING_DATA_DIR / "user_personas.json"
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
    with open(BANKING_TASK_SET_PATH, "r") as fp:
        task_data_list = json.load(fp)

    # Load personas for injection
    personas = load_personas()

    tasks = []
    for task_data in task_data_list:
        # Inject persona data if available
        task_data = inject_persona_data(task_data, personas)
        tasks.append(Task(**task_data))

    return tasks
