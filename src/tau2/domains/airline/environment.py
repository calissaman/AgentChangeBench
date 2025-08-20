# Copyright Sierra
import json
from typing import Optional

from tau2.data_model.tasks import Task
from tau2.domains.airline.data_model import FlightDB
from tau2.domains.airline.tools import AirlineTools
from tau2.domains.airline.user_tools import AirlineUserTools
from tau2.domains.airline.user_data_model import AirlineUserDB
from tau2.domains.airline.utils import (
    AIRLINE_DB_PATH,
    AIRLINE_POLICY_PATH,
    AIRLINE_TASK_SET_PATH,
    AIRLINE_USER_POLICY_PATH,
)
from tau2.environment.environment import Environment
from tau2.utils import load_file


def get_environment(
    db: Optional[FlightDB] = None,
    user_db: Optional[AirlineUserDB] = None,
    solo_mode: bool = False,
) -> Environment:
    if solo_mode:
        raise ValueError("Airline domain does not support solo mode")
    if db is None:
        db = FlightDB.load(AIRLINE_DB_PATH)
    tools = AirlineTools(db)

    if user_db is None:
        user_db = AirlineUserDB.load()
    user_tools = AirlineUserTools(user_db)

    if solo_mode:
        policy = load_file(AIRLINE_USER_POLICY_PATH)
    else:
        policy = load_file(AIRLINE_POLICY_PATH)

    return Environment(
        domain_name="airline",
        policy=policy,
        tools=tools,
        user_tools=user_tools,
    )


def get_tasks() -> list[Task]:
    with open(AIRLINE_TASK_SET_PATH, "r") as fp:
        tasks = json.load(fp)
    return [Task.model_validate(task) for task in tasks]
