# Copyright Sierra
import json
from typing import Optional

from tau2.data_model.tasks import Task
from tau2.domains.retail.data_model import RetailDB
from tau2.domains.retail.tools import RetailTools
from tau2.domains.retail.user_tools import RetailUserTools
from tau2.domains.retail.user_data_model import RetailUserDB
from tau2.domains.retail.utils import (
    RETAIL_DB_PATH,
    RETAIL_POLICY_PATH,
    RETAIL_TASK_SET_PATH,
    RETAIL_USER_POLICY_PATH,
)
from tau2.environment.environment import Environment
from tau2.utils import load_file


def get_environment(
    db: Optional[RetailDB] = None,
    user_db: Optional[RetailUserDB] = None,
    solo_mode: bool = False,
) -> Environment:
    if solo_mode:
        raise ValueError("Retail domain does not support solo mode")
    if db is None:
        db = RetailDB.load(RETAIL_DB_PATH)
    tools = RetailTools(db)

    if user_db is None:
        user_db = RetailUserDB.load()
    user_tools = RetailUserTools(user_db)

    if solo_mode:
        policy = load_file(RETAIL_USER_POLICY_PATH)
    else:
        policy = load_file(RETAIL_POLICY_PATH)

    return Environment(
        domain_name="retail",
        policy=policy,
        tools=tools,
        user_tools=user_tools,
    )


def get_tasks() -> list[Task]:
    with open(RETAIL_TASK_SET_PATH, "r") as fp:
        tasks = json.load(fp)
    return [Task.model_validate(task) for task in tasks]
