from functools import partial
from typing import Optional
import json

from tau2.data_model.tasks import Task
from tau2.domains.banking.data_model import BankingDB
from tau2.domains.banking.tools import BankingTools

from tau2.domains.banking.user_data_model import BankingUserDB
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


class BankingUserTools(ToolKitBase):
    """
    Minimal user tools for banking domain.
    Users no longer have tools - information is provided through known_info instead.
    """

    db: BankingUserDB

    def __init__(self, db: BankingUserDB):
        super().__init__(db)

    def update_user(self, **kwargs):
        """Update user data with the provided key-value pairs."""
        for key, value in kwargs.items():
            if hasattr(self.db, key):
                setattr(self.db, key, value)


class BankingEnvironment(Environment):
    tools: BankingTools
    user_tools: BankingUserTools

    def __init__(
        self,
        domain_name: str,
        policy: str,
        tools: BankingTools,
        user_tools: BankingUserTools,
    ):
        super().__init__(domain_name, policy, tools, user_tools)

    def sync_tools(self):
        """
        Sync environment state with current user status.
        Since users no longer have tools, this mainly updates user data model state.
        """
        phone = self.user_tools.db.phone_number
        if not phone:
            return

        try:
            customer = self.tools.get_customer_by_phone(phone)
            self.user_tools.db.customer_id = customer.customer_id

            # Sync primary account status
            if customer.account_ids:
                account = self.tools.get_account(customer.account_ids[0])
                self.user_tools.db.primary_account_active = account.status == "Active"
                self.user_tools.db.primary_account_id = account.account_id
                self.user_tools.db.account_balance = account.current_balance

            # Sync first card status
            if customer.card_ids:
                card = self.tools._get_card(customer.card_ids[0])
                self.user_tools.db.primary_card_active = card.status == "Active"
                self.user_tools.db.primary_card_id = card.card_id

        except Exception as e:
            # If sync fails, just continue - user tools will use default values
            print(f"Error syncing tools: {e}")
            pass


def get_environment(
    db: Optional[BankingDB] = None,
    user_db: Optional[BankingUserDB] = None,
    solo_mode: bool = False,
) -> BankingEnvironment:
    if db is None:
        db = BankingDB.load(BANKING_DB_PATH)
    tools = BankingTools(db)

    if user_db is None:
        user_db = BankingUserDB.load()
    user_tools = BankingUserTools(user_db)

    if solo_mode:
        policy = load_file(BANKING_USER_POLICY_PATH)
    else:
        policy = load_file(BANKING_POLICY_PATH)

    env = BankingEnvironment(
        domain_name="banking",
        policy=policy,
        tools=tools,
        user_tools=user_tools,
    )
    if solo_mode:
        env.set_solo_mode(True)

    return env


def load_personas() -> dict:
    """Load persona definitions from user_personas.json"""
    personas_path = BANKING_DATA_DIR / "user_personas.json"
    try:
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


def load_tasks(path: str) -> list[Task]:
    tasks = load_file(path)
    if isinstance(tasks, dict) and "tasks" in tasks:
        tasks = tasks["tasks"]

    personas = load_personas()

    processed_tasks = []
    for task in tasks:
        task_with_persona = inject_persona_data(task, personas)
        processed_tasks.append(Task.model_validate(task_with_persona))

    return processed_tasks


def get_environment_main() -> BankingEnvironment:
    return get_environment(solo_mode=False)


def get_environment_solo() -> BankingEnvironment:
    return get_environment(solo_mode=True)


def get_tasks() -> list[Task]:
    return load_tasks(BANKING_TASK_SET_PATH)
