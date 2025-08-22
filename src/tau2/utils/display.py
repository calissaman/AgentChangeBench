import json
from typing import List, Optional

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from tau2.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.simulation import RunConfig, SimulationRun
from tau2.data_model.tasks import Action, Task
from tau2.metrics.agent_metrics import AgentMetrics, is_successful


class ConsoleDisplay:
    console = Console()

    @classmethod
    def display_run_config(cls, config: RunConfig):
        # Create layout
        layout = Layout()

        # Split layout into sections
        layout.split(Layout(name="header"), Layout(name="body"))

        # Split body into columns
        layout["body"].split_row(
            Layout(name="agent", ratio=1),
            Layout(name="user", ratio=1),
            Layout(name="settings", ratio=1),
        )

        # Create content for each section
        header_content = Panel(
            f"[white]Domain:[/] {config.domain}\n"
            f"[white]Task Set:[/] {config.task_set_name if config.task_set_name else 'Default'}\n"
            f"[white]Task IDs:[/] {', '.join(map(str, config.task_ids)) if config.task_ids else 'All'}\n"
            f"[white]Number of trials:[/] {config.num_trials}\n"
            f"[white]Max steps:[/] {config.max_steps}\n"
            f"[white]Max errors:[/] {config.max_errors}",
            title="[bold blue]Simulation Configuration",
            border_style="blue",
        )

        agent_content = Panel(
            f"[white]Implementation:[/] {config.agent}\n"
            f"[white]Model:[/] {config.llm_agent}\n"
            "[white]LLM Arguments:[/]\n"
            f"{json.dumps(config.llm_args_agent, indent=2)}",
            title="[bold cyan]Agent Configuration",
            border_style="cyan",
        )

        user_content = Panel(
            f"[white]Implementation:[/] {config.user}\n"
            f"[white]Model:[/] {config.llm_user}\n"
            "[white]LLM Arguments:[/]\n"
            f"{json.dumps(config.llm_args_user, indent=2)}",
            title="[bold cyan]User Configuration",
            border_style="cyan",
        )

        settings_content = Panel(
            f"[white]Save To:[/] {config.save_to or 'Not specified'}\n"
            f"[white]Max Concurrency:[/] {config.max_concurrency}",
            title="[bold cyan]Additional Settings",
            border_style="cyan",
        )

        # Assign content to layout sections
        layout["header"].update(header_content)
        layout["agent"].update(agent_content)
        layout["user"].update(user_content)
        layout["body"]["settings"].update(settings_content)

        # Print the layout
        cls.console.print(layout)

    @classmethod
    def display_task(cls, task: Task):
        # Build content string showing only non-None fields
        content_parts = []

        if task.id is not None:
            content_parts.append(f"[white]ID:[/] {task.id}")

        if task.description:
            if task.description.purpose:
                content_parts.append(f"[white]Purpose:[/] {task.description.purpose}")
            if task.description.relevant_policies:
                content_parts.append(
                    f"[white]Relevant Policies:[/] {task.description.relevant_policies}"
                )
            if task.description.notes:
                content_parts.append(f"[white]Notes:[/] {task.description.notes}")

        # User Scenario section
        scenario_parts = []
        # Persona
        if task.user_scenario.persona:
            scenario_parts.append(f"[white]Persona:[/] {task.user_scenario.persona}")

        # User Instruction
        scenario_parts.append(
            f"[white]Task Instructions:[/] {task.user_scenario.instructions}"
        )

        if scenario_parts:
            content_parts.append(
                "[bold cyan]User Scenario:[/]\n" + "\n".join(scenario_parts)
            )

        # Initial State section
        if task.initial_state:
            initial_state_parts = []
            if task.initial_state.initialization_data:
                initial_state_parts.append(
                    f"[white]Initialization Data:[/]\n{task.initial_state.initialization_data.model_dump_json(indent=2)}"
                )
            if task.initial_state.initialization_actions:
                initial_state_parts.append(
                    f"[white]Initialization Actions:[/]\n{json.dumps([a.model_dump() for a in task.initial_state.initialization_actions], indent=2)}"
                )
            if task.initial_state.message_history:
                initial_state_parts.append(
                    f"[white]Message History:[/]\n{json.dumps([m.model_dump() for m in task.initial_state.message_history], indent=2)}"
                )

            if initial_state_parts:
                content_parts.append(
                    "[bold cyan]Initial State:[/]\n" + "\n".join(initial_state_parts)
                )

        # Evaluation Criteria section
        if task.evaluation_criteria:
            eval_parts = []
            if task.evaluation_criteria.actions:
                eval_parts.append(
                    f"[white]Required Actions:[/]\n{json.dumps([a.model_dump() for a in task.evaluation_criteria.actions], indent=2)}"
                )
            if task.evaluation_criteria.env_assertions:
                eval_parts.append(
                    f"[white]Env Assertions:[/]\n{json.dumps([a.model_dump() for a in task.evaluation_criteria.env_assertions], indent=2)}"
                )
            if task.evaluation_criteria.communicate_info:
                eval_parts.append(
                    f"[white]Information to Communicate:[/]\n{json.dumps(task.evaluation_criteria.communicate_info, indent=2)}"
                )
            if eval_parts:
                content_parts.append(
                    "[bold cyan]Evaluation Criteria:[/]\n" + "\n".join(eval_parts)
                )
        content = "\n\n".join(content_parts)

        # Create and display panel
        task_panel = Panel(
            content, title="[bold blue]Task Details", border_style="blue", expand=True
        )

        cls.console.print(task_panel)

    @classmethod
    def _display_tsr_actions(cls, sim_info, task, simulation):
        """Display actions based on TSR evaluation criteria."""
        from tau2.metrics.tsr import extract_action_sets_from_task
        
        # Check for new action_sets format first
        action_sets = extract_action_sets_from_task(task)
        if action_sets:
            sim_info.append("\nAction Checks:\n", style="bold magenta")
            
            # Extract tool calls from simulation
            tool_calls = []
            for msg in simulation.messages:
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        tool_calls.append({
                            'name': tool_call.name,
                            'arguments': getattr(tool_call, 'arguments', {})
                        })
            
            for i, action_set in enumerate(action_sets):
                action_id = action_set.action_id
                allowed_tools = action_set.allowed_tools
                
                if len(allowed_tools) == 1:
                    # Single tool expected
                    tool = allowed_tools[0]
                    func_name = tool.get('function_name', '')
                    params = tool.get('params', {})
                    
                    # Check if this tool was called
                    tool_called = any(
                        tc['name'] == func_name and 
                        all(tc['arguments'].get(k) == v for k, v in params.items())
                        for tc in tool_calls
                    )
                    
                    # Format parameters for display
                    param_str = ', '.join(f"{k}={repr(v)}" for k, v in params.items())
                    display_name = f"{func_name}({param_str})"
                    
                    sim_info.append(f"- {i}: {display_name} {'✅' if tool_called else '❌'}\n")
                
                else:
                    # Multiple tools allowed - show list
                    sim_info.append(f"- {i}: {action_id}\n")
                    for tool in allowed_tools:
                        func_name = tool.get('function_name', '')
                        params = tool.get('params', {})
                        
                        # Check if this specific tool was called
                        tool_called = any(
                            tc['name'] == func_name and 
                            all(tc['arguments'].get(k) == v for k, v in params.items())
                            for tc in tool_calls
                        )
                        
                        # Format parameters for display
                        param_str = ', '.join(f"{k}={repr(v)}" for k, v in params.items())
                        display_name = f"{func_name}({param_str})"
                        
                        sim_info.append(f"    • {display_name} {'✅' if tool_called else '❌'}\n")
        
        # Fallback to old actions format if no action_sets
        elif (task.evaluation_criteria and 
              hasattr(task.evaluation_criteria, 'actions') and 
              task.evaluation_criteria.actions):
            
            sim_info.append("\nAction Checks:\n", style="bold magenta")
            
            # Extract tool calls from simulation
            tool_calls = []
            for msg in simulation.messages:
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        tool_calls.append({
                            'name': tool_call.name,
                            'arguments': getattr(tool_call, 'arguments', {})
                        })
            
            for i, action in enumerate(task.evaluation_criteria.actions):
                func_name = action.get('name', '')
                expected_args = action.get('arguments', {})
                
                # Check if this action was performed
                action_performed = any(
                    tc['name'] == func_name and 
                    all(tc['arguments'].get(k) == v for k, v in expected_args.items())
                    for tc in tool_calls
                )
                
                # Format parameters for display
                param_str = ', '.join(f"{k}={repr(v)}" for k, v in expected_args.items())
                display_name = f"{func_name}({param_str})"
                
                sim_info.append(f"- {i}: {display_name} {'✅' if action_performed else '❌'}\n")

    @classmethod
    def display_simulation(cls, simulation: SimulationRun, show_details: bool = True, task=None):
        """
        Display the simulation content in a formatted way using Rich library.

        Args:
            simulation: The simulation object to display
            show_details: Whether to show detailed information
        """
        # Create main simulation info panel
        sim_info = Text()
        if show_details:
            sim_info.append("Simulation ID: ", style="bold cyan")
            sim_info.append(f"{simulation.id}\n")
        sim_info.append("Task ID: ", style="bold cyan")
        sim_info.append(f"{simulation.task_id}\n")
        sim_info.append("Trial: ", style="bold cyan")
        sim_info.append(f"{simulation.trial}\n")
        if show_details:
            sim_info.append("Start Time: ", style="bold cyan")
            sim_info.append(f"{simulation.start_time}\n")
            sim_info.append("End Time: ", style="bold cyan")
            sim_info.append(f"{simulation.end_time}\n")
        sim_info.append("Duration: ", style="bold cyan")
        sim_info.append(f"{simulation.duration:.2f}s\n")
        sim_info.append("Termination Reason: ", style="bold cyan")
        sim_info.append(f"{simulation.termination_reason}\n")
        if simulation.agent_cost is not None:
            sim_info.append("Agent Cost: ", style="bold cyan")
            sim_info.append(f"${simulation.agent_cost:.4f}\n")
        if simulation.user_cost is not None:
            sim_info.append("User Cost: ", style="bold cyan")
            sim_info.append(f"${simulation.user_cost:.4f}\n")
        if simulation.reward_info:
            sim_info.append("Reward: ", style="bold cyan")
            if simulation.reward_info.reward_breakdown:
                breakdown = sorted(
                    [
                        f"{k.value}: {v:.1f}"
                        for k, v in simulation.reward_info.reward_breakdown.items()
                    ]
                )
            else:
                breakdown = []

            sim_info.append(
                f"{simulation.reward_info.reward:.4f} ({', '.join(breakdown)})\n"
            )

            # Add DB check info if present
            if simulation.reward_info.db_check:
                sim_info.append("\nDB Check:", style="bold magenta")
                sim_info.append(
                    f"{'✅' if simulation.reward_info.db_check.db_match else '❌'} {simulation.reward_info.db_check.db_reward}\n"
                )

            # Add env assertions if present
            if simulation.reward_info.env_assertions:
                sim_info.append("\nEnv Assertions:\n", style="bold magenta")
                for i, assertion in enumerate(simulation.reward_info.env_assertions):
                    sim_info.append(
                        f"- {i}: {assertion.env_assertion.env_type} {assertion.env_assertion.func_name} {'✅' if assertion.met else '❌'} {assertion.reward}\n"
                    )

            # Add action checks if present
            if simulation.reward_info.action_checks:
                sim_info.append("\nAction Checks:\n", style="bold magenta")
                for i, check in enumerate(simulation.reward_info.action_checks):
                    sim_info.append(
                        f"- {i}: {check.action.name} {'✅' if check.action_match else '❌'} {check.action_reward}\n"
                    )
            elif task:
                cls._display_tsr_actions(sim_info, task, simulation)

            # Add communication checks if present
            if simulation.reward_info.communicate_checks:
                sim_info.append("\nCommunicate Checks:\n", style="bold magenta")
                for i, check in enumerate(simulation.reward_info.communicate_checks):
                    sim_info.append(
                        f"- {i}: {check.info} {'✅' if check.met else '❌'}\n"
                    )

            # Add NL assertions if present
            if simulation.reward_info.nl_assertions:
                sim_info.append("\nNL Assertions:\n", style="bold magenta")
                for i, assertion in enumerate(simulation.reward_info.nl_assertions):
                    # Format: id: score then task + reason
                    score = "1.0" if assertion.met else "0.0"
                    score_color = "green" if assertion.met else "red"
                    sim_info.append("- ", style="white")
                    sim_info.append(f"{i}", style="bold cyan")
                    sim_info.append(": ", style="white")
                    sim_info.append(f"{score}", style=f"bold {score_color}")
                    sim_info.append(f" {assertion.nl_assertion} {'✅' if assertion.met else '❌'}\n", style="white")
                    sim_info.append(f"        {assertion.justification}\n", style="dim white")

            # Add task-based metrics if present
            if simulation.reward_info.info:
                sim_info.append("\nTask Metrics:\n", style="bold magenta")
                for key, value in simulation.reward_info.info.items():
                    if key == "gsrt_v2" and isinstance(value, dict):
                        sim_info.append("GSRT Analysis:\n", style="bold yellow")
                        if "start_goal" in value:
                            start = value["start_goal"]
                            sim_info.append(f"  Initial Goal: {start.get('goal', 'N/A')} (Turn {start.get('turn', 'N/A')})\n")
                        
                        if "user_goal_shifts" in value and value["user_goal_shifts"]:
                            sim_info.append(f"  Goal Shifts ({len(value['user_goal_shifts'])}):\n")
                            for i, shift in enumerate(value["user_goal_shifts"], 1):
                                sim_info.append(f"    {i}. Turn {shift.get('turn', '?')}: {shift.get('from', '?')} → {shift.get('to', '?')}\n")
                                sim_info.append(f"       Agent Response: Turn {shift.get('agent_turn', '?')}")
                                if shift.get('acknowledgment_turn'):
                                    sim_info.append(f", Acknowledged: Turn {shift.get('acknowledgment_turn')}")
                                if shift.get('tool_turn'):
                                    sim_info.append(f", Tool Used: Turn {shift.get('tool_turn')}")
                                if shift.get('outcome_turn'):
                                    sim_info.append(f", Outcome: Turn {shift.get('outcome_turn')}")
                                if shift.get('transferred_to_human'):
                                    sim_info.append(f", Transferred to Human: Yes")
                                sim_info.append("\n")
                        else:
                            sim_info.append("  No goal shifts detected\n")
                    
                    elif key == "tsr_details" and isinstance(value, dict):
                        sim_info.append("TSR Breakdown:\n", style="bold yellow")
                        overall_tsr = value.get('overall_tsr')
                        if overall_tsr is not None:
                            sim_info.append(f"  Overall TSR: {overall_tsr:.3f}/1.0\n")
                        if value.get('has_actions'):
                            action_score = value.get('action')
                            if action_score is not None:
                                sim_info.append(f"  Actions Score: {action_score:.3f}/1.0\n")
                        if value.get('has_nl_assertions'):
                            nl_score = value.get('nl_assertion')
                            if nl_score is not None:
                                sim_info.append(f"  NL Assertions Score: {nl_score:.3f}/1.0\n")
                        if value.get('has_communicate_info'):
                            comm_score = value.get('communicate_info')
                            if comm_score is not None:
                                sim_info.append(f"  Communication Score: {comm_score:.3f}/1.0\n")
                        if "weights_used" in value:
                            weights = value["weights_used"]
                            sim_info.append(f"  Weights Applied: {', '.join(f'{k}={v:.3f}' for k, v in weights.items())}\n")
                    
                    elif key.startswith("tue_") and isinstance(value, dict):
                        sim_info.append("TUE Analysis:\n", style="bold yellow")
                        sim_info.append(f"  Tool Usage Efficiency: {value.get('overall_tue', 'N/A'):.3f}\n")
                        sim_info.append(f"  Tool Correctness: {value.get('tool_correctness', 'N/A'):.3f}\n")
                        sim_info.append(f"  Parameter Accuracy: {value.get('param_accuracy', 'N/A'):.3f}\n")
                    
                    elif key.startswith("tcrr_") and isinstance(value, dict):
                        sim_info.append("TCRR Analysis:\n", style="bold yellow")
                        sim_info.append(f"  Redundancy Ratio: {value.get('ratio', 'N/A'):.3f}\n")
                        sim_info.append(f"  Total Calls: {value.get('total_calls', 'N/A')}\n")
                        sim_info.append(f"  Redundant Calls: {value.get('redundant_calls', 'N/A')}\n")
                        sim_info.append(f"  Window Size: {value.get('window_size', 'N/A')} turns\n")
                    
                    else:
                        sim_info.append(f"{key}: {value}\n")

        cls.console.print(
            Panel(sim_info, title="Simulation Overview", border_style="blue")
        )

        # Create messages table
        if simulation.messages:
            table = Table(
                title="Messages",
                show_header=True,
                header_style="bold magenta",
                show_lines=True,  # Add horizontal lines between rows
            )
            table.add_column("Role", style="cyan", no_wrap=True)
            table.add_column("Content", style="green")
            table.add_column("Details", style="yellow")
            table.add_column("Turn", style="yellow", no_wrap=True)

            # Build GSRT v2 markers if cached
            gsrt_markers = {}
            if simulation.reward_info and simulation.reward_info.info:
                info = simulation.reward_info.info
                if isinstance(info, dict) and ("gsrt_v2" in info or "gsrt_enhanced" in info):
                    try:
                        data = info.get("gsrt_v2") or info.get("gsrt_enhanced")
                        start_goal = data.get("start_goal") or {}
                        if isinstance(start_goal, dict) and isinstance(
                            start_goal.get("turn"), int
                        ):
                            gsrt_markers.setdefault(start_goal["turn"], []).append(
                                f"[bold yellow]START_GOAL[/]: {start_goal.get('goal')}"
                            )
                        for s in data.get("user_goal_shifts", []) or []:
                            if isinstance(s, dict) and isinstance(s.get("turn"), int):
                                label = f"[bold yellow]GOAL_SHIFT[/]: {s.get('from')} → {s.get('to')}"
                                if s.get("agent_responded") and isinstance(
                                    s.get("agent_turn"), int
                                ):
                                    label += f" (agent@{s.get('agent_turn')})"
                                gsrt_markers.setdefault(int(s["turn"]), []).append(
                                    label
                                )
                    except Exception:
                        pass

            current_turn = None
            for msg in simulation.messages:
                content = msg.content if msg.content is not None else ""
                details = ""

                # Set different colors based on message type
                if isinstance(msg, AssistantMessage):
                    role_style = "bold blue"
                    content_style = "blue"
                    tool_style = "bright_blue"  # Lighter shade of blue
                elif isinstance(msg, UserMessage):
                    role_style = "bold green"
                    content_style = "green"
                    tool_style = "bright_green"  # Lighter shade of green
                elif isinstance(msg, ToolMessage):
                    # For tool messages, use the color of the requestor's tool style
                    if msg.requestor == "user":
                        role_style = "bold green"
                        content_style = "bright_green"  # Match user's tool style
                    else:  # assistant
                        role_style = "bold blue"
                        content_style = "bright_blue"  # Match assistant's tool style
                else:  # SystemMessage
                    role_style = "bold magenta"
                    content_style = "magenta"

                if isinstance(msg, AssistantMessage) or isinstance(msg, UserMessage):
                    if msg.tool_calls:
                        tool_calls = []
                        for tool in msg.tool_calls:
                            tool_calls.append(
                                f"[{tool_style}]Tool: {tool.name}[/]\n[{tool_style}]Args: {json.dumps(tool.arguments, indent=2)}[/]"
                            )
                        details = "\n".join(tool_calls)
                elif isinstance(msg, ToolMessage):
                    details = f"[{content_style}]Tool ID: {msg.id}. Requestor: {msg.requestor}[/]"
                    if msg.error:
                        details += " [bold red](Error)[/]"

                # Append GSRT markers for this turn if any
                if isinstance(msg, UserMessage) and isinstance(msg.turn_idx, int):
                    for marker in gsrt_markers.get(msg.turn_idx, []) or []:
                        if details:
                            details += "\n"
                        details += marker

                # Add empty row between turns
                if current_turn is not None and msg.turn_idx != current_turn:
                    table.add_row("", "", "", "")
                current_turn = msg.turn_idx

                table.add_row(
                    f"[{role_style}]{msg.role}[/]",
                    f"[{content_style}]{content}[/]",
                    details,
                    str(msg.turn_idx) if msg.turn_idx is not None else "",
                )
            if show_details:
                cls.console.print(table)

    @classmethod
    def display_agent_metrics(cls, metrics: AgentMetrics):
        # Create content for metrics panel
        content = Text()

        # Add average reward section
        content.append("🏆 Average Reward: ", style="bold cyan")
        content.append(f"{metrics.avg_reward:.4f}\n\n")

        # Add Pass^k metrics section
        content.append("📈 Pass^k Metrics:", style="bold cyan")
        for k, pass_hat_k in metrics.pass_hat_ks.items():
            content.append(f"\nk={k}: ", style="bold white")
            content.append(f"{pass_hat_k:.3f}")

        # Add average agent cost section
        content.append("\n\n💰 Average Cost per Conversation: ", style="bold cyan")
        content.append(f"${metrics.avg_agent_cost:.4f}\n\n")

        # Add AgentChangeBench metrics section
        content.append("🎯 AgentChangeBench Metrics:", style="bold cyan")
        
        # TSR with detailed breakdown
        content.append(f"\n📊 TSR (Task Success Rate): ", style="bold white")
        content.append(f"{metrics.tsr:.2%}")
        if hasattr(metrics, "tsr_communicate_info") and metrics.tsr_communicate_info is not None:
            content.append(f"\n  💬 Communicate Info: {metrics.tsr_communicate_info:.2%}")
        if hasattr(metrics, "tsr_action") and metrics.tsr_action is not None:
            content.append(f"\n  🔧 Actions: {metrics.tsr_action:.2%}")
        if hasattr(metrics, "tsr_nl_assertion") and metrics.tsr_nl_assertion is not None:
            content.append(f"\n  📝 NL Assertions: {metrics.tsr_nl_assertion:.2%}")
        
        # TUE with component breakdown  
        content.append(f"\n⚙️  TUE (Tool Usage Efficiency): ", style="bold white")
        content.append(f"{metrics.tue:.2%}")
        if hasattr(metrics, "tue_tool_correctness") and metrics.tue_tool_correctness is not None:
            content.append(f"\n  ✅ Tool Correctness: {metrics.tue_tool_correctness:.2%}")
        if hasattr(metrics, "tue_param_accuracy") and metrics.tue_param_accuracy is not None:
            content.append(f"\n  🎯 Parameter Accuracy: {metrics.tue_param_accuracy:.2%}")
        
        # TCRR with redundancy breakdown
        content.append(f"\n🔄 TCRR (Tool-Call Redundancy Ratio): ", style="bold white")
        content.append(f"{metrics.tcrr:.2%}")
        if hasattr(metrics, "tcrr_redundant_calls") and hasattr(metrics, "tcrr_total_calls"):
            content.append(f"\n  📊 Redundant Calls: {metrics.tcrr_redundant_calls}/{metrics.tcrr_total_calls}")
        if hasattr(metrics, "tcrr_window_size"):
            content.append(f"\n  🪟 Window Size: {metrics.tcrr_window_size} turns")
            
        content.append(f"\n🛠️  Total Tool Calls: ", style="bold white")
        content.append(f"{metrics.num_tool_calls}")

        # Add GSRT metrics
        content.append(f"\n🔀 GSRT (Goal Shift Recovery Time): ", style="bold white")
        if hasattr(metrics, "gsrt_num_shifts") and metrics.gsrt_num_shifts > 0:
            content.append(f"\n  📊 Goal Shifts: {metrics.gsrt_num_shifts}")
            
            # New GSRT v2 metrics with multi-variant recovery times
            if hasattr(metrics, "gsrt_median_ack") and metrics.gsrt_median_ack is not None:
                content.append(f"\n  🎯 Acknowledgment: {metrics.gsrt_median_ack:.1f} turns (median)")
            if hasattr(metrics, "gsrt_median_tool") and metrics.gsrt_median_tool is not None:
                content.append(f"\n  🛠️  Tool Usage: {metrics.gsrt_median_tool:.1f} turns (median)")
            if hasattr(metrics, "gsrt_median_outcome") and metrics.gsrt_median_outcome is not None:
                content.append(f"\n  ✅ Outcome Success: {metrics.gsrt_median_outcome:.1f} turns (median)")
            
            # Recovery and transfer rates
            if hasattr(metrics, "gsrt_recovery_rate") and metrics.gsrt_recovery_rate is not None:
                content.append(f"\n  📈 Recovery Rate: {metrics.gsrt_recovery_rate:.1%}")
            if hasattr(metrics, "gsrt_transfer_rate") and metrics.gsrt_transfer_rate is not None:
                content.append(f"\n  🔄 Transfer Rate: {metrics.gsrt_transfer_rate:.1%}")
        else:
            content.append("\n  ❌ No goal shifts detected")

        # Add Coverage Statistics section
        content.append(f"\n\n📋 Coverage Statistics:", style="bold cyan")
        if hasattr(metrics, "tasks_with_communicate_info"):
            content.append(f"\n  💬 Tasks with Communicate Info: {metrics.tasks_with_communicate_info}")
        if hasattr(metrics, "tasks_with_actions"):
            content.append(f"\n  🔧 Tasks with Actions: {metrics.tasks_with_actions}")
        if hasattr(metrics, "tasks_with_nl_assertions"):
            content.append(f"\n  📝 Tasks with NL Assertions: {metrics.tasks_with_nl_assertions}")
        if hasattr(metrics, "tasks_with_goal_shifts"):
            content.append(f"\n  🔀 Tasks with Goal Shifts: {metrics.tasks_with_goal_shifts}")

        # Add Component Breakdown section
        content.append(f"\n\n🔍 Component Breakdown:", style="bold cyan")
        
        # Communicate Info Metrics
        if hasattr(metrics, "communicate_info_avg_score") and metrics.communicate_info_avg_score is not None:
            content.append(f"\n  💬 Communicate Info:")
            content.append(f"\n    📊 Average Score: {metrics.communicate_info_avg_score:.2%}")
            if hasattr(metrics, "communicate_info_exact_matches"):
                content.append(f"\n    ✅ Exact Matches: {metrics.communicate_info_exact_matches:.2%}")
            if hasattr(metrics, "total_communicate_info_checks"):
                content.append(f"\n    🔢 Total Checks: {metrics.total_communicate_info_checks}")

        # Action Metrics
        if hasattr(metrics, "action_avg_score") and metrics.action_avg_score is not None:
            content.append(f"\n  🔧 Action Metrics:")
            content.append(f"\n    📊 Average Score: {metrics.action_avg_score:.2%}")
            if hasattr(metrics, "action_tool_correctness"):
                content.append(f"\n    ✅ Tool Correctness: {metrics.action_tool_correctness:.2%}")
            if hasattr(metrics, "action_param_correctness"):
                content.append(f"\n    🎯 Parameter Correctness: {metrics.action_param_correctness:.2%}")
            if hasattr(metrics, "total_action_checks"):
                content.append(f"\n    🔢 Total Checks: {metrics.total_action_checks}")

        # NL Assertion Metrics
        if hasattr(metrics, "nl_assertion_avg_score") and metrics.nl_assertion_avg_score is not None:
            content.append(f"\n  📝 NL Assertion Metrics:")
            content.append(f"\n    📊 Average Score: {metrics.nl_assertion_avg_score:.2%}")
            if hasattr(metrics, "total_nl_assertions"):
                content.append(f"\n    🔢 Total Assertions: {metrics.total_nl_assertions}")

        # Partial Scoring Impact
        if hasattr(metrics, "tasks_benefiting_from_partial") and hasattr(metrics, "avg_reward_increase"):
            content.append(f"\n\n🎯 Partial Scoring Impact:", style="bold cyan")
            content.append(f"\n  📈 Tasks Benefiting: {metrics.tasks_benefiting_from_partial}")
            content.append(f"\n  ⬆️  Average Reward Increase: {metrics.avg_reward_increase:.2%}")

        # Create and display panel
        metrics_panel = Panel(
            content,
            title="[bold blue]Agent Metrics",
            border_style="blue",
            expand=True,
        )

        cls.console.print(metrics_panel)


class MarkdownDisplay:
    @classmethod
    def display_actions(cls, actions: List[Action]) -> str:
        """Display actions in markdown format."""
        return f"```json\n{json.dumps([action.model_dump() for action in actions], indent=2)}\n```"

    @classmethod
    def display_messages(cls, messages: list[Message]) -> str:
        """Display messages in markdown format."""
        return "\n\n".join(cls.display_message(msg) for msg in messages)

    @classmethod
    def display_simulation(cls, sim: SimulationRun) -> str:
        """Display simulation in markdown format."""
        # Otherwise handle SimulationRun object
        output = []

        # Add basic simulation info
        output.append(f"**Task ID**: {sim.task_id}")
        output.append(f"**Trial**: {sim.trial}")
        output.append(f"**Duration**: {sim.duration:.2f}s")
        output.append(f"**Termination**: {sim.termination_reason}")
        if sim.agent_cost is not None:
            output.append(f"**Agent Cost**: ${sim.agent_cost:.4f}")
        if sim.user_cost is not None:
            output.append(f"**User Cost**: ${sim.user_cost:.4f}")

        # Add reward info if present
        if sim.reward_info:
            breakdown = sorted(
                [
                    f"{k.value}: {v:.1f}"
                    for k, v in sim.reward_info.reward_breakdown.items()
                ]
            )
            output.append(
                f"**Reward**: {sim.reward_info.reward:.4f} ({', '.join(breakdown)})\n"
            )
            output.append(f"**Reward**: {sim.reward_info.reward:.4f}")

            # Add DB check info if present
            if sim.reward_info.db_check:
                output.append("\n**DB Check**")
                output.append(
                    f"- Status: {'✅' if sim.reward_info.db_check.db_match else '❌'} {sim.reward_info.db_check.db_reward}"
                )

            # Add env assertions if present
            if sim.reward_info.env_assertions:
                output.append("\n**Env Assertions**")
                for i, assertion in enumerate(sim.reward_info.env_assertions):
                    output.append(
                        f"- {i}: {assertion.env_assertion.env_type} {assertion.env_assertion.func_name} {'✅' if assertion.met else '❌'} {assertion.reward}"
                    )

            # Add action checks if present
            if sim.reward_info.action_checks:
                output.append("\n**Action Checks**")
                for i, check in enumerate(sim.reward_info.action_checks):
                    output.append(
                        f"- {i}: {check.action.name} {'✅' if check.action_match else '❌'} {check.action_reward}"
                    )

            # Add communication checks if present
            if sim.reward_info.communicate_checks:
                output.append("\n**Communicate Checks**")
                for i, check in enumerate(sim.reward_info.communicate_checks):
                    output.append(
                        f"- {i}: {check.info} {'✅' if check.met else '❌'} {check.justification}"
                    )

            # Add NL assertions if present
            if sim.reward_info.nl_assertions:
                output.append("\n**NL Assertions**")
                for i, assertion in enumerate(sim.reward_info.nl_assertions):
                    output.append(
                        f"- {i}: {assertion.nl_assertion} {'✅' if assertion.met else '❌'} {assertion.justification}"
                    )

            # Add additional info if present
            if sim.reward_info.info:
                output.append("\n**Additional Info**")
                for key, value in sim.reward_info.info.items():
                    output.append(f"- {key}: {value}")

        # Add messages using the display_message method
        if sim.messages:
            output.append("\n**Messages**:")
            output.extend(cls.display_message(msg) for msg in sim.messages)

        return "\n\n".join(output)

    @classmethod
    def display_result(
        cls,
        task: Task,
        sim: SimulationRun,
        reward: Optional[float] = None,
        show_task_id: bool = False,
    ) -> str:
        """Display a single result with all its components in markdown format."""
        output = [
            f"## Task {task.id}" if show_task_id else "## Task",
            "\n### User Instruction",
            task.user_scenario.instructions,
            "\n### Ground Truth Actions",
            cls.display_actions(task.evaluation_criteria.actions),
        ]

        if task.evaluation_criteria.communicate_info:
            output.extend(
                [
                    "\n### Communicate Info",
                    "```\n" + str(task.evaluation_criteria.communicate_info) + "\n```",
                ]
            )

        if reward is not None:
            output.extend(["\n### Reward", f"**{reward:.3f}**"])

        output.extend(["\n### Simulation", cls.display_simulation(sim)])

        return "\n".join(output)

    @classmethod
    def display_message(cls, msg: Message) -> str:
        """Display a single message in markdown format."""
        # Common message components
        parts = []

        # Add turn index if present
        turn_prefix = f"[TURN {msg.turn_idx}] " if msg.turn_idx is not None else ""

        # Format based on message type
        if isinstance(msg, AssistantMessage) or isinstance(msg, UserMessage):
            parts.append(f"{turn_prefix}**{msg.role}**:")
            if msg.content:
                parts.append(msg.content)
            if msg.tool_calls:
                tool_calls = []
                for tool in msg.tool_calls:
                    tool_calls.append(
                        f"**Tool Call**: {tool.name}\n```json\n{json.dumps(tool.arguments, indent=2)}\n```"
                    )
                parts.extend(tool_calls)

        elif isinstance(msg, ToolMessage):
            status = " (Error)" if msg.error else ""
            parts.append(f"{turn_prefix}**tool{status}**:")
            parts.append(f"Reponse to: {msg.requestor}")
            if msg.content:
                parts.append(f"```\n{msg.content}\n```")

        elif isinstance(msg, SystemMessage):
            parts.append(f"{turn_prefix}**system**:")
            if msg.content:
                parts.append(msg.content)

        return "\n".join(parts)
