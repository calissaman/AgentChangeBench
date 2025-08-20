from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt, Confirm
from rich.table import Table
from rich.text import Text
from typing import Optional

from tau2.data_model.ias_rating import (
    IASRating,
    IASCriteriaScore,
    IASCriteria,
    IAS_CRITERIA_DESCRIPTIONS,
)
from tau2.data_model.simulation import SimulationRun
from tau2.data_model.tasks import Task
from tau2.utils.ias_manager import IASManager
from tau2.utils.display import ConsoleDisplay


class IASRatingInterface:
    """Interactive interface for rating simulations with IAS criteria."""

    def __init__(self, ias_manager: IASManager):
        self.ias_manager = ias_manager
        self.console = Console()

    def show_rating_summary(self, simulation_id: str):
        """Display existing ratings summary for a simulation."""
        summary = self.ias_manager.get_rating_summary(simulation_id)

        if summary["num_ratings"] == 0:
            self.console.print(
                f"\n[yellow]No IAS ratings found for simulation {simulation_id}[/]"
            )
            return

        # Create summary panel
        content = Text()
        content.append("📊 Rating Summary\n\n", style="bold cyan")
        content.append(f"Number of ratings: ", style="bold white")
        content.append(f"{summary['num_ratings']}\n")

        if summary["average_score"]:
            content.append(f"Average score: ", style="bold white")
            content.append(f"{summary['average_score']:.2f}/5.0\n")

        content.append(f"Raters: ", style="bold white")
        content.append(f"{', '.join(summary['raters'])}\n")

        if summary["reliability"] is not None and summary["num_ratings"] > 1:
            content.append(f"Inter-rater reliability (std dev): ", style="bold white")
            content.append(f"{summary['reliability']:.2f}")

        panel = Panel(
            content, title="[bold blue]IAS Rating Summary", border_style="blue"
        )
        self.console.print(panel)

    def display_criteria_guide(self):
        """Display the IAS criteria and scoring guide."""
        table = Table(title="IAS Rating Criteria (1-5 Scale)")
        table.add_column("Criteria", style="cyan", no_wrap=True)
        table.add_column("Description", style="white")
        table.add_column("Examples", style="yellow")

        for criteria, info in IAS_CRITERIA_DESCRIPTIONS.items():
            examples = "\n".join(
                [f"{score}: {desc}" for score, desc in info["examples"].items()]
            )
            table.add_row(info["name"], info["description"], examples)

        self.console.print(table)
        self.console.print(
            "\n[dim]Score 1 = Poor, 2 = Below Average, 3 = Average, 4 = Good, 5 = Excellent[/]"
        )

    def display_conversation_for_rating(self, simulation: SimulationRun, task: Task):
        """Display the conversation in a format suitable for IAS rating."""
        self.console.print("\n[bold blue]Conversation to Rate:[/]")

        if task.user_scenario.persona:
            persona_panel = Panel(
                f"[bold]Persona:[/] {task.user_scenario.persona}",
                title="User Persona",
                border_style="green",
            )
            self.console.print(persona_panel)

        instructions_text = ""
        if hasattr(task.user_scenario.instructions, "task_instructions"):
            instructions_text = task.user_scenario.instructions.task_instructions
        else:
            instructions_text = str(task.user_scenario.instructions)

        instructions_panel = Panel(
            instructions_text, title="Task Instructions", border_style="blue"
        )
        self.console.print(instructions_panel)

        ConsoleDisplay.display_simulation(simulation, show_details=True)

    def rate_single_criteria(self, criteria: IASCriteria) -> IASCriteriaScore:
        """Rate a single IAS criteria interactively."""
        info = IAS_CRITERIA_DESCRIPTIONS[criteria]

        self.console.print(f"\n[bold cyan]Rating: {info['name']}[/]")
        self.console.print(f"[white]{info['description']}[/]\n")

        for score, example in info["examples"].items():
            self.console.print(f"[yellow]{score}:[/] {example}")

        while True:
            score = IntPrompt.ask(f"\nEnter score for {info['name']} (1-5)", default=3)
            if 1 <= score <= 5:
                break
            self.console.print("[red]Score must be between 1 and 5[/]")

        notes = Prompt.ask(
            f"Optional notes for {info['name']} (press Enter to skip)", default=""
        )

        return IASCriteriaScore(
            criteria=criteria, score=score, notes=notes if notes else None
        )

    def conduct_rating(
        self, simulation: SimulationRun, task: Task, rater_name: str
    ) -> Optional[IASRating]:
        """Conduct a complete IAS rating session."""
        self.console.clear()

        if self.ias_manager.has_rating_by_rater(simulation.id, rater_name):
            self.console.print(
                f"[yellow]You have already rated simulation {simulation.id}[/]"
            )
            overwrite = Confirm.ask("Do you want to overwrite your previous rating?")
            if not overwrite:
                return None

        self.display_conversation_for_rating(simulation, task)

        self.console.print("\n" + "=" * 80)
        self.display_criteria_guide()

        if not Confirm.ask("\n[bold]Ready to start rating?[/]"):
            return None

        criteria_scores = {}
        for criteria in [
            IASCriteria.TONE_MATCH,
            IASCriteria.CLARITY,
            IASCriteria.PACING,
            IASCriteria.ADAPTIVITY,
        ]:
            criteria_scores[criteria.value] = self.rate_single_criteria(criteria)

        overall_notes = Prompt.ask(
            "\nAny overall notes about this conversation? (press Enter to skip)",
            default="",
        )

        rating = IASRating(
            simulation_id=simulation.id,
            rater_name=rater_name,
            tone_match=criteria_scores["tone_match"],
            clarity=criteria_scores["clarity"],
            pacing=criteria_scores["pacing"],
            adaptivity=criteria_scores["adaptivity"],
            overall_notes=overall_notes if overall_notes else None,
        )

        # Show rating summary and confirm
        self.show_rating_preview(rating)

        if Confirm.ask("\n[bold]Save this rating?[/]"):
            return rating
        else:
            return None

    def show_rating_preview(self, rating: IASRating):
        """Show a preview of the rating before saving."""
        content = Text()
        content.append("📝 Rating Preview\n\n", style="bold cyan")
        content.append(f"Rater: ", style="bold white")
        content.append(f"{rating.rater_name}\n")
        content.append(f"Overall Score: ", style="bold white")
        content.append(f"{rating.overall_score:.2f}/5.0\n\n")

        content.append("Individual Scores:\n", style="bold white")
        content.append(f"• Tone Match: {rating.tone_match.score}/5")
        if rating.tone_match.notes:
            content.append(f" ({rating.tone_match.notes})")
        content.append("\n")

        content.append(f"• Clarity: {rating.clarity.score}/5")
        if rating.clarity.notes:
            content.append(f" ({rating.clarity.notes})")
        content.append("\n")

        content.append(f"• Pacing: {rating.pacing.score}/5")
        if rating.pacing.notes:
            content.append(f" ({rating.pacing.notes})")
        content.append("\n")

        content.append(f"• Adaptivity: {rating.adaptivity.score}/5")
        if rating.adaptivity.notes:
            content.append(f" ({rating.adaptivity.notes})")
        content.append("\n")

        if rating.overall_notes:
            content.append(f"\nOverall Notes: {rating.overall_notes}")

        panel = Panel(content, title="[bold green]Rating Preview", border_style="green")
        self.console.print(panel)
