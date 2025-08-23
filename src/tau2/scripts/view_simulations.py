#!/usr/bin/env python3
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.prompt import IntPrompt, Prompt
from rich.text import Text

from tau2.data_model.simulation import Results
from tau2.metrics.agent_metrics import compute_metrics, is_successful
from tau2.utils.display import ConsoleDisplay
from tau2.utils.utils import DATA_DIR
from tau2.utils.ias_manager import get_default_ias_manager
from tau2.utils.ias_interface import IASRatingInterface


def get_available_simulations():
    """Get list of available simulation result files."""
    sim_dir = Path(DATA_DIR) / "simulations"
    if not sim_dir.exists():
        return []

    return sorted([f for f in sim_dir.glob("*.json")])


def display_simulation_list(
    results: Results, only_show_failed: bool = False, only_show_all_failed: bool = False
):
    """Display a numbered list of simulations with basic info."""
    ConsoleDisplay.console.print("\n[bold blue]Available Simulations:[/]")

    # calculate number of successful and total trials for each task
    num_success = defaultdict(int)
    for sim in results.simulations:
        if is_successful(sim.reward_info.reward):
            num_success[sim.task_id] += 1

    for i, sim in enumerate(results.simulations, 1):
        reward = sim.reward_info.reward if sim.reward_info else None

        # filter out simulations based on the flags
        if only_show_failed:
            if is_successful(reward):
                continue
        if only_show_all_failed:
            if num_success[sim.task_id] > 0:
                continue

        reward_str = "✅" if is_successful(reward) else "❌"
        db_match = "N/A"
        if sim.reward_info and sim.reward_info.db_check:
            db_match = "YES" if sim.reward_info.db_check.db_match else "NO"

        # Create text with task ID
        task_text = Text()
        task_text.append(f"{i}.", style="cyan")
        task_text.append(" Task: ")
        task_text.append(sim.task_id)  # This will display square brackets correctly
        task_text.append(
            f" | Trial: {sim.trial} | Reward: {reward_str} | Duration: {sim.duration:.2f}s | DB Match: {db_match} | "
        )

        ConsoleDisplay.console.print(task_text)

    if only_show_all_failed:
        num_all_failed = len([1 for v in num_success.values() if v == 0])
        ConsoleDisplay.console.print(f"Total number of failed trials: {num_all_failed}")


def display_available_files(files):
    """Display a numbered list of available simulation files."""
    ConsoleDisplay.console.print("\n[bold blue]Available Simulation Files:[/]")
    for i, file in enumerate(files, 1):
        ConsoleDisplay.console.print(f"[cyan]{i}.[/] {file.name}")


def display_simulation_with_task(
    simulation, task, results_file: str, sim_index: int, show_details: bool = True
):
    """Display a simulation along with its associated task."""
    ConsoleDisplay.console.print("\n" + "=" * 80)  # Separator
    ConsoleDisplay.console.print("[bold blue]Task Details:[/]")
    ConsoleDisplay.display_task(task)

    ConsoleDisplay.console.print("\n" + "=" * 80)  # Separator
    ConsoleDisplay.console.print("[bold blue]Simulation Details:[/]")
    ConsoleDisplay.display_simulation(simulation, show_details=show_details, task=task)

    # Prompt for notes
    ConsoleDisplay.console.print("\n" + "=" * 80)  # Separator
    ConsoleDisplay.console.print("[bold blue]Add Notes:[/]")
    note = Prompt.ask("Enter your notes about this simulation (press Enter to skip)")

    if note.strip():
        save_simulation_note(simulation, task, note, results_file, sim_index)
        ConsoleDisplay.console.print("[green]Note saved successfully![/]")


def parse_key(key: str) -> tuple[str, int]:
    """Parse a key into a task ID and trial number."""
    task_id, trial = key.split("-")
    return task_id, int(trial)


def find_task_by_id(tasks, task_id):
    """Find a task in the task list by its ID."""
    for task in tasks:
        if task.id == task_id:
            return task
    return None


def find_simulation_by_task_id_and_trial(results, task_id, trial):
    """Get a simulation by its task ID and trial number."""
    return next(
        (
            sim
            for sim in results.simulations
            if sim.task_id == task_id and sim.trial == trial
        ),
        None,
    )


def save_simulation_note(
    simulation, task, note: str, results_file: str, sim_index: int
):
    """Save a note about a simulation to a CSV file."""
    notes_file = Path(DATA_DIR) / "simulations" / "simulation_notes.csv"
    file_exists = notes_file.exists()

    # Prepare the row data
    row = {
        "timestamp": datetime.now().isoformat(),
        "simulation_id": simulation.id,
        "task_id": simulation.task_id,
        "trial": simulation.trial,
        "duration": simulation.duration,
        "reward": simulation.reward_info.reward if simulation.reward_info else None,
        "db_match": simulation.reward_info.db_check.db_match
        if simulation.reward_info and simulation.reward_info.db_check
        else None,
        "results_file": results_file,
        "sim_index": sim_index,
        "note": note,
    }

    # Write to CSV file
    with open(notes_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def handle_ias_rating(results: Results):
    """Handle IAS rating workflow."""
    ias_manager = get_default_ias_manager()
    ias_interface = IASRatingInterface(ias_manager)

    # Get rater name first
    rater_name = Prompt.ask("\n[bold cyan]Enter your name/ID for rating[/]")
    if not rater_name.strip():
        ConsoleDisplay.console.print("[red]Rater name cannot be empty[/]")
        return

    # Show list of simulations with IAS status
    ConsoleDisplay.console.print("\n[bold blue]Simulations Available for Rating:[/]")

    for i, sim in enumerate(results.simulations, 1):
        # Get IAS rating status
        summary = ias_manager.get_rating_summary(sim.id)
        status_text = Text()
        status_text.append(f"{i}.", style="cyan")
        status_text.append(f" Task: {sim.task_id}, Trial: {sim.trial}")

        if summary["num_ratings"] == 0:
            status_text.append(" [red](NOT RATED)[/]", style="red")
        else:
            status_text.append(
                f" [green](Rated by {summary['num_ratings']} raters)[/]", style="green"
            )
            if rater_name in summary["raters"]:
                status_text.append(
                    " [yellow](You already rated this)[/]", style="yellow"
                )

        ConsoleDisplay.console.print(status_text)

    # Get simulation selection
    sim_count = len(results.simulations)
    sim_index = IntPrompt.ask(f"\nSelect simulation to rate (1-{sim_count})", default=1)

    if not (1 <= sim_index <= sim_count):
        ConsoleDisplay.console.print("[red]Invalid simulation number[/]")
        return

    sim = results.simulations[sim_index - 1]
    task = find_task_by_id(results.tasks, sim.task_id)

    if not task:
        ConsoleDisplay.console.print(
            f"[red]Could not find task for simulation {sim.id}[/]"
        )
        return

    # Conduct the rating
    rating = ias_interface.conduct_rating(sim, task, rater_name)

    if rating:
        try:
            ias_manager.add_rating(rating)
            ConsoleDisplay.console.print(
                "\n[bold green]✅ Rating saved successfully![/]"
            )
        except Exception as e:
            ConsoleDisplay.console.print(f"\n[red]❌ Error saving rating: {e}[/]")
    else:
        ConsoleDisplay.console.print("\n[yellow]Rating cancelled[/]")


def handle_view_ias_ratings(results: Results):
    """Handle viewing IAS ratings."""
    ias_manager = get_default_ias_manager()
    ias_interface = IASRatingInterface(ias_manager)

    # Show overview of all ratings
    ConsoleDisplay.console.print("\n[bold blue]IAS Ratings Overview:[/]")

    rated_sims = 0
    total_ratings = 0

    for i, sim in enumerate(results.simulations, 1):
        summary = ias_manager.get_rating_summary(sim.id)

        if summary["num_ratings"] > 0:
            rated_sims += 1
            total_ratings += summary["num_ratings"]

            status_text = Text()
            status_text.append(f"{i}.", style="cyan")
            status_text.append(f" Task: {sim.task_id}, Trial: {sim.trial}")
            status_text.append(f" - {summary['num_ratings']} ratings")
            if summary["average_score"]:
                status_text.append(f" (avg: {summary['average_score']:.2f}/5.0)")

            ConsoleDisplay.console.print(status_text)

    if rated_sims == 0:
        ConsoleDisplay.console.print("[yellow]No IAS ratings found[/]")
        return

    ConsoleDisplay.console.print(
        f"\n[bold white]Summary: {rated_sims}/{len(results.simulations)} simulations rated, {total_ratings} total ratings[/]"
    )

    # Ask if user wants to see details for a specific simulation
    if (
        Prompt.ask(
            "\nView detailed ratings for a specific simulation? (y/n)", default="n"
        ).lower()
        == "y"
    ):
        sim_index = IntPrompt.ask(
            f"Enter simulation number (1-{len(results.simulations)})", default=1
        )

        if 1 <= sim_index <= len(results.simulations):
            sim = results.simulations[sim_index - 1]
            ias_interface.show_rating_summary(sim.id)

            # Show individual ratings
            ratings = ias_manager.get_ratings_for_simulation(sim.id)
            for rating in ratings:
                ConsoleDisplay.console.print(
                    f"\n[bold cyan]Rating by {rating.rater_name}:[/]"
                )
                ConsoleDisplay.console.print(
                    f"Overall Score: {rating.overall_score:.2f}/5.0"
                )
                ConsoleDisplay.console.print(f"Tone Match: {rating.tone_match.score}/5")
                ConsoleDisplay.console.print(f"Clarity: {rating.clarity.score}/5")
                ConsoleDisplay.console.print(f"Pacing: {rating.pacing.score}/5")
                ConsoleDisplay.console.print(f"Adaptivity: {rating.adaptivity.score}/5")
                if rating.overall_notes:
                    ConsoleDisplay.console.print(f"Notes: {rating.overall_notes}")


def handle_agent_metrics_with_ias(results: Results):
    """Handle displaying agent metrics with IAS options."""
    ConsoleDisplay.console.clear()
    metrics = compute_metrics(results)
    ConsoleDisplay.display_agent_metrics(metrics)

    # Add IAS options
    ConsoleDisplay.console.print("\n[bold blue]IAS Options:[/]")
    ConsoleDisplay.console.print("1. Rate a specific simulation")
    ConsoleDisplay.console.print("2. View all IAS ratings")
    ConsoleDisplay.console.print("3. Back to main menu")

    choice = Prompt.ask(
        "\nWhat would you like to do with IAS ratings?",
        choices=["1", "2", "3"],
        default="3",
    )

    if choice == "1":
        handle_ias_rating(results)
    elif choice == "2":
        handle_view_ias_ratings(results)
    # No else: Back to main menu


def main(
    sim_file: Optional[str] = None,
    only_show_failed: bool = False,
    only_show_all_failed: bool = False,
):
    # Get available simulation files
    if sim_file is None:
        sim_files = get_available_simulations()
    else:
        sim_files = [Path(sim_file)]

    if not sim_files:
        ConsoleDisplay.console.print(
            "[red]No simulation files found in data/simulations/[/]"
        )
        return

    results = None
    current_file = None
    while True:
        # Show main menu
        ConsoleDisplay.console.print("\n[bold yellow]Main Menu:[/]")
        ConsoleDisplay.console.print("1. Select simulation file")
        ConsoleDisplay.console.print(
            "   [dim]Choose a simulation results file to load and analyze[/]"
        )
        if results:
            ConsoleDisplay.console.print("2. View agent performance metrics")
            ConsoleDisplay.console.print(
                "   [dim]Display agent performance metrics and IAS ratings[/]"
            )
            ConsoleDisplay.console.print("3. View simulation")
            ConsoleDisplay.console.print(
                "   [dim]Examine a specific simulation in detail with all its data[/]"
            )
            ConsoleDisplay.console.print("4. View task details")
            ConsoleDisplay.console.print(
                "   [dim]Look at the configuration and parameters of a specific task[/]"
            )
            ConsoleDisplay.console.print("5. Exit")
            ConsoleDisplay.console.print("   [dim]Close the simulation viewer[/]")
            choices = ["1", "2", "3", "4", "5"]
            default_choice = "3"
        else:
            ConsoleDisplay.console.print("2. Exit")
            ConsoleDisplay.console.print("   [dim]Close the simulation viewer[/]")
            choices = ["1", "2"]
            default_choice = "1"

        choice = Prompt.ask(
            "\nWhat would you like to do?", choices=choices, default=default_choice
        )

        if choice == "1":
            # Show available files and get selection
            display_available_files(sim_files)
            # default to view the last file
            file_num = IntPrompt.ask(
                f"\nSelect file number (1-{len(sim_files)})", default=len(sim_files)
            )

            if 1 <= file_num <= len(sim_files):
                try:
                    current_file = sim_files[file_num - 1].name
                    results = Results.load(sim_files[file_num - 1])
                    ConsoleDisplay.console.print(
                        f"\n[bold green]Loaded {len(results.simulations)} simulations from {current_file}[/]"
                    )
                    results.simulations = sorted(
                        results.simulations, key=lambda x: (x.task_id, x.trial)
                    )
                except Exception as e:
                    ConsoleDisplay.console.print(
                        f"[red]Error loading results:[/] {str(e)}"
                    )
            else:
                ConsoleDisplay.console.print("[red]Invalid file number[/]")

        elif choice == "2" and not results:
            break

        elif results and choice == "2":
            # Display metrics with IAS options
            handle_agent_metrics_with_ias(results)
            continue

        elif results and choice == "3":
            # Show list of simulations
            display_simulation_list(results, only_show_failed, only_show_all_failed)

            # Get simulation selection by index
            sim_count = len(results.simulations)
            sim_index = IntPrompt.ask(
                f"\nEnter simulation number (1-{sim_count})", default=1
            )

            if 1 <= sim_index <= sim_count:
                sim = results.simulations[sim_index - 1]
                task = find_task_by_id(results.tasks, sim.task_id)
                if task:
                    display_simulation_with_task(
                        sim, task, current_file, sim_index, show_details=True
                    )
                else:
                    ConsoleDisplay.console.print(
                        f"[red]Warning: Could not find task for simulation {sim.id}[/]"
                    )
                    ConsoleDisplay.display_simulation(sim, show_details=True)
                continue
            else:
                ConsoleDisplay.console.print("[red]Invalid simulation number[/]")
                continue

        elif results and choice == "4":
            # Show list of tasks
            ConsoleDisplay.console.print("\n[bold blue]Available Tasks:[/]")
            for i, task in enumerate(results.tasks, 1):
                task_text = Text()
                task_text.append(f"{i}.", style="cyan")
                task_text.append(" Task ID: ")
                task_text.append(task.id)  # This will display square brackets correctly
                ConsoleDisplay.console.print(task_text)

            # Get task selection
            task_count = len(results.tasks)
            task_num = IntPrompt.ask(f"\nEnter task number (1-{task_count})", default=1)

            if 1 <= task_num <= task_count:
                ConsoleDisplay.console.clear()
                ConsoleDisplay.display_task(results.tasks[task_num - 1])
                continue
            else:
                ConsoleDisplay.console.print("[red]Invalid task number[/]")
                continue

        else:  # Exit options (choice == "5" when results exist, choice == "2" when no results)
            break

    ConsoleDisplay.console.print("\n[green]Thanks for using the simulation viewer![/]")


if __name__ == "__main__":
    main()
