"""
TEI Evaluation Framework - Comparison Tool.

This module provides visualization comparisons for TEI evaluation results:

1. Internal run comparison: Compare multiple processing runs for a single model
   - Visualizes performance across different runs (e.g., different prompts, temperatures)
   - Output: results/model/{model}/processing_comparison/processing_comparison.png

2. Cross-model comparison: Compare multiple models (one run per model)
   - Visualizes performance across different LLM models
   - Output: results/cross_model/cross_model_comp_{models}.png

The tool supports both interactive and command-line interfaces for flexible usage.
"""

import argparse
import json
import logging
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import get_path_config
from tei_evaluator.reporting.visualization import (
    create_cross_model_comparison,
    create_processing_comparison,
)
from tei_evaluator.utils.logging_config import setup_logger

__version__ = "1.0.0"


# -------------------------
# Discovery utilities
# -------------------------


def _results_root() -> Path:
    """
    Get the root directory for model results.

    Returns:
        Path to results/model directory.
    """
    return Path("results/model")


def discover_models() -> List[str]:
    """
    Discover all available models in the results directory.

    Returns:
        Sorted list of model names.
    """
    logger = logging.getLogger(__name__)
    base = _results_root()
    if not base.exists():
        logger.warning("Results directory not found: %s", base)
        return []
    models = sorted([d.name for d in base.iterdir() if d.is_dir()])
    logger.info("Discovered %d model(s)", len(models))
    return models


def discover_runs_for_model(model: str) -> List[Path]:
    """
    Discover all processing runs for a given model.

    Args:
        model: Model name to search for.

    Returns:
        Sorted list of paths to processing run directories containing unified summaries.
    """
    logger = logging.getLogger(__name__)
    base = _results_root() / model
    if not base.exists():
        logger.warning("Model directory not found: %s", base)
        return []
    runs = []
    for proc in base.glob("processing_*"):
        # unified summary may be directly inside the processing_* directory
        if (proc / "unified_evaluation_summary.json").exists():
            runs.append(proc)
            continue
        # or inside a json subfolder
        json_dir = proc / "json"
        if (json_dir / "unified_evaluation_summary.json").exists():
            runs.append(json_dir)
    logger.info("Found %d processing run(s) for model '%s'", len(runs), model)
    return sorted(runs)


def load_unified_summary(unified_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Load unified evaluation summary from a directory.

    Args:
        unified_dir: Path to directory containing unified_evaluation_summary.json.

    Returns:
        Dictionary containing summary data, or None if loading fails.
    """
    logger = logging.getLogger(__name__)
    file = unified_dir / "unified_evaluation_summary.json"
    if not file.exists():
        logger.warning("Unified summary not found: %s", file)
        return None
    try:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.debug("Loaded unified summary from %s", file)
        return data
    except (IOError, OSError, json.JSONDecodeError) as e:
        logger.error("Failed to load unified summary from %s: %s", file, str(e))
        return None


def extract_file_results(unified_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract file results from unified evaluation data.

    Args:
        unified_data: Unified evaluation summary dictionary.

    Returns:
        List of per-file result dictionaries.
    """
    if not unified_data:
        return []
    return unified_data.get("files", []) or []


def get_schema_mode_from_files(file_results: List[Dict[str, Any]]) -> str:
    """
    Determine schema validation mode from file results.

    Args:
        file_results: List of per-file results.

    Returns:
        Schema mode string ('none', 'tei', 'project', or 'both').
    """
    if not file_results:
        return "both"
    return file_results[0].get("schema_mode", "both")


_PROMPTS_CACHE: Optional[Dict[str, Any]] = None


def _load_prompts_config() -> Dict[str, Any]:
    """
    Load prompts.json once (cached) to enrich legend labels with prompt details.

    The prompts.json file contains prompt configuration details:
    {
      "prompt_v1": {"encoding_rules": "detailed", "few_shot_examples": 0},
      ...
    }

    Returns:
        Dictionary mapping prompt versions to their configurations.
    """
    global _PROMPTS_CACHE  # noqa: PLW0603
    logger = logging.getLogger(__name__)

    if _PROMPTS_CACHE is not None:
        return _PROMPTS_CACHE

    prompts_path = Path("data/output/prompts.json")
    try:
        with prompts_path.open("r", encoding="utf-8") as f:
            _PROMPTS_CACHE = json.load(f)
        logger.debug("Loaded prompts configuration from %s", prompts_path)
    except (IOError, OSError, json.JSONDecodeError) as e:
        logger.warning("Failed to load prompts configuration: %s", str(e))
        _PROMPTS_CACHE = {}

    return _PROMPTS_CACHE


def load_processing_metadata_label(
    unified_dir: Path, fallback_label: str, model_name: Optional[str] = None
) -> str:
    """
    Load processing metadata and create a formatted label for visualization legends.

    Args:
        unified_dir: Path to unified results directory.
        fallback_label: Label to use if metadata cannot be loaded.
        model_name: Optional model name to include in label.

    Returns:
        Formatted metadata label string.
    """
    logger = logging.getLogger(__name__)

    # Resolve processing_* directory whether unified_dir is processing_* or processing_*/json
    processing_dir = (
        unified_dir
        if unified_dir.name.startswith("processing_")
        else unified_dir.parent
    )
    metadata_file = processing_dir / "processing_metadata.json"

    if not metadata_file.exists():
        logger.debug("Metadata file not found: %s", metadata_file)
        return fallback_label

    try:
        with open(metadata_file, "r", encoding="utf-8") as f:
            md = json.load(f)

        ts = md.get("timestamp", "N/A")
        temp = md.get("temperature", "N/A")
        prompt_v = md.get("prompt_version", "N/A")
        thinking = md.get("thinking_mode", "None")
        ptime = md.get("processing_time", "N/A")
        cost_usd = md.get("estimated_cost_usd", None)

        # Enrich prompt version with details from prompts.json
        prompts_cfg = _load_prompts_config()
        prompt_cfg = (
            prompts_cfg.get(prompt_v, {}) if isinstance(prompts_cfg, dict) else {}
        )
        encoding_rules = prompt_cfg.get("encoding_rules")
        few_shot = prompt_cfg.get("few_shot_examples")

        # Human-friendly prompt description, e.g. "Prompt: detailed + 2-shot"
        if encoding_rules is not None and few_shot is not None:
            prompt_summary = f"Prompt: {encoding_rules} + {few_shot}-shot"
        elif encoding_rules is not None:
            prompt_summary = f"Prompt: {encoding_rules}"
        else:
            prompt_summary = "Prompt: N/A"

        # Legend label: start with model name (if provided) or processing run, then metadata
        # Use fixed-width keys for vertical alignment across legend entries
        parts = []
        if model_name:
            parts.append(model_name)
        else:
            parts.append(f"{ts}")  # Use timestamp (processing run name)

        parts.append(f"{'Temp:':<8}{temp}")
        parts.append(f"{'Processing Time:':<20}{ptime}")

        if isinstance(cost_usd, (int, float)):
            parts.append(f"{'API-Costs:':<11}~{cost_usd:.2f} USD")

        # Thinking before Prompt
        if isinstance(thinking, str):
            show_thinking = thinking.lower() != "none"
        else:
            show_thinking = thinking is not None

        if show_thinking:
            parts.append(f"{'Thinking:':<12}{thinking}")

        parts.append(f"{'Prompt:':<9}{prompt_summary.replace('Prompt: ', '')}")

        label = " | ".join(parts)
        logger.debug("Created metadata label for %s", processing_dir.name)
        return label

    except (IOError, OSError, json.JSONDecodeError, KeyError) as e:
        logger.warning("Failed to load metadata from %s: %s", metadata_file, str(e))
        return fallback_label


# -------------------------
# Interactive Prompts
# -------------------------


def prompt_menu() -> str:
    """
    Display main menu and get user choice.

    Returns:
        User's menu choice as string ('1', '2', or '3').
    """
    print("\n" + "=" * 70)
    print("TEI LLM Evaluation - Unified Comparison")
    print("=" * 70)
    print("Choose an option:")
    print("1. Internal run comparison (one model, multiple processing runs)")
    print("2. Cross-model comparison (multiple models, one run each)")
    print("3. Exit")
    print("-" * 70)
    while True:
        choice = input("Enter choice (1-3): ").strip()
        if choice in ["1", "2", "3"]:
            return choice
        print("[ERROR] Invalid choice. Please enter 1, 2, or 3.")


def prompt_select_model(
    models: List[str], prompt_text: str = "Select model"
) -> Optional[str]:
    """
    Prompt user to select a single model from available models.

    Args:
        models: List of available model names.
        prompt_text: Custom prompt text to display.

    Returns:
        Selected model name, or None if user quits.
    """
    if not models:
        print("\n[ERROR] No models found in results/model/")
        return None
    print("\nAvailable models:\n")
    for i, m in enumerate(models, 1):
        print(f"{i:2d}. {m}")
    while True:
        val = input(f"\n{prompt_text} (1-{len(models)}) or 'q' to quit: ").strip()
        if val.lower() == "q":
            return None
        try:
            idx = int(val)
            if 1 <= idx <= len(models):
                return models[idx - 1]
        except ValueError:
            pass
        print(f"[ERROR] Please enter a number between 1 and {len(models)}, or 'q'.")


def prompt_select_multiple_models(models: List[str]) -> List[str]:
    """
    Prompt user to select multiple models from available models.

    Args:
        models: List of available model names.

    Returns:
        List of selected model names, or empty list if user quits.
    """
    if not models:
        print("\n[ERROR] No models found in results/model/")
        return []
    print("\nAvailable models:\n")
    for i, m in enumerate(models, 1):
        print(f"{i:2d}. {m}")
    print("\nEnter comma-separated indices (e.g., 1,3,4) or 'q' to quit.")
    while True:
        val = input("Select models: ").strip()
        if val.lower() == "q":
            return []
        parts = [p.strip() for p in val.split(",") if p.strip()]
        try:
            indices = sorted(set(int(p) for p in parts))
            selected = [models[i - 1] for i in indices if 1 <= i <= len(models)]
            if selected:
                return selected
        except ValueError:
            pass
        print("[ERROR] Please enter valid indices like 1,2 or 'q'.")


def prompt_select_run(unified_dirs: List[Path], title: str) -> Optional[Path]:
    """
    Prompt user to select a processing run from available runs.

    Args:
        unified_dirs: List of paths to unified result directories.
        title: Title to display above the selection menu.

    Returns:
        Selected directory path, or None if user quits.
    """
    if not unified_dirs:
        print("\n[ERROR] No processing runs found.")
        return None
    print(f"\n{title}")
    for i, u in enumerate(unified_dirs, 1):
        run_name = u.name if u.name.startswith("processing_") else u.parent.name
        ts = run_name.replace("processing_", "")
        print(f"{i:2d}. {run_name} ({ts})")
    while True:
        val = input(f"Select run (1-{len(unified_dirs)}) or 'q' to quit: ").strip()
        if val.lower() == "q":
            return None
        try:
            idx = int(val)
            if 1 <= idx <= len(unified_dirs):
                return unified_dirs[idx - 1]
        except ValueError:
            pass
        print(
            f"[ERROR] Please enter a number between 1 and {len(unified_dirs)}, or 'q'."
        )


# -------------------------
# Workflow Functions
# -------------------------


def run_internal_comparison() -> None:
    """
    Run internal comparison workflow: compare multiple processing runs for a single model.

    This function guides the user through selecting a model and processing runs,
    then generates a comparison visualization.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting internal run comparison workflow")

    models = discover_models()
    model = prompt_select_model(models, "Select model for internal comparison")
    if not model:
        logger.info("User cancelled model selection")
        return

    runs = discover_runs_for_model(model)
    if not runs:
        logger.error("No processing runs with unified summaries found for %s", model)
        print(f"\n[ERROR] No processing_* runs with unified summaries found for {model}")
        return

    # Let user choose multiple runs or all
    print("\nAvailable runs:")
    for i, u in enumerate(runs, 1):
        run_name = u.name if u.name.startswith("processing_") else u.parent.name
        print(f"{i:2d}. {run_name}")
    print("\nEnter comma-separated run indices, 'a' for all, or 'q' to quit.")
    while True:
        val = input("Select runs: ").strip().lower()
        if val == "q":
            logger.info("User cancelled run selection")
            return
        if val == "a":
            selected_runs = runs
            logger.info("User selected all %d runs", len(runs))
            break
        try:
            indices = sorted(
                set(int(p.strip()) for p in val.split(",") if p.strip())
            )
            selected_runs = [runs[i - 1] for i in indices if 1 <= i <= len(runs)]
            if selected_runs:
                logger.info("User selected %d run(s)", len(selected_runs))
                break
        except ValueError:
            pass
        print("[ERROR] Please enter valid indices like 1,2 or 'a' or 'q'.")

    # Load data
    processing_data: Dict[str, List[Dict[str, Any]]] = {}
    metadata_labels: Dict[str, str] = {}
    for u in selected_runs:
        run_name = u.name if u.name.startswith("processing_") else u.parent.name
        data = load_unified_summary(u)
        if not data:
            logger.warning("Skipping %s: cannot load summary", run_name)
            print(f"[WARNING] Skipping {run_name}: cannot load summary")
            continue
        files = extract_file_results(data)
        if not files:
            logger.warning("Skipping %s: no file results", run_name)
            print(f"[WARNING] Skipping {run_name}: no file results")
            continue
        processing_data[run_name] = files
        # For internal comparison, use processing run name (timestamp) instead of model name
        metadata_labels[run_name] = load_processing_metadata_label(
            u, run_name, model_name=None
        )

    if not processing_data:
        logger.error("No valid processing data loaded")
        print("\n[ERROR] No valid processing data loaded.")
        return

    # Derive schema mode from first loaded run
    first_key = next(iter(processing_data.keys()))
    schema_mode = get_schema_mode_from_files(processing_data[first_key])
    logger.info("Using schema mode: %s", schema_mode)

    # Output dir
    out_dir = _results_root() / model / "processing_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot
    logger.info("Creating processing comparison visualization")
    path = create_processing_comparison(
        processing_data,
        out_dir,
        schema_mode=schema_mode,
        metadata_labels=metadata_labels,
        model_name=model,
    )
    logger.info("Visualization created: %s", path)
    print(f"\nCreated: {path}")


def run_cross_model_comparison() -> None:
    """
    Run cross-model comparison workflow: compare multiple models (one run per model).

    This function guides the user through selecting multiple models and a processing
    run for each, then generates a comparison visualization.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting cross-model comparison workflow")

    models = discover_models()
    selected_models = prompt_select_multiple_models(models)
    if not selected_models:
        logger.info("User cancelled model selection")
        return

    # For each model, choose a run
    model_to_run: Dict[str, Path] = {}
    for m in selected_models:
        runs = discover_runs_for_model(m)
        if not runs:
            logger.error("No processing runs with unified summaries found for %s", m)
            print(
                f"\n[ERROR] No processing_* runs with unified summaries found for {m}"
            )
            return
        choice = prompt_select_run(runs, f"Select processing run for model: {m}")
        if not choice:
            logger.info("User cancelled run selection for %s", m)
            return
        model_to_run[m] = choice
        logger.debug("Selected run for %s: %s", m, choice.name)

    # Load and check schema modes
    model_results: Dict[str, List[Dict[str, Any]]] = {}
    metadata_labels: Dict[str, str] = {}
    modes: List[str] = []
    for m, u in model_to_run.items():
        data = load_unified_summary(u)
        if not data:
            run_name = u.name if u.name.startswith("processing_") else u.parent.name
            logger.error("Failed loading summary for %s (%s)", m, run_name)
            print(f"[ERROR] Failed loading summary for {m} ({run_name})")
            return
        files = extract_file_results(data)
        if not files:
            run_name = u.name if u.name.startswith("processing_") else u.parent.name
            logger.error("No file results for %s (%s)", m, run_name)
            print(f"[ERROR] No file results for {m} ({run_name})")
            return
        model_results[m] = files
        # Load metadata label for this model's run
        metadata_labels[m] = load_processing_metadata_label(u, m, model_name=m)
        modes.append(get_schema_mode_from_files(files))

    # Enforce same schema_mode across all selections
    if len(set(modes)) != 1:
        logger.error("Selected models/runs use different schema modes: %s", set(modes))
        print("\n[ERROR] Selected models/runs use different schema modes.")
        print(
            "Please rerun unified evaluation for selected models/runs with the same mode and try again."
        )
        return
    schema_mode = modes[0]
    logger.info("Using schema mode: %s", schema_mode)

    # Output dir and plot
    out_dir = Path("results") / "cross_model"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Creating cross-model comparison visualization")
    path = create_cross_model_comparison(
        model_results, out_dir, schema_mode=schema_mode, metadata_labels=metadata_labels
    )
    logger.info("Visualization created: %s", path)
    print(f"\nCreated: {path}")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="TEI Evaluation Framework - Comparison Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (default)
  python compare.py

  # Internal comparison for specific model (CLI)
  python compare.py --mode internal --model gpt-4

  # Cross-model comparison (CLI)
  python compare.py --mode cross --models gpt-4,claude-3

  # With logging
  python compare.py --mode internal --model gpt-4 --log-level DEBUG
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["internal", "cross"],
        help="Comparison mode: internal (one model, multiple runs) or cross (multiple models)",
    )

    parser.add_argument("--model", type=str, help="Model name for internal comparison")

    parser.add_argument(
        "--models",
        type=str,
        help="Comma-separated model names for cross-model comparison (e.g., gpt-4,claude-3)",
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Force interactive mode even if arguments are provided",
    )

    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )

    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output (DEBUG level)"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set logging level (default: INFO, or DEBUG if --verbose)",
    )

    return parser.parse_args()


def main() -> None:
    """
    Main function with menu system and command-line interface.

    Supports both interactive and command-line modes for running
    comparison visualizations.
    """
    args = parse_arguments()

    # Determine log level
    if args.log_level:
        log_level = getattr(logging, args.log_level)
    elif args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    # Setup logging
    paths = get_path_config()
    log_file = paths.logs_dir / "comparison.log"
    setup_logger(
        "tei_evaluator", log_file=str(log_file), level=log_level, console_output=True
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting comparison tool (version %s)", __version__)

    # If interactive flag is set or no mode specified, use interactive menu
    if args.interactive or args.mode is None:
        logger.info("Running in interactive mode")
        choice = prompt_menu()
        if choice == "1":
            run_internal_comparison()
        elif choice == "2":
            run_cross_model_comparison()
        else:
            logger.info("User exited")
            print("\nExiting...")
    else:
        # Command-line mode
        logger.info("Running in command-line mode: %s", args.mode)
        if args.mode == "internal":
            if args.model:
                # Non-interactive internal comparison not yet fully implemented
                logger.warning(
                    "Non-interactive internal comparison not yet fully implemented"
                )
                print(
                    "[INFO] Non-interactive mode for internal comparison coming soon."
                )
                print("Please use interactive mode for now.")
            else:
                run_internal_comparison()
        elif args.mode == "cross":
            if args.models:
                # Non-interactive cross-model comparison not yet fully implemented
                logger.warning(
                    "Non-interactive cross-model comparison not yet fully implemented"
                )
                print(
                    "[INFO] Non-interactive mode for cross-model comparison coming soon."
                )
                print("Please use interactive mode for now.")
            else:
                run_cross_model_comparison()

    logger.info("Comparison tool finished")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
    except (IOError, OSError, RuntimeError, ValueError) as e:
        print(f"\n[ERROR] Error: {e}")
        traceback.print_exc()
    except Exception as e:  # noqa: BLE001
        # Catch-all for unexpected errors to maintain reproducibility
        print(f"\n[ERROR] Unexpected error: {e}")
        traceback.print_exc()


