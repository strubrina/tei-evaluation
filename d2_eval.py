"""
TEI Evaluation Framework - Dimension 2: Schema Compliance Evaluation

This script evaluates XML files against TEI and project-specific schemas.

Usage:
    python d2_eval.py --xml-dir <path> --output-dir <path> [options]
    python d2_eval.py --interactive
"""

import argparse
import logging
import sys
import traceback
from pathlib import Path
from typing import Optional

# Add the parent directory to Python path so we can import tei_evaluator
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import get_path_config, setup_evaluation_paths
from tei_evaluator import D2Schema
from tei_evaluator.reporting import D2Reporter
from tei_evaluator.utils.logging_config import setup_logger

__version__ = "1.0.0"


def run_evaluation(
    xml_directory: str,
    output_directory: str,
    pattern: str = "*.xml",
    validation_mode: str = "combined",
    quiet: bool = False
) -> bool:
    """
    Run Dimension 2 schema validation evaluation.

    Args:
        xml_directory: Directory containing XML files to evaluate
        output_directory: Directory to save evaluation reports
        pattern: Glob pattern for XML files (default: "*.xml")
        validation_mode: Validation mode - "combined", "tei_only", or "project_only"
        quiet: If True, suppress console output except errors

    Returns:
        True if evaluation completed successfully, False otherwise
    """
    logger = logging.getLogger(__name__)

    try:
        if not quiet:
            print("\n=== TEI DIMENSION 2 SCHEMA COMPLIANCE EVALUATION ===\n")
            print(f"XML Directory: {xml_directory}")
            print(f"Output Directory: {output_directory}")
            print(f"Validation Mode: {validation_mode}")
            print(f"File Pattern: {pattern}")
            print("-" * 60)

        # Create evaluator and reporter
        evaluator = D2Schema(quiet=quiet)
        reporter = D2Reporter()

        # Check if directory exists
        if not Path(xml_directory).exists():
            logger.error("XML directory not found: %s", xml_directory)
            print(f"[ERROR] XML directory not found: {xml_directory}")
            return False

        # Run evaluation based on mode
        logger.info("Starting %s validation", validation_mode)
        if validation_mode == "combined":
            results = evaluator.evaluate_full(xml_directory, pattern)
        elif validation_mode == "tei_only":
            results = evaluator.evaluate_tei_only(xml_directory, pattern)
        elif validation_mode == "project_only":
            results = evaluator.evaluate_project_only(xml_directory, pattern)
        else:
            logger.error("Invalid validation mode: %s", validation_mode)
            print(f"[ERROR] Invalid validation mode: {validation_mode}")
            return False

        # Print summary
        if not quiet:
            evaluator.print_batch_summary(results)

        # Save reports
        if results:
            base_output_dir = Path(output_directory)
            base_output_dir.mkdir(parents=True, exist_ok=True)

            reporter.save_detailed_report(results, str(base_output_dir))
            logger.info("Reports saved to: %s", base_output_dir)

            if not quiet:
                print(f"\n[OUTPUT] Reports saved to: {base_output_dir}")
        else:
            logger.warning("No results to save")
            if not quiet:
                print("[INFO] No results to save")

        return True

    except (IOError, OSError, RuntimeError, ValueError) as e:
        logger.error("Error during batch evaluation: %s", str(e), exc_info=True)
        print(f"[ERROR] Error during batch evaluation: {str(e)}")
        traceback.print_exc()
        return False


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="TEI Evaluation - Dimension 2: Schema Compliance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Combined validation (TEI + Project schema)
  python d2_eval.py --xml-dir data/output/model1 --output-dir results/d2

  # TEI schema validation only
  python d2_eval.py --xml-dir data/output/model1 --output-dir results/d2 --mode tei_only

  # Project schema validation only
  python d2_eval.py --xml-dir data/output/model1 --output-dir results/d2 --mode project_only

  # Interactive mode
  python d2_eval.py --interactive

  # Verbose logging
  python d2_eval.py --xml-dir data/output/model1 --output-dir results/d2 --verbose

  # Quiet mode (suppress console output)
  python d2_eval.py --xml-dir data/output/model1 --output-dir results/d2 --quiet
        """
    )

    parser.add_argument(
        '--xml-dir',
        type=str,
        help='Directory containing XML files to evaluate'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Directory to save evaluation reports'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.xml',
        help='Glob pattern for XML files (default: *.xml)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['combined', 'tei_only', 'project_only'],
        default='combined',
        help='Validation mode: combined (default), tei_only, or project_only'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode with menu'
    )
    parser.add_argument(
        '--version',
        action='version',
        version=f'%(prog)s {__version__}'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose (DEBUG level) logging'
    )
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress console output except errors'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Set log level explicitly'
    )

    args = parser.parse_args()

    # Determine log level
    if args.log_level:
        log_level = getattr(logging, args.log_level)
    elif args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    # Setup initial logging (will be reconfigured with proper path)
    setup_logger(
        'tei_evaluator',
        level=log_level,
        console_output=not args.quiet
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting Dimension 2 evaluation (version %s)", __version__)

    # Get path configuration for log directory
    paths = get_path_config()

    # If arguments provided, use them directly
    if args.xml_dir and args.output_dir:
        xml_directory = args.xml_dir
        output_directory = args.output_dir

        # Setup logging to configured logs directory
        log_file = paths.logs_dir / "d2_evaluation.log"
        setup_logger(
            'tei_evaluator',
            log_file=str(log_file),
            level=log_level,
            console_output=not args.quiet
        )

        if not args.quiet:
            print(f"[INFO] Using XML directory: {xml_directory}")
            print(f"[INFO] Output will be saved to: {output_directory}")

        # Run evaluation
        success = run_evaluation(
            xml_directory,
            output_directory,
            args.pattern,
            args.mode,
            args.quiet
        )
        sys.exit(0 if success else 1)

    # Interactive mode or use config
    if args.interactive or (not args.xml_dir and not args.output_dir):
        if not args.quiet:
            print("\n" + "="*60)
            print("TEI EVALUATION - DIMENSION 2 SCHEMA VALIDATION")
            print("="*60)

        # Use helper function to get paths
        result = setup_evaluation_paths(paths, allow_batch=False)
        if result is None:
            return

        xml_directory, output_base, _, _ = result

        # Setup logging to configured logs directory
        log_file = paths.logs_dir / "d2_evaluation.log"
        setup_logger(
            'tei_evaluator',
            log_file=str(log_file),
            level=log_level,
            console_output=not args.quiet
        )

        if not args.quiet:
            print(f"[INFO] Using XML directory: {xml_directory}")
            print(f"[INFO] Output will be saved to: {output_base}")

        # Display menu for mode selection (unless --mode was explicitly provided)
        # Check if mode was explicitly set by user or is just the default
        mode_was_set = '--mode' in sys.argv or '-m' in sys.argv

        if not mode_was_set and not args.quiet:
            print("\nChoose evaluation mode:")
            print("1. TEI + Project Schema Validation (Combined)")
            print("2. TEI Schema Validation Only")
            print("3. Project Schema Validation Only")
            print("4. Exit")
            print("-" * 60)

            while True:
                choice = input("Enter choice (1-4): ").strip()
                if choice in ['1', '2', '3', '4']:
                    break
                print("[ERROR] Invalid choice. Please enter 1, 2, 3, or 4.")

            if choice == '4':
                print("\nGoodbye!")
                return

            mode_map = {
                '1': 'combined',
                '2': 'tei_only',
                '3': 'project_only'
            }
            validation_mode = mode_map[choice]
        else:
            validation_mode = args.mode

        # Run evaluation
        success = run_evaluation(
            xml_directory,
            output_base,
            args.pattern,
            validation_mode,
            args.quiet
        )

        if not args.quiet:
            print("\nEvaluation complete.")

        sys.exit(0 if success else 1)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
