"""
Dimension 1 Evaluation Entry Point.

This script provides command-line interface for running Dimension 1
(Source Fidelity) evaluation on TEI files.
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
from tei_evaluator import D1Source
from tei_evaluator.reporting import D1Reporter
from tei_evaluator.utils.logging_config import setup_logger

__version__ = "1.0.0"

def run_evaluation(xml_directory: Optional[str] = None,
                  txt_directory: Optional[str] = None,
                  output_directory: Optional[str] = None,
                  quiet: bool = False) -> bool:
    """
    Run Dimension 1 (Source Fidelity) evaluation.

    Args:
        xml_directory: Path to XML files. If None, uses paths from config
        txt_directory: Path to original text files. If None, uses paths from config
        output_directory: Output directory for reports. If None, uses paths from config
        quiet: If True, suppress console output except errors

    Returns:
        bool: True if evaluation completed successfully, False otherwise
    """
    logger = logging.getLogger(__name__)

    if not quiet:
        print("\n=== TEI DIMENSION 1 SOURCE FIDELITY EVALUATION ===\n")

    # Initialize path configuration if needed
    if xml_directory is None or txt_directory is None or output_directory is None:
        paths = get_path_config()

        if txt_directory is None:
            txt_directory = str(paths.input_dir)

        if output_directory is None and xml_directory is None:
            output_directory = "evaluation/dimension1"

    # Create evaluator and reporter
    evaluator = D1Source(quiet=quiet)
    reporter = D1Reporter()

    # Validate directories
    if xml_directory and not Path(xml_directory).exists():
        error_msg = f"XML directory not found: {xml_directory}"
        logger.error(error_msg)
        print(f"[ERROR] {error_msg}")
        return False

    if not Path(txt_directory).exists():
        error_msg = f"Text directory not found: {txt_directory}"
        logger.error(error_msg)
        print(f"[ERROR] {error_msg}")
        return False

    logger.info("Starting evaluation for XML directory: %s", xml_directory)
    logger.info("Comparing with text directory: %s", txt_directory)
    if not quiet:
        print(f"Evaluating XML files in: {xml_directory}")
        print(f"Comparing with text files in: {txt_directory}")
        print("-" * 60)

    try:
        # Run evaluation
        results = evaluator.evaluate_directory(xml_directory, txt_directory)

        # Print summary
        if not quiet:
            reporter.print_batch_summary(results)

        # Save reports
        base_output_dir = Path(output_directory)
        base_output_dir.mkdir(parents=True, exist_ok=True)

        reporter.save_detailed_report(results, str(base_output_dir))
        logger.info("Reports saved to: %s", base_output_dir)
        if not quiet:
            print(f"\n[OUTPUT] Reports saved to: {base_output_dir}")

        return True

    except (IOError, OSError, RuntimeError, ValueError) as e:
        logger.error("Error during batch evaluation: %s", str(e), exc_info=True)
        print(f"[ERROR] Error during batch evaluation: {str(e)}")
        traceback.print_exc()
        return False

def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Dimension 1: Source Fidelity Evaluation for TEI files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (prompts for paths)
  python d1_eval.py

  # Evaluate specific directories
  python d1_eval.py --xml-dir data/output/model1 --txt-dir data/input --output-dir results/d1

  # Quiet mode for batch processing
  python d1_eval.py --xml-dir data/output/model1 --txt-dir data/input --output-dir results/d1 --quiet
        """
    )

    parser.add_argument(
        '--version',
        action='version',
        version=f'%(prog)s {__version__}'
    )

    parser.add_argument(
        '--xml-dir',
        type=str,
        help='Directory containing XML files to evaluate'
    )

    parser.add_argument(
        '--txt-dir',
        type=str,
        help='Directory containing original text files'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for evaluation reports'
    )

    parser.add_argument(
        '--pattern',
        type=str,
        default='*.xml',
        help='File pattern to match (default: *.xml)'
    )

    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Use interactive mode to select paths'
    )

    # Verbosity control
    verbosity_group = parser.add_mutually_exclusive_group()
    verbosity_group.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output (DEBUG level logging)'
    )
    verbosity_group.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress console output except errors'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set logging level (default: INFO). Overridden by --verbose or --quiet'
    )

    return parser.parse_args()


def main():
    """Main function to handle user interaction and command-line arguments."""
    args = parse_arguments()

    # Determine log level
    if args.verbose:
        log_level = logging.DEBUG
    elif args.quiet:
        log_level = logging.ERROR
    else:
        log_level = getattr(logging, args.log_level)

    # Setup logging
    setup_logger(
        'tei_evaluator',
        level=log_level,
        console_output=not args.quiet
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting Dimension 1 evaluation (version %s)", __version__)

    # Get path configuration for log directory
    paths = get_path_config()

    # If arguments provided, use them directly
    if args.xml_dir and args.txt_dir and args.output_dir:
        xml_directory = args.xml_dir
        txt_directory = args.txt_dir
        output_directory = args.output_dir

        # Setup logging to configured logs directory
        log_file = paths.logs_dir / "d1_evaluation.log"
        setup_logger(
            'tei_evaluator',
            log_file=str(log_file),
            level=log_level,
            console_output=not args.quiet
        )

        if not args.quiet:
            print(f"[INFO] Using XML directory: {xml_directory}")
            print(f"[INFO] Using text directory: {txt_directory}")
            print(f"[INFO] Output will be saved to: {output_directory}")

        success = run_evaluation(xml_directory, txt_directory, output_directory, quiet=args.quiet)

        if not success:
            logger.error("Evaluation failed")
            print("[ERROR] Evaluation failed. Please check the error messages above.")
            sys.exit(1)

        logger.info("Evaluation completed successfully")
        if not args.quiet:
            print("\nEvaluation complete.")
        sys.exit(0)

    # Otherwise, use interactive mode
    if not args.quiet:
        print("No arguments provided. Starting interactive mode...")

    # Initialize path configuration
    paths = get_path_config()

    # Use helper function to get paths
    result = setup_evaluation_paths(paths, allow_batch=False)
    if result is None:
        return

    xml_directory, output_base, _, _ = result
    txt_directory = str(paths.input_dir)

    # Setup logging to configured logs directory
    log_file = paths.logs_dir / "d1_evaluation.log"
    setup_logger(
        'tei_evaluator',
        log_file=str(log_file),
        level=log_level,
        console_output=not args.quiet
    )

    if not args.quiet:
        print(f"[INFO] Using XML directory: {xml_directory}")
        print(f"[INFO] Using text directory: {txt_directory}")
        print(f"[INFO] Output will be saved to: {output_base}")

    output_directory = f"{output_base}"

    success = run_evaluation(xml_directory, txt_directory, output_directory, quiet=args.quiet)

    if not success:
        logger.error("Evaluation failed")
        print("[ERROR] Evaluation failed. Please check the error messages above.")

    logger.info("Evaluation completed")
    if not args.quiet:
        print("\nEvaluation complete. Run the script again to perform another evaluation.")


if __name__ == "__main__":
    main()

