"""
TEI Evaluation Framework - Dimension 3: Structural Comparison Evaluation

This script evaluates XML files for structural similarity against reference files.

Usage:
    python d3_eval.py --xml-dir <path> --ref-dir <path> --output-dir <path> [options]
    python d3_eval.py --interactive
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
from tei_evaluator.core.d3_structural import D3Structural
from tei_evaluator.reporting.d3_report import D3Reporter
from tei_evaluator.utils.logging_config import setup_logger

__version__ = "1.0.0"


def run_evaluation(
    xml_directory: str,
    reference_directory: Optional[str],
    output_directory: str,
    pattern: str = "*.xml",
    quiet: bool = False
) -> bool:
    """
    Run Dimension 3 structural comparison evaluation.

    Args:
        xml_directory: Directory containing XML files to evaluate
        reference_directory: Directory containing reference XML files
        output_directory: Directory to save evaluation reports
        pattern: Glob pattern for XML files (default: "*.xml")
        quiet: If True, suppress console output except errors

    Returns:
        True if evaluation completed successfully, False otherwise
    """
    logger = logging.getLogger(__name__)

    try:
        if not quiet:
            print("\n=== TEI DIMENSION 3 STRUCTURAL COMPARISON EVALUATION ===\n")
            print(f"XML Directory: {xml_directory}")
            if reference_directory:
                print(f"Reference Directory: {reference_directory}")
            else:
                print("[WARNING] No reference directory configured")
                print("          Files without matching references will be skipped")
            print(f"Output Directory: {output_directory}")
            print(f"File Pattern: {pattern}")
            print("-" * 60)

        # Create evaluator and reporter
        evaluator = D3Structural(reference_directory=reference_directory, quiet=quiet)
        reporter = D3Reporter()

        # Check if directory exists
        if not Path(xml_directory).exists():
            logger.error("XML directory not found: %s", xml_directory)
            print(f"[ERROR] XML directory not found: {xml_directory}")
            return False

        # Configuration: don't fail if no reference file found, just skip
        config = {
            'require_reference': False
        }

        # Run batch evaluation
        logger.info("Starting structural comparison evaluation")
        results = evaluator.evaluate_batch(xml_directory, pattern=pattern, config=config)

        # Print summary
        if not quiet:
            evaluator.print_batch_summary(results)

        # Save reports
        if results:
            output_dir = Path(output_directory)
            output_dir.mkdir(parents=True, exist_ok=True)
            reporter.save_detailed_report(results, str(output_dir))
            logger.info("Reports saved to: %s", output_dir)

            if not quiet:
                print(f"\n[OUTPUT] Reports saved to: {output_dir}")
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
        description="TEI Evaluation - Dimension 3: Structural Comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Structural comparison with explicit paths
  python d3_eval.py --xml-dir data/output/model1 --ref-dir data/references --output-dir results/d3

  # Interactive mode
  python d3_eval.py --interactive

  # Verbose logging
  python d3_eval.py --xml-dir data/output/model1 --ref-dir data/references --output-dir results/d3 --verbose

  # Quiet mode (suppress console output)
  python d3_eval.py --xml-dir data/output/model1 --ref-dir data/references --output-dir results/d3 --quiet
        """
    )

    parser.add_argument(
        '--xml-dir',
        type=str,
        help='Directory containing XML files to evaluate'
    )
    parser.add_argument(
        '--ref-dir',
        type=str,
        help='Directory containing reference XML files'
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
        '--interactive',
        action='store_true',
        help='Run in interactive mode'
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
    logger.info("Starting Dimension 3 evaluation (version %s)", __version__)

    # Get path configuration for log directory
    paths = get_path_config()

    # If arguments provided, use them directly
    if args.xml_dir and args.output_dir:
        xml_directory = args.xml_dir
        output_directory = args.output_dir
        reference_directory = args.ref_dir if args.ref_dir else (str(paths.references_dir) if paths.references_dir.exists() else None)

        # Setup logging to configured logs directory
        log_file = paths.logs_dir / "d3_evaluation.log"
        setup_logger(
            'tei_evaluator',
            log_file=str(log_file),
            level=log_level,
            console_output=not args.quiet
        )

        if not args.quiet:
            print(f"[INFO] Using XML directory: {xml_directory}")
            if reference_directory:
                print(f"[INFO] Using reference directory: {reference_directory}")
            else:
                print("[INFO] No reference directory - files will be skipped if no matching reference")
            print(f"[INFO] Output will be saved to: {output_directory}")

        # Run evaluation
        success = run_evaluation(
            xml_directory,
            reference_directory,
            output_directory,
            args.pattern,
            args.quiet
        )
        sys.exit(0 if success else 1)

    # Interactive mode or use config
    if args.interactive or (not args.xml_dir and not args.output_dir):
        if not args.quiet:
            print("\n" + "="*60)
            print("TEI EVALUATION - DIMENSION 3 STRUCTURAL COMPARISON")
            print("="*60)
            print("Analysis Type: Structural comparison against reference files")
            print("Purpose: Verify XML structure matches expected reference")
            print("Output: Detailed structural analysis with tree edit distance")
            print("="*60)

        # Use helper function to get paths
        result = setup_evaluation_paths(paths, allow_batch=False)
        if result is None:
            return

        xml_directory, output_base, _, _ = result
        reference_directory = str(paths.references_dir) if paths.references_dir.exists() else None

        # Setup logging to configured logs directory
        log_file = paths.logs_dir / "d3_evaluation.log"
        setup_logger(
            'tei_evaluator',
            log_file=str(log_file),
            level=log_level,
            console_output=not args.quiet
        )

        if not args.quiet:
            print(f"[INFO] Using XML directory: {xml_directory}")
            if reference_directory:
                print(f"[INFO] Using reference directory: {reference_directory}")
            else:
                print("[INFO] No reference directory found - files will be skipped if no matching reference")
            print(f"[INFO] Output will be saved to: {output_base}")

        # Run evaluation
        success = run_evaluation(
            xml_directory,
            reference_directory,
            output_base,
            args.pattern,
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
