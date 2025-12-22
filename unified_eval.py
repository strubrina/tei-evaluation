"""
Unified TEI Evaluation Framework.

This module orchestrates the execution of all evaluation dimensions (D0-D4)
and generates unified reports. It provides both interactive and command-line
interfaces for running evaluations on single models, multiple models, or
specific processing runs.

The framework supports:
- Running complete evaluation pipelines (all dimensions + unified report)
- Running individual dimension evaluations
- Generating unified reports from existing results
- Batch processing of multiple models and processing runs
- Configurable schema validation modes
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# Add the parent directory to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import (
    EvaluationConfig,
    PathConfig,
    copy_processing_metadata,
    get_path_config,
    select_model_interactive,
    select_processing_run_interactive,
)
from d0_eval import run_evaluation as d0_run_evaluation
from d1_eval import run_evaluation as d1_run_evaluation
from d2_eval import run_evaluation as d2_run_evaluation
from d3_eval import run_evaluation as d3_run_evaluation
from d4_eval import run_evaluation as d4_run_evaluation
from tei_evaluator.reporting.unified_report import create_unified_report
from tei_evaluator.utils.logging_config import setup_logger

__version__ = "1.0.0"


class EvaluationOrchestrator:
    """
    Orchestrates the execution of all evaluation dimensions.

    This class manages the workflow for running TEI evaluations across multiple
    dimensions, handling both single and batch processing scenarios.

    Attributes:
        paths: Configuration object containing all path settings.
        eval_config: Evaluation configuration including schema validation mode.
        current_model: Name of the currently selected model.
        current_processing_run: Identifier of the current processing run.
        logger: Logger instance for this orchestrator.
        quiet: If True, suppresses console output.
    """

    def __init__(
        self,
        paths: PathConfig,
        eval_config: Optional[EvaluationConfig] = None,
        quiet: bool = False,
    ):
        """
        Initialize the evaluation orchestrator.

        Args:
            paths: Path configuration object.
            eval_config: Evaluation configuration. If None, uses default.
            quiet: If True, suppresses console output.
        """
        self.paths = paths
        self.eval_config = eval_config or EvaluationConfig.default()
        self.current_model: Optional[str] = None
        self.current_processing_run: Optional[str] = None
        self.logger = logging.getLogger(__name__)
        self.quiet = quiet

    def select_model(self) -> Optional[str]:
        """
        Let user select a model for evaluation interactively.

        Returns:
            Selected model name, or None if cancelled.
        """
        return select_model_interactive(self.paths)

    def run_full_evaluation(
        self, model_name: Optional[str] = None, processing_run: Optional[str] = None
    ) -> bool:
        """
        Run all 5 dimensions of evaluation sequentially.

        Args:
            model_name: Model to evaluate, or None to prompt user.
            processing_run: Specific processing run, 'all', or None to prompt user.

        Returns:
            True if evaluation completed successfully.
        """
        self.logger.info("Starting full TEI evaluation (all 5 dimensions)")
        if not self.quiet:
            print("=== RUNNING FULL TEI EVALUATION (ALL 5 DIMENSIONS) ===\n")

        # If no model specified, let user select
        if model_name is None:
            model_name = self.select_model()
            if model_name is None:
                return False

        # Handle 'all models' option
        if model_name == 'all':
            models = self.paths.get_model_dirs()
            self.logger.info("Evaluating all %d models with all processing runs", len(models))
            if not self.quiet:
                print(f"\nEvaluating all {len(models)} models with all their processing runs...\n")

            all_success = True
            for model in models:
                self.logger.info("Processing model: %s", model)
                if not self.quiet:
                    print(f"{'='*60}")
                    print(f"Processing model: {model}")
                    print(f"{'='*60}")
                # For 'all models', evaluate all processing runs for each model
                success = self._run_evaluation_for_model_all_runs(model)
                if not success:
                    all_success = False

            return all_success
        else:
            # Single model selected
            self.current_model = model_name

            # Check if we need to select processing run
            if processing_run is None and not self.paths.has_old_structure(model_name):
                processing_run = select_processing_run_interactive(self.paths, model_name)
                if processing_run is None:
                    return False

            # Handle 'all runs' option for single model
            if processing_run == 'all':
                self.current_processing_run = 'all'
                return self._run_evaluation_for_model_all_runs(model_name)
            else:
                self.current_processing_run = processing_run
                return self._run_evaluation_for_model_single_run(model_name, processing_run)


    def _run_evaluation_for_model_all_runs(self, model_name: str) -> bool:
        """
        Run all 4 dimensions of evaluation for all processing runs of a model.

        Args:
            model_name: Name of the model to evaluate

        Returns:
            True if at least one evaluation completed successfully
        """
        # Get all XML directories for this model
        xml_dirs = self.paths.get_xml_directories(model_name)

        if not xml_dirs:
            self.logger.error("No XML files found for model: %s", model_name)
            if not self.quiet:
                print(f"[ERROR] No XML files found for model: {model_name}")
            return False

        self.logger.info("Found %d processing run(s) for %s", len(xml_dirs), model_name)
        if not self.quiet:
            print(f"\nFound {len(xml_dirs)} processing run(s) for {model_name}")

        all_success = True
        for xml_dir in xml_dirs:
            # Determine run identifier for output naming
            if self.paths.has_old_structure(model_name):
                run_identifier = "direct"
            else:
                run_identifier = xml_dir.name  # e.g., "processing_20251103_122008"

            self.logger.info("Processing: %s / %s", model_name, run_identifier)
            if not self.quiet:
                print(f"\n{'='*60}")
                print(f"Processing: {model_name} / {run_identifier}")
                print(f"{'='*60}")

            success = self._run_evaluation_for_directory(
                model_name,
                str(xml_dir),
                run_identifier
            )
            if not success:
                all_success = False

        return all_success

    def _run_evaluation_for_model_single_run(self, model_name: str, processing_run: Optional[str]) -> bool:
        """
        Run all 4 dimensions of evaluation for a single processing run.

        Args:
            model_name: Name of the model to evaluate
            processing_run: Processing run identifier ('direct' for old structure,
                          or processing run name for new structure)

        Returns:
            True if evaluation completed successfully
        """
        # Determine the XML directory
        if processing_run == 'direct' or self.paths.has_old_structure(model_name):
            xml_dir = self.paths.get_model_output_dir(model_name)
            run_identifier = "direct"
        else:
            xml_dir = self.paths.get_processing_dir(model_name, processing_run)
            run_identifier = processing_run

        if not xml_dir.exists():
            self.logger.error("Directory not found: %s", xml_dir)
            if not self.quiet:
                print(f"[ERROR] Directory not found: {xml_dir}")
            return False

        self.logger.info("Evaluating: %s / %s", model_name, run_identifier)
        if not self.quiet:
            print(f"\nEvaluating: {model_name} / {run_identifier}")

        return self._run_evaluation_for_directory(
            model_name,
            str(xml_dir),
            run_identifier
        )

    def _run_evaluation_for_directory(self, model_name: str, test_directory: str, run_identifier: str) -> bool:
        """
        Run all 4 dimensions of evaluation for a specific directory.

        Args:
            model_name: Name of the model
            test_directory: Directory containing XML files to evaluate
            run_identifier: Identifier for this run (for output naming)

        Returns:
            True if evaluation completed successfully
        """
        # Create output directory structure
        # Structure: results/model/{model_name}/{run_identifier}/
        if run_identifier == "direct":
            output_base = str(self.paths.get_model_evaluation_dir(model_name))
        else:
            output_base = str(self.paths.evaluation_dir / "model" / model_name / run_identifier)

        self.logger.info("Input directory: %s", test_directory)
        self.logger.info("Output directory: %s", output_base)
        if not self.quiet:
            print(f"Input directory: {test_directory}")
            print(f"Output directory: {output_base}")

        if not Path(test_directory).exists():
            self.logger.error("Test directory not found: %s", test_directory)
            if not self.quiet:
                print(f"[ERROR] Test directory not found: {test_directory}")
            return False

        # Check for XML files
        xml_files = list(Path(test_directory).glob("*.xml"))
        if not xml_files:
            self.logger.error("No XML files found in %s", test_directory)
            if not self.quiet:
                print(f"[ERROR] No XML files found in {test_directory}")
            return False

        self.logger.info("Found %d XML files to evaluate", len(xml_files))
        if not self.quiet:
            print(f"Found {len(xml_files)} XML files to evaluate")

        # Create output directory structure
        Path(output_base).mkdir(parents=True, exist_ok=True)

        # Copy processing metadata if available
        copy_processing_metadata(test_directory, output_base)

        success_count = 0
        total_dimensions = 5

        # Run Dimension 0 evaluation (Wellformedness)
        success_count += self._run_dimension0(test_directory, output_base)

        # Run Dimension 1 evaluation (Source Fidelity)
        success_count += self._run_dimension1(test_directory, output_base)

        # Run Dimension 2 evaluation (Schema Validation)
        success_count += self._run_dimension2(test_directory, output_base)

        # Run Dimension 3 evaluation (Structural Comparison)
        success_count += self._run_dimension3(test_directory, output_base)

        # Run Dimension 4 evaluation (Semantic Content Matching)
        success_count += self._run_dimension4(test_directory, output_base)

        self.logger.info("Evaluation summary: %d/%d dimensions completed", success_count, total_dimensions)
        if not self.quiet:
            print(f"\nEvaluation Summary: {success_count}/{total_dimensions} dimensions completed successfully")

        return success_count > 0

    def _run_dimension0(self, test_directory: str, output_base: str) -> int:
        """
        Run Dimension 0: XML Wellformedness evaluation.

        Args:
            test_directory: Directory containing XML files to evaluate.
            output_base: Base output directory for results.

        Returns:
            1 if successful, 0 if failed.
        """
        self.logger.info("Running Dimension 0 (Wellformedness) evaluation")
        if not self.quiet:
            print("\nStep 1/5: Running Dimension 0 (Wellformedness) evaluation...")
        try:
            dimension0_output = f"{output_base}"
            success = d0_run_evaluation(test_directory, output_directory=dimension0_output)
            if success:
                self.logger.info("Dimension 0 completed successfully")
                if not self.quiet:
                    print("   Dimension 0 completed successfully")
                return 1
            else:
                self.logger.error("Dimension 0 failed")
                if not self.quiet:
                    print("   [ERROR] Dimension 0 failed")
                return 0
        except (IOError, OSError, RuntimeError, ValueError) as e:
            self.logger.error("Dimension 0 failed: %s", str(e), exc_info=True)
            if not self.quiet:
                print(f"   [ERROR] Dimension 0 failed: {e}")
            return 0

    def _run_dimension1(self, test_directory: str, output_base: str) -> int:
        """
        Run Dimension 1: Source Fidelity evaluation.

        Args:
            test_directory: Directory containing XML files to evaluate.
            output_base: Base output directory for results.

        Returns:
            1 if successful, 0 if failed.
        """
        self.logger.info("Running Dimension 1 (Source Fidelity) evaluation")
        if not self.quiet:
            print("\nStep 2/5: Running Dimension 1 (Source Fidelity) evaluation...")
        try:
            dimension1_output = f"{output_base}"
            txt_directory = str(self.paths.input_dir)
            success = d1_run_evaluation(
                test_directory,
                txt_directory=txt_directory,
                output_directory=dimension1_output,
            )
            if success:
                self.logger.info("Dimension 1 completed successfully")
                if not self.quiet:
                    print("   Dimension 1 completed successfully")
                return 1
            else:
                self.logger.error("Dimension 1 failed")
                if not self.quiet:
                    print("   [ERROR] Dimension 1 failed")
                return 0
        except (IOError, OSError, RuntimeError, ValueError) as e:
            self.logger.error("Dimension 1 failed: %s", str(e), exc_info=True)
            if not self.quiet:
                print(f"   [ERROR] Dimension 1 failed: {e}")
            return 0

    def _run_dimension2(self, test_directory: str, output_base: str) -> int:
        """
        Run Dimension 2: Schema validation.

        Args:
            test_directory: Directory containing XML files to evaluate.
            output_base: Base output directory for results.

        Returns:
            1 if successful, 0 if failed.
        """
        self.logger.info("Running Dimension 2 (Schema Validation) evaluation")
        if not self.quiet:
            print("\nStep 3/5: Running Dimension 2 (Schema Validation) evaluation...")
        try:
            dimension2_output = f"{output_base}"
            mode = (self.eval_config.schema_validation_mode or "both").lower()

            # Map mode names to validation_mode parameter
            if mode == "tei":
                validation_mode = "tei_only"
                self.logger.info("D2 mode: TEI-only validation")
                if not self.quiet:
                    print("   D2 mode: TEI-only validation")
            elif mode == "project":
                validation_mode = "project_only"
                self.logger.info("D2 mode: Project-only validation")
                if not self.quiet:
                    print("   D2 mode: Project-only validation")
            else:
                # Default to combined for "both" or "none"
                validation_mode = "combined"
                self.logger.info("D2 mode: Combined (TEI + Project) validation")
                if not self.quiet:
                    print("   D2 mode: Combined (TEI + Project) validation")

            success = d2_run_evaluation(
                test_directory,
                output_directory=dimension2_output,
                validation_mode=validation_mode,
                quiet=self.quiet
            )

            if success:
                self.logger.info("Dimension 2 completed successfully")
                if not self.quiet:
                    print("   Dimension 2 completed successfully")
                return 1
            else:
                self.logger.error("Dimension 2 failed")
                if not self.quiet:
                    print("   [ERROR] Dimension 2 failed")
                return 0

        except (IOError, OSError, RuntimeError, ValueError) as e:
            self.logger.error("Dimension 2 failed: %s", str(e), exc_info=True)
            if not self.quiet:
                print(f"   [ERROR] Dimension 2 failed: {e}")
            return 0

    def _run_dimension3(self, test_directory: str, output_base: str) -> int:
        """
        Run Dimension 3: Structural comparison.

        Args:
            test_directory: Directory containing XML files to evaluate.
            output_base: Base output directory for results.

        Returns:
            1 if successful, 0 if failed.
        """
        self.logger.info("Running Dimension 3 (Structural Comparison) evaluation")
        if not self.quiet:
            print("\nStep 4/5: Running Dimension 3 (Structural Comparison) evaluation...")
        try:
            dimension3_output = f"{output_base}"
            reference_directory = str(self.paths.references_dir)
            success = d3_run_evaluation(
                test_directory,
                reference_directory=reference_directory,
                output_directory=dimension3_output,
            )
            if success:
                self.logger.info("Dimension 3 completed successfully")
                if not self.quiet:
                    print("   Dimension 3 completed successfully")
                return 1
            else:
                self.logger.error("Dimension 3 failed")
                if not self.quiet:
                    print("   [ERROR] Dimension 3 failed")
                return 0
        except (ImportError, IOError, OSError, RuntimeError, ValueError) as e:
            self.logger.error("Dimension 3 failed: %s", str(e), exc_info=True)
            if not self.quiet:
                print(f"   [ERROR] Dimension 3 failed: {e}")
            return 0

    def _run_dimension4(self, test_directory: str, output_base: str) -> int:
        """
        Run Dimension 4: Semantic content matching.

        Args:
            test_directory: Directory containing XML files to evaluate.
            output_base: Base output directory for results.

        Returns:
            1 if successful, 0 if failed.
        """
        self.logger.info("Running Dimension 4 (Semantic Content Matching) evaluation")
        if not self.quiet:
            print("\nStep 5/5: Running Dimension 4 (Semantic Content Matching) evaluation...")
        try:
            dimension4_output = f"{output_base}"
            reference_directory = str(self.paths.references_dir)
            success = d4_run_evaluation(
                test_directory,
                reference_directory=reference_directory,
                output_directory=dimension4_output,
            )
            if success:
                self.logger.info("Dimension 4 completed successfully")
                if not self.quiet:
                    print("   Dimension 4 completed successfully")
                return 1
            else:
                self.logger.error("Dimension 4 failed")
                if not self.quiet:
                    print("   [ERROR] Dimension 4 failed")
                return 0
        except (IOError, OSError, RuntimeError, ValueError) as e:
            self.logger.error("Dimension 4 failed: %s", str(e), exc_info=True)
            if not self.quiet:
                print(f"   [ERROR] Dimension 4 failed: {e}")
            return 0

    def generate_unified_report(self, model_name: Optional[str] = None, processing_run: Optional[str] = None) -> bool:
        """
        Generate unified report from existing evaluation outputs.

        Args:
            model_name: Model name, or None to prompt user
            processing_run: Processing run identifier, or None to prompt user

        Returns:
            True if report generation succeeded
        """

        self.logger.info("Generating unified report from existing results")
        if not self.quiet:
            print("=== GENERATING UNIFIED REPORT FROM EXISTING RESULTS ===\n")

        # If no model specified, let user select from evaluated models
        if model_name is None:
            # Look for evaluated models in evaluation directory
            if not self.paths.evaluation_dir.exists():
                self.logger.error("No evaluation directory found at: %s", self.paths.evaluation_dir)
                if not self.quiet:
                    print(f"[ERROR] No evaluation directory found at: {self.paths.evaluation_dir}")
                return False

            # Look in results/model/ directory
            model_dir = self.paths.evaluation_dir / "model"
            if not model_dir.exists():
                self.logger.error("No model evaluation directory found at: %s", model_dir)
                if not self.quiet:
                    print(f"[ERROR] No model evaluation directory found at: {model_dir}")
                return False

            evaluated_models = [d.name for d in model_dir.iterdir() if d.is_dir()]
            if not evaluated_models:
                self.logger.error("No evaluated models found")
                if not self.quiet:
                    print("[ERROR] No evaluated models found")
                return False

            print("Available evaluated models:")
            for i, model in enumerate(sorted(evaluated_models), 1):
                print(f"   {i}. {model}")
            print(f"   {len(evaluated_models) + 1}. All models (all processing runs)")

            run_all_models = False
            while True:
                try:
                    choice = (
                        input(f"\nSelect model (1-{len(evaluated_models)} or 'all'): ")
                        .strip()
                        .lower()
                    )

                    if choice in ("all", "a"):
                        run_all_models = True
                        break

                    idx = int(choice) - 1
                    if 0 <= idx < len(evaluated_models):
                        model_name = sorted(evaluated_models)[idx]
                        break
                    elif idx == len(evaluated_models):
                        run_all_models = True
                        break
                    else:
                        print(
                            f"[ERROR] Please enter a number between 1 and {len(evaluated_models)} (or 'all')"
                        )
                except ValueError:
                    print("[ERROR] Please enter a valid number or 'all'")
                except KeyboardInterrupt:
                    print("\n\nSelection cancelled")
                    return False

            if run_all_models:
                all_success = True
                for model in sorted(evaluated_models):
                    self.logger.info("Generating reports for model: %s (all processing runs)", model)
                    if not self.quiet:
                        print(f"\n{'='*60}")
                        print(f"Generating reports for model: {model} (all processing runs)")
                        print(f"{'='*60}")
                    success = self.generate_unified_report(model, processing_run="all")
                    if not success:
                        all_success = False
                return all_success

        # Check if there are multiple processing runs for this model
        model_eval_dir = self.paths.evaluation_dir / "model" / model_name
        if not model_eval_dir.exists():
            self.logger.error("No evaluation directory found for model: %s", model_name)
            if not self.quiet:
                print(f"[ERROR] No evaluation directory found for model: {model_name}")
            return False

        # Look for processing run subdirectories
        run_dirs = [d for d in model_eval_dir.iterdir() if d.is_dir()]

        # Determine if we have old structure (direct) or new structure (processing runs)
        has_processing_runs = any(d.name.startswith("processing_") for d in run_dirs)

        if has_processing_runs and processing_run is None:
            # Let user select which processing run
            processing_runs = [d.name for d in run_dirs if d.name.startswith("processing_")]

            print(f"\nAvailable processing runs for '{model_name}':")
            for i, run in enumerate(sorted(processing_runs, reverse=True), 1):
                timestamp = run.replace("processing_", "")
                print(f"   {i}. {timestamp}")

            while True:
                try:
                    choice = (
                        input(
                            f"\nSelect run (1-{len(processing_runs)}, or 'all' for all runs): "
                        )
                        .strip()
                        .lower()
                    )

                    if choice == "all":
                        processing_run = "all"
                        break

                    idx = int(choice) - 1
                    if 0 <= idx < len(processing_runs):
                        processing_run = sorted(processing_runs, reverse=True)[idx]
                        break
                    else:
                        print(
                            f"[ERROR] Please enter a number between 1 and {len(processing_runs)}"
                        )
                except ValueError:
                    print("[ERROR] Please enter a valid number or 'all'")
                except KeyboardInterrupt:
                    print("\n\nSelection cancelled")
                    return False

        # Handle 'all' processing runs
        if processing_run == 'all':
            processing_runs = [d.name for d in run_dirs if d.name.startswith("processing_")]
            all_success = True
            for run in sorted(processing_runs, reverse=True):
                self.logger.info("Generating report for: %s / %s", model_name, run)
                if not self.quiet:
                    print(f"\n{'='*60}")
                    print(f"Generating report for: {model_name} / {run}")
                    print(f"{'='*60}")
                success = self._generate_report_for_run(model_name, run)
                if not success:
                    all_success = False
            return all_success
        else:
            return self._generate_report_for_run(model_name, processing_run)

    def _generate_report_for_run(self, model_name: str, processing_run: Optional[str]) -> bool:
        """
        Generate unified report for a specific processing run.

        Args:
            model_name: Model name
            processing_run: Processing run identifier (None for old structure)

        Returns:
            True if successful
        """
        # Determine base path
        if processing_run and processing_run != 'direct':
            base_path = self.paths.evaluation_dir / "model" / model_name / processing_run
        else:
            base_path = self.paths.get_model_evaluation_dir(model_name)

        if not base_path.exists():
            self.logger.error("No evaluation output directory found at: %s", base_path)
            if not self.quiet:
                print(f"[ERROR] No evaluation output directory found at: {base_path}")
                print("Please run evaluations first using option 1.")
            return False

        # Check which dimension results exist (new structure: json dir + excel root)
        json_dir = base_path / "json"
        expected_json = {
            "Dimension 0": ["d0_wellformedness_report.json"],
            "Dimension 1": ["d1_source_fidelity_report.json"],
            # Accept any of the schema validation outputs depending on mode
            "Dimension 2": [
                "d2_project_validation_report.json",
                "d2_tei_validation_report.json",
                "d2_schema_validation_report.json",
            ],
            "Dimension 3": ["d3_structural_fidelity_report.json", "d3_detailed_report.json", "d3_detailed_report.json"],
            "Dimension 4": ["d4_semantic_validation_report.json", "d4_detailed_report.json"],
        }

        existing_dimensions = []
        if not json_dir.exists():
            self.logger.error("No JSON directory found at: %s", json_dir)
            if not self.quiet:
                print(f"\n[ERROR] No JSON directory found at: {json_dir}")
            return False

        for dim_name, file_names in expected_json.items():
            found_file = None
            for file_name in file_names:
                file_path = json_dir / file_name
                if file_path.exists():
                    found_file = file_name
                    break
            if found_file:
                self.logger.info("Found %s JSON: %s", dim_name, found_file)
                if not self.quiet:
                    print(f"   Found {dim_name} JSON: {found_file}")
                existing_dimensions.append(dim_name)
            else:
                joined = ", ".join(file_names)
                self.logger.warning("Missing %s JSON (checked): %s", dim_name, joined)
                if not self.quiet:
                    print(f"   [WARNING] Missing {dim_name} JSON (checked): {joined}")

        if not existing_dimensions:
            self.logger.error("No evaluation results found")
            if not self.quiet:
                print("\n[ERROR] No evaluation results found. Please run evaluations first.")
            return False

        self.logger.info("Found results for %d evaluation dimensions", len(existing_dimensions))
        self.logger.info("Schema validation mode: %s", self.eval_config.schema_validation_mode)
        if not self.quiet:
            print(f"\nFound results for {len(existing_dimensions)} evaluation dimensions")
            print("Generating unified report...")
            print(f"   Schema validation mode: {self.eval_config.schema_validation_mode}")

        try:
            create_unified_report(
                str(base_path), schema_mode=self.eval_config.schema_validation_mode
            )

            self.logger.info("Unified report generated successfully")
            if not self.quiet:
                print("   Unified report generated successfully!")
                print("\n[OUTPUT] Reports saved to processing root and json/")

            return True

        except (IOError, OSError, RuntimeError, ValueError) as e:
            self.logger.error("Error generating unified report: %s", str(e), exc_info=True)
            if not self.quiet:
                print(f"   [ERROR] Error generating unified report: {e}")
            return False

    def run_complete_pipeline(self) -> None:
        """
        Run complete evaluation pipeline: all dimensions + unified report.

        This method runs all 5 evaluation dimensions followed by unified report
        generation for the selected model and processing run.
        """
        self.logger.info("Starting complete evaluation pipeline")
        if not self.quiet:
            print("=== COMPLETE EVALUATION PIPELINE ===\n")

        # Step 1: Run all evaluations
        success = self.run_full_evaluation()

        if not success:
            self.logger.error("Evaluation pipeline failed - no results to combine")
            if not self.quiet:
                print("\n[ERROR] Evaluation pipeline failed - no results to combine")
            return

        if not self.quiet:
            print("\n" + "=" * 60)

        # Step 2: Generate unified report using the model info from the evaluation
        self.logger.info("Proceeding to unified report generation")
        if not self.quiet:
            print("Proceeding to unified report generation...")
        if self.current_model:
            self.generate_unified_report(
                self.current_model, processing_run=self.current_processing_run
            )
        else:
            self.generate_unified_report()

    def run_complete_pipeline_all_models(self) -> None:
        """
        Run complete evaluation pipeline for all available models and all their processing runs.

        This method iterates through all models in the output directory and runs
        the complete evaluation pipeline (all dimensions + unified report) for each.
        """
        self.logger.info("Starting complete evaluation pipeline for all models")
        if not self.quiet:
            print("=== COMPLETE EVALUATION PIPELINE - ALL MODELS ===\n")

        models = self.paths.get_model_dirs()

        if not models:
            self.logger.error("No model directories found in data/output/")
            if not self.quiet:
                print("[ERROR] No model directories found in data/output/")
            return

        self.logger.info("Found %d model(s) to evaluate", len(models))
        if not self.quiet:
            print(f"Found {len(models)} model(s) to evaluate")
            print(
                "This will run all 5 dimensions + unified report for each model and all their processing runs\n"
            )

        confirm = input("Continue? (Y/n): ").strip().lower()
        if confirm != "y" and confirm != "":
            self.logger.info("Operation cancelled by user")
            if not self.quiet:
                print("Operation cancelled")
            return

        total_success = 0
        total_failed = 0

        for model in models:
            self.logger.info("Processing model: %s", model)
            if not self.quiet:
                print(f"\n{'='*70}")
                print(f"PROCESSING MODEL: {model}")
                print(f"{'='*70}")

            # Run complete pipeline for this model (all processing runs)
            self.current_model = model
            success = self.run_full_evaluation(model_name=model, processing_run="all")

            if success:
                self.logger.info("Generating unified reports for all processing runs of %s", model)
                if not self.quiet:
                    print(f"\nGenerating unified reports for all processing runs of {model}...")
                self.generate_unified_report(model, processing_run="all")
                total_success += 1
            else:
                self.logger.error("Evaluation failed for %s", model)
                if not self.quiet:
                    print(f"\n[ERROR] Evaluation failed for {model}")
                total_failed += 1

        self.logger.info("All models processed: %d/%d successful", total_success, len(models))
        if not self.quiet:
            print(f"\n{'='*70}")
            print("ALL MODELS PROCESSED")
            print(f"{'='*70}")
            print(f"Successfully evaluated: {total_success}/{len(models)} models")
            if total_failed > 0:
                print(f"Failed: {total_failed}/{len(models)} models")

    def run_complete_pipeline_single_model_all_runs(self) -> None:
        """
        Run complete evaluation pipeline for all processing runs of a single model.

        This method prompts the user to select a model, then runs the complete
        evaluation pipeline for all processing runs of that model.
        """
        self.logger.info("Starting complete evaluation pipeline for single model (all runs)")
        if not self.quiet:
            print("=== COMPLETE EVALUATION PIPELINE - SINGLE MODEL (ALL RUNS) ===\n")

        # Let user select model
        model_name = self.select_model()
        if model_name is None:
            self.logger.info("Operation cancelled by user")
            if not self.quiet:
                print("Operation cancelled")
            return

        if model_name == "all":
            self.logger.error("User selected 'all' instead of specific model")
            if not self.quiet:
                print("[ERROR] Please select a specific model, not 'all'")
            return

        self.current_model = model_name

        # Get processing runs for this model
        if self.paths.has_old_structure(model_name):
            self.logger.info("Model '%s' uses old structure (single output directory)", model_name)
            if not self.quiet:
                print(
                    f"\n[INFO] Model '{model_name}' uses old structure (single output directory)"
                )
            runs = ["direct"]
        else:
            runs = self.paths.get_processing_runs(model_name)
            if not runs:
                self.logger.error("No processing runs found for model '%s'", model_name)
                if not self.quiet:
                    print(f"[ERROR] No processing runs found for model '{model_name}'")
                return

        self.logger.info("Found %d processing run(s) for '%s'", len(runs), model_name)
        if not self.quiet:
            print(f"\nFound {len(runs)} processing run(s) for '{model_name}'")
            print(
                "This will run all 5 dimensions + unified report for each processing run\n"
            )

        confirm = input("Continue? (y/n): ").strip().lower()
        if confirm != "y":
            self.logger.info("Operation cancelled by user")
            if not self.quiet:
                print("Operation cancelled")
            return

        # Run evaluation for all processing runs
        success = self.run_full_evaluation(model_name=model_name, processing_run="all")

        if success:
            self.logger.info("Generating unified reports for all processing runs of %s", model_name)
            if not self.quiet:
                print(
                    f"\nGenerating unified reports for all processing runs of {model_name}..."
                )
            self.generate_unified_report(model_name, processing_run="all")
        else:
            self.logger.error("Evaluation failed for %s", model_name)
            if not self.quiet:
                print(f"\n[ERROR] Evaluation failed for {model_name}")

        self.logger.info("Evaluation complete for model: %s", model_name)
        if not self.quiet:
            print(f"\n{'='*70}")
            print(f"EVALUATION COMPLETE FOR MODEL: {model_name}")
            print(f"{'='*70}")


def check_evaluation_status():
    """Check what evaluation results are available"""

    print("=== CHECKING EVALUATION STATUS ===\n")

    paths = get_path_config()
    base_path = paths.evaluation_dir

    if not base_path.exists():
        print("[ERROR] No evaluation directory found")
        print("Run option 1 to perform evaluations")
        return

    # Check each dimension
    dimensions_info = {
        "dimension0": {
            "name": "Dimension 0 (Wellformedness)",
            "files": ["d0_wellformedness_report.json", "d0_wellformedness_summary.xlsx"]
        },
        "dimension1": {
            "name": "Dimension 1 (Source Fidelity)",
            "files": ["d1_source_fidelity_report.json", "d1_source_fidelity_summary.xlsx"]
        },
        "dimension2": {
            "name": "Dimension 2 (Schema Validation)",
            "files": ["d2_detailed_report.json", "d2_summary.xlsx"]
        },
        "dimension3": {
            "name": "Dimension 3 (Structural Comparison)",
            "files": ["d3_detailed_report.json", "d3_summary.xlsx"]
        },
        "dimension4": {
            "name": "Dimension 4 (Semantic Content Matching)",
            "files": ["d4_detailed_report.json", "d4_summary.xlsx"]
        }
    }

    completed_dimensions = 0

    for dim_dir, info in dimensions_info.items():
        dim_path = base_path / dim_dir
        print(f"{info['name']}:")

        if dim_path.exists():
            found_files = []
            missing_files = []

            for file_name in info['files']:
                file_path = dim_path / file_name
                if file_path.exists():
                    found_files.append(file_name)
                else:
                    missing_files.append(file_name)

            if found_files:
                print(f"   Directory exists with {len(found_files)} files")
                for file_name in found_files:
                    print(f"      - {file_name}")
                completed_dimensions += 1

            if missing_files:
                print("   [WARNING] Missing files:")
                for file_name in missing_files:
                    print(f"      - {file_name}")
        else:
            print(f"   [ERROR] Directory not found: {dim_path}")

        print()

    # Check for unified report in new structure
    print("Unified Report:")
    excel_file = base_path / "unified_evaluation_summary.xlsx"
    json_file = base_path / "json" / "unified_evaluation_summary.json"

    if excel_file.exists() and json_file.exists():
        print("   Unified report exists")
        print("      - unified_evaluation_summary.xlsx")
        print("      - json/unified_evaluation_summary.json")
    else:
        missing = []
        if not excel_file.exists():
            missing.append("unified_evaluation_summary.xlsx")
        if not json_file.exists():
            missing.append("json/unified_evaluation_summary.json")
        print(f"   [WARNING] Unified report files missing: {', '.join(missing)}")

    print(f"\nSummary: {completed_dimensions}/5 evaluation dimensions completed")

    if completed_dimensions > 0:
        print("You can generate a unified report using option 3")
    else:
        print("Run evaluations first using option 1 or 2")

def display_menu():
    """Display the main menu"""
    print("\n" + "="*60)
    print("TEI EVALUATION FRAMEWORK - UNIFIED ANALYSIS")
    print("="*60)
    print("Choose an option:")
    print("1. Run complete evaluation pipeline for ALL models with ALL processing runs")
    print("2. Run complete evaluation pipeline for ONE model with ALL processing runs")
    print("3. Run complete evaluation pipeline for selected model/run (All 4 dimensions + Unified report)")
    print("4. Run individual evaluations only (Dimensions 1-4)")
    print("5. Generate unified report from existing results")
    print("6. Exit")
    print("-" * 60)

def get_user_choice():
    """Get and validate user choice"""
    while True:
        choice = input("Enter choice (1-6): ").strip()
        if choice in ['1', '2', '3', '4', '5', '6']:
            return choice
        print("[ERROR] Invalid choice. Please enter 1, 2, 3, 4, 5, or 6.")

def select_schema_validation_mode() -> str:
    """
    Let user interactively select schema validation mode for overall scoring.

    Returns:
        Selected schema mode: "none", "tei", "project", or "both"
    """
    print("\n" + "="*60)
    print("SCHEMA VALIDATION MODE CONFIGURATION")
    print("="*60)
    print("\nThis setting controls how schema validation affects the overall score.")
    print("\nAvailable modes:")
    print("  1. 'none'    - No schema validation impact on overall score")
    print("  2. 'tei'     - TEI schema validation only")
    print("  3. 'project' - Project-specific schema validation only")
    print("  4. 'both'    - Both TEI and project schema validation")

    print("-" * 60)

    mode_map = {
        '1': 'none',
        '2': 'tei',
        '3': 'project',
        '4': 'both'
    }

    while True:
        choice = input("\nSelect schema validation mode (1-4) [default: 4]: ").strip()

        # Default to 'both' if user just presses Enter
        if not choice:
            choice = '4'

        if choice in mode_map:
            selected_mode = mode_map[choice]
            print(f"\n✓ Schema validation mode set to: '{selected_mode}'")
            return selected_mode
        else:
            print("[ERROR] Invalid choice. Please enter 1, 2, 3, or 4.")

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="TEI Evaluation Framework - Unified Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (default)
  python unified_eval.py

  # Run complete pipeline for all models
  python unified_eval.py --mode all-models --schema-mode both

  # Run complete pipeline for specific model
  python unified_eval.py --mode complete --model gpt-4 --run processing_20251103_122008

  # Run evaluations only (no unified report)
  python unified_eval.py --mode eval-only --model gpt-4

  # Generate unified report from existing results
  python unified_eval.py --mode report-only --model gpt-4 --run all

  # Quiet mode with logging
  python unified_eval.py --mode all-models --quiet --log-level INFO
        """,
    )

    parser.add_argument(
        "--mode",
        choices=[
            "all-models",
            "single-model-all-runs",
            "complete",
            "eval-only",
            "report-only",
        ],
        help="Execution mode: all-models (all models with all runs), "
        "single-model-all-runs (one model, all runs), "
        "complete (selected model/run with report), "
        "eval-only (dimensions only), "
        "report-only (generate report from existing results)",
    )

    parser.add_argument(
        "--model", type=str, help="Model name to evaluate (or 'all' for all models)"
    )

    parser.add_argument(
        "--run",
        type=str,
        help="Processing run identifier (or 'all' for all runs, 'direct' for old structure)",
    )

    parser.add_argument(
        "--schema-mode",
        choices=["none", "tei", "project", "both"],
        default="both",
        help="Schema validation mode for scoring (default: both)",
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
        "--quiet", action="store_true", help="Suppress console output except errors"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set logging level (default: INFO, or DEBUG if --verbose, or WARNING if --quiet)",
    )

    return parser.parse_args()


def main() -> None:
    """
    Main function with menu system and command-line interface.

    Supports both interactive and command-line modes for running
    TEI evaluations across all dimensions.
    """
    args = parse_arguments()

    # Determine log level
    if args.log_level:
        log_level = getattr(logging, args.log_level)
    elif args.verbose:
        log_level = logging.DEBUG
    elif args.quiet:
        log_level = logging.WARNING
    else:
        log_level = logging.INFO

    # Setup logging
    paths = get_path_config()
    log_file = paths.logs_dir / "unified_evaluation.log"
    setup_logger(
        "tei_evaluator",
        log_file=str(log_file),
        level=log_level,
        console_output=not args.quiet,
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting unified evaluation (version %s)", __version__)

    if not args.quiet:
        print("TEI Evaluation Framework - Unified Analysis Tool")
        print("=" * 60)
        print(
            "This tool runs evaluations across all 5 dimensions and generates unified reports\n"
        )

    # Initialize the orchestrator with configuration
    eval_config = EvaluationConfig.default()
    eval_config.schema_validation_mode = args.schema_mode

    # Validate configuration
    try:
        eval_config.validate()
    except ValueError as e:
        logger.error("Configuration validation failed: %s", str(e))
        if not args.quiet:
            print(f"[ERROR] Configuration validation failed: {e}")
        return

    orchestrator = EvaluationOrchestrator(paths, eval_config, quiet=args.quiet)

    # If interactive flag is set or no mode specified, use interactive menu
    if args.interactive or args.mode is None:
        logger.info("Running in interactive mode")
        display_menu()
        choice = get_user_choice()

        # For options that generate unified reports, ask for schema validation mode
        if choice in ["1", "2", "3", "5"]:
            schema_mode = select_schema_validation_mode()
            eval_config.schema_validation_mode = schema_mode
            orchestrator.eval_config = eval_config

        if not args.quiet:
            print("\n" + "=" * 60)

        if choice == "1":
            orchestrator.run_complete_pipeline_all_models()
        elif choice == "2":
            orchestrator.run_complete_pipeline_single_model_all_runs()
        elif choice == "3":
            orchestrator.run_complete_pipeline()
        elif choice == "4":
            orchestrator.run_full_evaluation()
        elif choice == "5":
            orchestrator.generate_unified_report()
        elif choice == "6":
            logger.info("User exited")
            if not args.quiet:
                print("\nGoodbye!")
            return
    else:
        # Command-line mode
        logger.info("Running in command-line mode: %s", args.mode)
        if not args.quiet:
            print("\n" + "=" * 60)

        if args.mode == "all-models":
            orchestrator.run_complete_pipeline_all_models()
        elif args.mode == "single-model-all-runs":
            if args.model:
                orchestrator.current_model = args.model
                orchestrator.run_complete_pipeline_single_model_all_runs()
            else:
                # Let user select interactively
                orchestrator.run_complete_pipeline_single_model_all_runs()
        elif args.mode == "complete":
            success = orchestrator.run_full_evaluation(
                model_name=args.model, processing_run=args.run
            )
            if success and args.model:
                orchestrator.generate_unified_report(
                    model_name=args.model, processing_run=args.run
                )
        elif args.mode == "eval-only":
            orchestrator.run_full_evaluation(
                model_name=args.model, processing_run=args.run
            )
        elif args.mode == "report-only":
            orchestrator.generate_unified_report(
                model_name=args.model, processing_run=args.run
            )

    logger.info("Unified evaluation complete")
    if not args.quiet:
        print("\nEvaluation complete. Run the script again to perform another evaluation.")


if __name__ == "__main__":
    main()