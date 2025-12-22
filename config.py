"""
TEI Evaluation Framework - Centralized Configuration
====================================================

This module provides centralized path management and configuration for the
TEI evaluation framework. All paths are defined here to ensure consistency
across the codebase.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime
import os


class PathConfig:
    """Centralized path configuration for TEI evaluation framework"""

    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize path configuration.

        Args:
            project_root: Root directory of the project. If None, uses current directory.
        """
        self.project_root = project_root or Path.cwd()

        # ========================================================================
        # DATA DIRECTORIES - Production Data
        # ========================================================================
        # These paths point to your evaluation data (input files, LLM outputs,
        # reference files, and schemas). Modify these paths to point to your specific
        # data directories.
        #
        # Structure:
        #   data/
        #     ├── input/              # Source text files (.txt) for evaluation
        #     ├── output/             # LLM-generated XML files organized by model
        #     │   └── {model-name}/
        #     │       └── processing_YYYYMMDD_HHMMSS/
        #     ├── references/         # Reference XML files (gold standard for comparison)
        #     └── schemas/            # Project-specific RelaxNG schemas (.rng)
        #
        # See README.md "Project Structure" section for detailed directory layout.
        # ========================================================================
        self.data_dir = self.project_root / "data"
        self.input_dir = self.data_dir / "input"
        self.output_dir = self.data_dir / "output"
        self.references_dir = self.data_dir / "references"
        self.schemas_dir = self.data_dir / "schemas"

        # Bundled resources directories (inside tei_evaluator package)
        self.bundled_resources_dir = self.project_root / "tei_evaluator" / "resources"
        self.bundled_schemas_dir = self.bundled_resources_dir / "schemas"
        self.bundled_tools_dir = self.bundled_resources_dir / "tools"

        # Evaluation output directory
        self.evaluation_dir = self.project_root / "results"

        # Logging directory
        self.logs_dir = self.project_root / "logs"

        # Schema files
        # TEI schema: Always use bundled version (standard TEI P5)
        self.tei_schema = self.bundled_schemas_dir / "tei_all.rng"

        # Project schema: User must provide in data/schemas/
        self.project_schema = self._get_project_schema_path()

        # Tools: Use bundled version
        self.jing_jar = self.bundled_tools_dir / "jing-RELEASE220.jar"

    def _get_project_schema_path(self) -> Optional[Path]:
        """
        Get project schema path from user data directory.

        Location: data/schemas/*.rng (any RNG file)

        Returns:
            Path to project schema file if exists, None otherwise
        """
        if not self.schemas_dir.exists():
            return None

        # Find any .rng file in data/schemas/
        rng_files = list(self.schemas_dir.glob("*.rng"))
        if rng_files:
            # Return the first .rng file found (or could sort if multiple)
            return sorted(rng_files)[0]
        return None

    def get_model_dirs(self) -> List[str]:
        """
        Get list of available model directories in data/output/.

        Supports both XML directly in data/output/{model}/ and subfolder structure:
        - XML in model folder: data/output/{model}/*.xml
        - XML in processing subfolders: data/output/{model}/processing_{timestamp}/*.xml

        Returns:
            List of model directory names
        """
        if not self.output_dir.exists():
            return []

        models = []
        for item in self.output_dir.iterdir():
            if item.is_dir():
                # Check if directory contains XML files
                xml_files = list(item.glob("*.xml"))
                if xml_files:
                    models.append(item.name)
                else:
                    # Check for alternative structure: processing_* subdirectories
                    processing_dirs = list(item.glob("processing_*"))
                    if processing_dirs:
                        # Check if any processing directory has XML files
                        has_xml = any(
                            list(proc_dir.glob("*.xml"))
                            for proc_dir in processing_dirs
                            if proc_dir.is_dir()
                        )
                        if has_xml:
                            models.append(item.name)

        return sorted(models)

    def get_model_output_dir(self, model_name: str) -> Path:
        """
        Get the output directory for a specific model.

        Note: This returns the base model directory. For new structure with
        processing_* subdirectories, use get_processing_runs() to list runs
        or get_latest_processing_dir() to get the most recent one.
        """
        return self.output_dir / model_name

    def get_processing_runs(self, model_name: str) -> List[str]:
        """
        Get list of processing run timestamps for a specific model.

        Returns list of processing run directory names (e.g., "processing_20251103_122008")
        sorted by timestamp (newest first).

        Args:
            model_name: Name of the model

        Returns:
            List of processing run directory names, or empty list if none found
        """
        model_dir = self.get_model_output_dir(model_name)
        if not model_dir.exists():
            return []

        processing_dirs = [
            d.name for d in model_dir.iterdir()
            if d.is_dir() and d.name.startswith("processing_")
        ]

        # Sort by timestamp (newest first) - the format processing_YYYYMMDD_HHMMSS allows string sorting
        return sorted(processing_dirs, reverse=True)

    def get_latest_processing_dir(self, model_name: str) -> Optional[Path]:
        """
        Get the most recent processing run directory for a model.

        Args:
            model_name: Name of the model

        Returns:
            Path to latest processing directory, or None if no processing runs exist
        """
        runs = self.get_processing_runs(model_name)
        if runs:
            return self.get_model_output_dir(model_name) / runs[0]
        return None

    def get_processing_dir(self, model_name: str, processing_run: str) -> Path:
        """
        Get the path to a specific processing run directory.

        Args:
            model_name: Name of the model
            processing_run: Processing run name (e.g., "processing_20251103_122008")

        Returns:
            Path to the processing directory
        """
        return self.get_model_output_dir(model_name) / processing_run

    def has_old_structure(self, model_name: str) -> bool:
        """
        Check if model uses structure with XML files directly in model directory.

        Args:
            model_name: Name of the model

        Returns:
            True if XML files are directly in model directory
        """
        model_dir = self.get_model_output_dir(model_name)
        if not model_dir.exists():
            return False

        # Check for XML files directly in model directory
        xml_files = list(model_dir.glob("*.xml"))
        return len(xml_files) > 0

    def get_xml_directories(self, model_name: str, processing_run: Optional[str] = None) -> List[Path]:
        """
        Get list of directories containing XML files for a model.

        Handles both folder structures:
        - XML directly in model folder: Returns [model_dir] if it contains XML files
        - Subfolder structure: Returns specific processing run or all processing runs

        Args:
            model_name: Name of the model
            processing_run: Specific processing run name, or None for all runs

        Returns:
            List of paths to directories containing XML files
        """
        model_dir = self.get_model_output_dir(model_name)
        if not model_dir.exists():
            return []

        # Check for XML files directly in model directory
        if self.has_old_structure(model_name):
            return [model_dir]

        # Subfolder structure with processing runs
        if processing_run:
            # Specific processing run
            proc_dir = self.get_processing_dir(model_name, processing_run)
            if proc_dir.exists():
                return [proc_dir]
            return []
        else:
            # All processing runs
            runs = self.get_processing_runs(model_name)
            return [
                self.get_processing_dir(model_name, run)
                for run in runs
                if (self.get_processing_dir(model_name, run)).exists()
            ]

    def get_model_evaluation_dir(self, model_name: str) -> Path:
        """
        Get the evaluation directory for a specific model.

        Args:
            model_name: Name of the model

        Returns:
            Path to evaluation directory: results/model/{model_name}/
        """
        return self.evaluation_dir / "model" / model_name

    def validate_paths(self, check_output: bool = True) -> Dict[str, bool]:
        """
        Validate that required paths exist.

        Args:
            check_output: Whether to check for model output directories

        Returns:
            Dictionary of path names and their existence status
        """
        validation = {
            "data_dir": self.data_dir.exists(),
            "input_dir": self.input_dir.exists(),
            "output_dir": self.output_dir.exists(),
            "references_dir": self.references_dir.exists(),
            "schemas_dir": self.schemas_dir.exists(),
            "bundled_resources_dir": self.bundled_resources_dir.exists(),
            "bundled_schemas_dir": self.bundled_schemas_dir.exists(),
            "bundled_tools_dir": self.bundled_tools_dir.exists(),
            "tei_schema": self.tei_schema.exists(),
            "project_schema": self.project_schema.exists(),
            "jing_jar": self.jing_jar.exists(),
        }

        if check_output:
            validation["output_dir"] = self.output_dir.exists()
            validation["has_models"] = len(self.get_model_dirs()) > 0

        return validation

    def print_validation_report(self):
        """Print a formatted validation report"""
        print("\n" + "="*60)
        print("PATH CONFIGURATION VALIDATION")
        print("="*60)

        validation = self.validate_paths()

        # Group output for clarity
        print("\n[DATA DIRECTORIES]")
        for key in ['data_dir', 'input_dir', 'output_dir', 'references_dir', 'schemas_dir']:
            if key in validation:
                status = "[OK]" if validation[key] else "[MISSING]"
                print(f"{status} {key}")

        print("\n[BUNDLED RESOURCES]")
        for key in ['bundled_resources_dir', 'bundled_schemas_dir', 'bundled_tools_dir',
                    'tei_schema', 'jing_jar']:
            if key in validation:
                status = "[OK]" if validation[key] else "[MISSING]"
                print(f"{status} {key}")

        print("\n[PROJECT SCHEMA]")
        status = "[OK]" if validation.get('project_schema', False) else "[MISSING]"
        if self.project_schema and self.project_schema.exists():
            print(f"{status} project_schema (User-provided: {self.project_schema.relative_to(self.project_root)})")
        else:
            print(f"{status} project_schema (Missing - add any .rng file to data/schemas/)")

        if 'has_models' in validation:
            print(f"\n[MODELS] Found: {len(self.get_model_dirs())} model(s)")

        print("="*60)

    def ensure_evaluation_dir(self, model_name: str) -> Path:
        """
        Ensure evaluation directory exists and return its path.

        Args:
            model_name: Name of the model

        Returns:
            Path to created evaluation directory
        """
        eval_dir = self.get_model_evaluation_dir(model_name)
        eval_dir.mkdir(parents=True, exist_ok=True)
        return eval_dir



@dataclass
class EvaluationConfig:
    """Configuration for TEI evaluation"""

    # Path configuration
    paths: PathConfig = field(default_factory=PathConfig)

    # Dimension 2 settings
    schema_path: Optional[str] = None
    jing_jar_path: Optional[str] = None

    # Unified scoring settings
    schema_validation_mode: str = "both"  # Options: "none", "tei", "project", "both"

    # Scoring settings
    base_score: float = 100.0
    penalty_weights: Optional[Dict[str, float]] = None

    # Output settings
    output_format: str = "json"  # json, csv, html
    detailed_errors: bool = True

    @classmethod
    def default(cls):
        """Create default configuration"""
        return cls()

    def validate(self):
        """Validate configuration settings"""
        if self.schema_path and not Path(self.schema_path).exists():
            raise ValueError(f"Schema file not found: {self.schema_path}")

        if self.jing_jar_path and not Path(self.jing_jar_path).exists():
            raise ValueError(f"Jing jar not found: {self.jing_jar_path}")

        if not 0 <= self.base_score <= 100:
            raise ValueError("Base score must be between 0 and 100")

        # Validate schema validation mode
        valid_modes = ["none", "tei", "project", "both"]
        if self.schema_validation_mode not in valid_modes:
            raise ValueError(f"schema_validation_mode must be one of {valid_modes}, got: {self.schema_validation_mode}")


# Global path configuration instance
_global_path_config = None


def get_path_config(project_root: Optional[Path] = None) -> PathConfig:
    """
    Get the global path configuration instance.

    Args:
        project_root: Optional project root directory. Only used on first call.

    Returns:
        PathConfig instance
    """
    global _global_path_config

    if _global_path_config is None:
        _global_path_config = PathConfig(project_root)

    return _global_path_config


def select_model_interactive(paths: Optional[PathConfig] = None) -> Optional[str]:
    """
    Interactive model selection from available models.

    Args:
        paths: PathConfig instance. If None, uses global config.

    Returns:
        Selected model name or 'all' for all models, or None if cancelled
    """
    if paths is None:
        paths = get_path_config()

    models = paths.get_model_dirs()

    if not models:
        print("[ERROR] No model directories found in data/output/")
        print(f"        Expected location: {paths.output_dir}")
        return None

    print("\nAvailable models:")
    for i, model in enumerate(models, 1):
        model_dir = paths.get_model_output_dir(model)

        # Count XML files based on structure
        if paths.has_old_structure(model):
            xml_count = len(list(model_dir.glob("*.xml")))
            print(f"   {i}. {model} ({xml_count} XML files)")
        else:
            # New structure - count processing runs
            runs = paths.get_processing_runs(model)
            total_xml = sum(
                len(list(paths.get_processing_dir(model, run).glob("*.xml")))
                for run in runs
            )
            print(f"   {i}. {model} ({len(runs)} processing run{'s' if len(runs) != 1 else ''}, {total_xml} total XML files)")

    while True:
        try:
            choice = input(f"\nSelect model (1-{len(models)}, or 'all' for all models, 'q' to quit): ").strip().lower()

            if choice == 'q':
                return None

            if choice == 'all':
                return 'all'

            idx = int(choice) - 1
            if 0 <= idx < len(models):
                return models[idx]
            else:
                print(f"[ERROR] Please enter a number between 1 and {len(models)}")
        except ValueError:
            print("[ERROR] Please enter a valid number, 'all', or 'q'")
        except KeyboardInterrupt:
            print("\n\nSelection cancelled")
            return None


def select_processing_run_interactive(paths: PathConfig, model_name: str) -> Optional[str]:
    """
    Interactive processing run selection for a specific model.

    Args:
        paths: PathConfig instance
        model_name: Name of the model

    Returns:
        Selected processing run name, 'all' for all runs, 'latest' for most recent,
        or None if cancelled or model uses folders with XML files directly
    """
    # Check if model uses folder structure with XML files directly
    if paths.has_old_structure(model_name):
        return 'direct'  # Special marker for this folder structure

    runs = paths.get_processing_runs(model_name)

    if not runs:
        print(f"[WARNING] No processing runs found for model: {model_name}")
        return None

    print(f"\nAvailable processing runs for '{model_name}':")
    for i, run in enumerate(runs, 1):
        run_dir = paths.get_processing_dir(model_name, run)
        xml_count = len(list(run_dir.glob("*.xml")))
        # Extract timestamp for display
        timestamp = run.replace("processing_", "")
        print(f"   {i}. {timestamp} ({xml_count} XML files)")

    while True:
        try:
            choice = input(f"\nSelect run (1-{len(runs)}, 'all' for all runs, 'latest' for most recent, 'q' to quit): ").strip().lower()

            if choice == 'q':
                return None

            if choice == 'all':
                return 'all'

            if choice == 'latest':
                return runs[0]  # Already sorted newest first

            idx = int(choice) - 1
            if 0 <= idx < len(runs):
                return runs[idx]
            else:
                print(f"[ERROR] Please enter a number between 1 and {len(runs)}")
        except ValueError:
            print("[ERROR] Please enter a valid number, 'all', 'latest', or 'q'")
        except KeyboardInterrupt:
            print("\n\nSelection cancelled")
            return None


def setup_evaluation_paths(paths: Optional[PathConfig] = None,
                           allow_batch: bool = False) -> Optional[tuple]:
    """
    Helper function to interactively select model and processing run,
    then return the appropriate directories for evaluation.

    This eliminates code duplication across all dimension evaluation scripts.

    Args:
        paths: PathConfig instance. If None, uses global config.
        allow_batch: If True, allows 'all' selections. If False, rejects them.

    Returns:
        Tuple of (xml_directory: str, output_directory: str, model_name: str, processing_run: str)
        or None if user cancels or selects 'all' when batch not allowed

    Example:
        >>> paths = get_path_config()
        >>> result = setup_evaluation_paths(paths, allow_batch=False)
        >>> if result:
        >>>     xml_dir, output_dir, model, run = result
        >>>     # Use xml_dir and output_dir for evaluation
    """
    if paths is None:
        paths = get_path_config()

    # Step 1: Select model
    print("\nSelect model for evaluation:")
    model_name = select_model_interactive(paths)

    if model_name is None:
        print("[INFO] No model selected. Exiting.")
        return None

    if model_name == 'all':
        if not allow_batch:
            print("[ERROR] Batch mode 'all' not supported in interactive mode. Please use unified_eval.py")
            return None
        # For batch mode, this would be handled by unified_eval.py
        return None

    # Step 2: Select processing run (if applicable)
    processing_run = None
    if not paths.has_old_structure(model_name):
        processing_run = select_processing_run_interactive(paths, model_name)
        if processing_run is None:
            print("[INFO] No processing run selected. Exiting.")
            return None
        if processing_run == 'all':
            if not allow_batch:
                print("[ERROR] Batch mode 'all' not supported in interactive mode. Please use unified_eval.py")
                return None
            # For batch mode, this would be handled by unified_eval.py
            return None

    # Step 3: Determine directories based on structure
    if processing_run and processing_run != 'direct':
        # Subfolder structure: processing_* subdirectories
        xml_directory = str(paths.get_processing_dir(model_name, processing_run))
        output_directory = str(paths.evaluation_dir / "model" / model_name / processing_run)
    else:
        # Direct structure: XML files in model folder
        xml_directory = str(paths.get_model_output_dir(model_name))
        output_directory = str(paths.get_model_evaluation_dir(model_name))

    # Step 4: Copy processing metadata if available
    copy_processing_metadata(xml_directory, output_directory)

    return (xml_directory, output_directory, model_name, processing_run or 'direct')


def copy_processing_metadata(xml_directory: str, output_directory: str) -> None:
    """
    Copy processing_metadata.json from source processing folder to results folder.

    Checks if metadata file exists in source and copies it to results if not already present.
    This preserves processing information alongside evaluation results.

    Args:
        xml_directory: Source directory containing XML files (e.g., data/output/model/processing_*/
        output_directory: Target results directory (e.g., results/model/model/processing_*/)
    """
    import shutil

    # Build path to source metadata file
    source_metadata = Path(xml_directory) / "log" / "processing_metadata.json"

    # Build path to target metadata file
    target_metadata = Path(output_directory) / "processing_metadata.json"

    # Only copy if source exists and target doesn't exist
    if source_metadata.exists() and not target_metadata.exists():
        # Ensure output directory exists
        Path(output_directory).mkdir(parents=True, exist_ok=True)

        # Copy the file
        shutil.copy2(source_metadata, target_metadata)
        print(f"[INFO] Copied processing metadata to: {target_metadata}")