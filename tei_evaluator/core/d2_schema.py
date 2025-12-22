"""
Dimension 2: Schema Compliance and Standard Usage Evaluation.

This module evaluates XML/TEI files against formal schema definitions and validates
adherence to TEI standards and project-specific requirements.
"""

import logging
from typing import Tuple, List, Optional, Dict, Any
from pathlib import Path
from lxml import etree

from ..models import EvaluationResult, Error, ErrorType
from .base_evaluator import BaseEvaluator
from ..utils.jing_validator import JingValidator
from ..utils.tei_wrapper import TEIWrapper


class D2Schema(BaseEvaluator):
    """
    Dimension 2: Schema Compliance and Standard Usage Evaluation

    Evaluates XML/TEI files against formal schema definitions and validates
    adherence to TEI standards and project-specific requirements.
    """

    def __init__(self,
                 tei_schema_path: Optional[str] = None,
                 project_schema_path: Optional[str] = None,
                 jing_jar_path: Optional[str] = None,
                 auto_wrap_tei: bool = False,
                 reference_directory: Optional[str] = None,
                 quiet: bool = False):
        """
        Initialize D2Schema evaluator.

        Args:
            tei_schema_path: Path to TEI schema file (optional, uses bundled if not provided)
            project_schema_path: Path to project schema file (optional, searches data/schemas/)
            jing_jar_path: Path to Jing validator JAR file (optional, uses bundled if not provided)
            auto_wrap_tei: Automatically wrap non-TEI files with TEI structure
            reference_directory: Directory containing reference files for validation matching
            quiet: If True, suppress print statements (logging still active)
        """
        super().__init__()
        self.dimension = 2
        self.quiet = quiet
        self.logger = logging.getLogger(__name__)

        # Schema paths setup
        self.tei_schema_path = self._setup_schema_path(tei_schema_path, "tei_all.rng")
        self.project_schema_path = self._setup_project_schema_path(project_schema_path)

        # Initialize Jing validator
        self.jing_validator = JingValidator(jing_jar_path)

        # Initialize TEI wrapper
        self.tei_wrapper = TEIWrapper()
        self.auto_wrap_tei = auto_wrap_tei

        # Reference directory setup for validation matching
        self.reference_directory = self._setup_reference_directory(reference_directory)

        # Penalty weights for graduated scoring (1 point per error)
        self.penalty_weights = {
            ErrorType.TEI_SCHEMA_VIOLATION: 1,
            ErrorType.PROJECT_SCHEMA_VIOLATION: 1
        }

    def _setup_schema_path(self, provided_path: Optional[str], default_name: str) -> Optional[Path]:
        """Setup schema file path with fallback to bundled schemas"""
        if provided_path:
            path = Path(provided_path)
            if path.exists():
                return path

        # Look for bundled schema
        package_dir = Path(__file__).parent.parent
        bundled_path = package_dir / "resources" / "schemas" / default_name

        if bundled_path.exists():
            return bundled_path

        return None

    def _setup_project_schema_path(self, provided_path: Optional[str]) -> Optional[Path]:
        """
        Setup project schema path from user data directory.
        Looks for any .rng file in data/schemas/ directory.

        Args:
            provided_path: Explicit path if provided by user

        Returns:
            Path to project schema file if exists, None otherwise
        """
        # If explicit path provided, use it
        if provided_path:
            path = Path(provided_path)
            if path.exists():
                return path

        # Otherwise, look in data/schemas/ for any .rng file
        # Try to find schemas directory relative to project root
        project_root = Path(__file__).parent.parent.parent
        schemas_dir = project_root / "data" / "schemas"
        if schemas_dir.exists():
            rng_files = list(schemas_dir.glob("*.rng"))
            if rng_files:
                return sorted(rng_files)[0]

        return None

    def _setup_reference_directory(self, provided_path: Optional[str]) -> Optional[Path]:
        """
        Setup reference directory with fallback to config paths.

        Priority:
        1. User-provided path
        2. Config path: data/references/

        Returns:
            Path to reference directory or None
        """
        if provided_path:
            path = Path(provided_path)
            if path.exists():
                return path

        # Try to use config path (if available)
        # Note: config is at project root, not in package, so import may fail
        # This is handled gracefully - config is optional
        try:
            from config import get_path_config

            paths = get_path_config()

            # Use data/references/ for reference TEI XML files
            if paths.references_dir.exists():
                return paths.references_dir

        except (ImportError, AttributeError):
            # config module not available - this is OK, fallback to None
            pass

        return None

    def find_reference_file(self, xml_file_path: str) -> Optional[str]:
        """Find corresponding reference file for given XML file"""
        if not self.reference_directory:
            return None

        xml_path = Path(xml_file_path)
        reference_file = self.reference_directory / xml_path.name

        return str(reference_file) if reference_file.exists() else None

    def _prepare_file_for_validation(self, file_path: str, auto_wrap: bool = None, wrap_metadata: Dict[str, Any] = None) -> Tuple[str, bool]:
        """
        Prepare file for validation, wrapping with TEI if needed

        Returns:
            Tuple of (actual_file_path, was_wrapped)
        """

        try:
            # Read file content to check if it's already TEI
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()

            # Check for escaped XML and unescape it FIRST
            if ('&lt;' in content and '&gt;' in content and
                ('&lt;text' in content or '&lt;TEI' in content or '&lt;body' in content)):
                import html
                self.logger.info("Found escaped XML content - unescaping...")
                if not self.quiet:
                    print("   [INFO] Found escaped XML content - unescaping...")
                unescaped_content = html.unescape(content)

                # Create temporary unescaped file instead of overwriting original
                file_path_obj = Path(file_path)
                temp_unescaped_path = file_path_obj.parent / f"{file_path_obj.stem}_unescaped_temp.xml"

                with open(temp_unescaped_path, 'w', encoding='utf-8') as f:
                    f.write(unescaped_content)

                self.logger.info("Created temporary unescaped file")
                self.logger.debug("Unescaped content: %s", unescaped_content[:500] + "..." if len(unescaped_content) > 500 else unescaped_content)
                if not self.quiet:
                    print("   [INFO] Created temporary unescaped file")

                # Check if the unescaped content is TEI XML
                if self.tei_wrapper._is_tei_xml(unescaped_content):
                    self.logger.info("Unescaped content is TEI XML - using temporary file")
                    if not self.quiet:
                        print("   [INFO] Unescaped content is TEI XML - using temporary file")
                    return str(temp_unescaped_path), True  # Return temp file, mark as "wrapped" for cleanup

                # If not TEI, continue with wrapping the unescaped content
                content = unescaped_content
                file_path = str(temp_unescaped_path)  # Use temp file for further processing

            # Check if it's already TEI XML (after potential unescaping)
            if self.tei_wrapper._is_tei_xml(content):
                return file_path, False

            # File needs wrapping
            self.logger.info("File needs wrapping - proceeding...")
            if not self.quiet:
                print("   [INFO] File needs wrapping - proceeding...")
            file_path_obj = Path(file_path)

            # Set up metadata
            metadata = wrap_metadata or {}
            if 'title' not in metadata:
                metadata['title'] = file_path_obj.stem.replace('_', ' ').title()

            # Create wrapped file in temp location
            wrapped_file_path = file_path_obj.parent / f"{file_path_obj.stem}_tei_temp.xml"

            actual_file_path = self.tei_wrapper.process_file(
                file_path,
                str(wrapped_file_path),
                metadata=metadata
            )

            return actual_file_path, True

        except (IOError, OSError, ValueError, RuntimeError) as e:
            # If wrapping fails, continue with original file
            self.logger.warning("Could not wrap file with TEI: %s", str(e))
            if not self.quiet:
                print(f"Warning: Could not wrap file with TEI: {e}")
            return file_path, False


    def validate_tei_schema(self, file_path: str, quick_only: bool = False, auto_wrap: bool = None, wrap_metadata: Dict[str, Any] = None) -> Tuple[bool, List[Error], Dict[str, int]]:
        """
        Validate against TEI schema, with optional automatic TEI wrapping

        Returns:
            Tuple of (is_valid, errors_list, jing_categories_count)
        """
        if not self.tei_schema_path:
            return True, [], {}

        # ADDED: Prepare file (potentially wrap with TEI)
        actual_file_path, was_wrapped = self._prepare_file_for_validation(file_path, auto_wrap, wrap_metadata)

        try:
            if quick_only:
                is_valid, error_msg = self.jing_validator.validate_quick(actual_file_path, str(self.tei_schema_path))
                if is_valid:
                    return True, [], {}
                else:
                    # Create a basic error for quick validation failure
                    error = Error(
                        type=ErrorType.TEI_SCHEMA_VIOLATION,
                        severity=7,
                        location="Unknown",
                        message=f"TEI schema validation failed: {error_msg}",
                        raw_error=error_msg
                    )
                    return False, [error], {"unknown": 1}

            # Detailed validation
            is_valid, errors = self.jing_validator.validate_detailed(actual_file_path, str(self.tei_schema_path), "TEI")

            # Count Jing categories - handle missing jing_category attribute gracefully
            jing_categories = {}
            for error in errors:
                # Try to get jing_category, fall back to parsing from message
                try:
                    category = getattr(error, 'jing_category', None) or "other"
                except AttributeError:
                    category = "other"

                # If jing_category is None, try to extract from message
                if category == "other" and hasattr(error, 'message'):
                    if "element" in error.message.lower() and "not allowed" in error.message.lower():
                        category = "element_not_allowed"
                    elif "missing" in error.message.lower() or "required" in error.message.lower():
                        category = "missing_required_element"
                    elif "attribute" in error.message.lower():
                        category = "invalid_attribute"
                    elif "content" in error.message.lower():
                        category = "content_model_violation"

                jing_categories[category] = jing_categories.get(category, 0) + 1

            return is_valid, errors, jing_categories

        finally:
            if was_wrapped and actual_file_path != file_path:
                try:
                    Path(actual_file_path).unlink()
                except (OSError, PermissionError):
                    pass  # Ignore cleanup errors

    def validate_project_schema(self, file_path: str, quick_only: bool = False, auto_wrap: bool = None, wrap_metadata: Dict[str, Any] = None) -> Tuple[bool, List[Error], Dict[str, int]]:
        """
        Validate against project schema, with optional automatic TEI wrapping

        Returns:
            Tuple of (is_valid, errors_list, jing_categories_count)
        """
        if not self.project_schema_path:
            return True, [], {}

        # ADDED: Prepare file (potentially wrap with TEI)
        actual_file_path, was_wrapped = self._prepare_file_for_validation(file_path, auto_wrap, wrap_metadata)

        try:
            if quick_only:
                is_valid, error_msg = self.jing_validator.validate_quick(actual_file_path, str(self.project_schema_path))
                if is_valid:
                    return True, [], {}
                else:
                    # Create a basic error for quick validation failure
                    error = Error(
                        type=ErrorType.PROJECT_SCHEMA_VIOLATION,
                        severity=5,
                        location="Unknown",
                        message=f"Project schema validation failed: {error_msg}",
                        raw_error=error_msg
                    )
                    return False, [error], {"unknown": 1}

            # Detailed validation
            is_valid, errors = self.jing_validator.validate_detailed(actual_file_path, str(self.project_schema_path), "Project")

            # Count Jing categories - handle missing jing_category attribute gracefully
            jing_categories = {}
            for error in errors:
                # Try to get jing_category, fall back to parsing from message
                try:
                    category = getattr(error, 'jing_category', None) or "other"
                except AttributeError:
                    category = "other"

                # If jing_category is None, try to extract from message
                if category == "other" and hasattr(error, 'message'):
                    if "element" in error.message.lower() and "not allowed" in error.message.lower():
                        category = "element_not_allowed"
                    elif "missing" in error.message.lower() or "required" in error.message.lower():
                        category = "missing_required_element"
                    elif "attribute" in error.message.lower():
                        category = "invalid_attribute"
                    elif "content" in error.message.lower():
                        category = "content_model_violation"

                jing_categories[category] = jing_categories.get(category, 0) + 1

            return is_valid, errors, jing_categories

        finally:
            if was_wrapped and actual_file_path != file_path:
                try:
                    Path(actual_file_path).unlink()
                except (OSError, PermissionError):
                    pass  # Ignore cleanup errors

    def _validate_reference_file(self, file_path: str, validate_tei: bool, validate_project: bool) -> Dict[str, Any]:
        """
        Validate the reference file (if available) and compare with LLM output validation results

        Returns:
            Dictionary with reference validation metrics:
            - reference_file: path to reference file or None
            - tei_validity_reference: True/False/None (None if no reference or TEI validation disabled)
            - tei_validity_match: True/False/None
            - project_validity_reference: True/False/None
            - project_validity_match: True/False/None
        """
        result = {
            "reference_file": None,
            "tei_validity_reference": None,
            "tei_validity_match": None,
            "project_validity_reference": None,
            "project_validity_match": None
        }

        # Find reference file
        reference_file = self.find_reference_file(file_path)
        if not reference_file:
            return result

        result["reference_file"] = reference_file

        try:
            # Validate reference against TEI schema if TEI validation is enabled
            if validate_tei and self.tei_schema_path:
                ref_tei_valid, _, _ = self.validate_tei_schema(reference_file, quick_only=True)
                result["tei_validity_reference"] = ref_tei_valid

            # Validate reference against project schema if project validation is enabled
            if validate_project and self.project_schema_path:
                ref_project_valid, _, _ = self.validate_project_schema(reference_file, quick_only=True)
                result["project_validity_reference"] = ref_project_valid

        except (IOError, OSError, RuntimeError, ValueError) as e:
            # If reference validation fails, log but don't fail the whole evaluation
            self.logger.warning("Could not validate reference file: %s", str(e))
            if not self.quiet:
                print(f"   [WARNING] Could not validate reference file: {e}")

        return result

    def calculate_score(self, errors: List[Error], base_score: float = 100.0) -> float:
        """Calculate overall graduated penalty score"""
        total_penalty = 0

        # Count errors by type
        error_counts = {}
        for error in errors:
            error_counts[error.type] = error_counts.get(error.type, 0) + 1

        # Apply penalties per error type
        for error_type, count in error_counts.items():
            penalty_per_error = self.penalty_weights.get(error_type, 4)
            total_penalty += penalty_per_error * count

        # Cap maximum penalty at base_score
        total_penalty = min(total_penalty, base_score)
        final_score = max(0, base_score - total_penalty)
        return final_score

    def calculate_component_score(self, errors: List[Error], error_types: List[ErrorType], base_score: float = 100.0) -> float:
        """Calculate score for specific component (TEI or Project)"""
        component_errors = [e for e in errors if e.type in error_types]
        return self.calculate_score(component_errors, base_score)

    def evaluate_file(self, file_path: str, config: Optional[Dict[str, Any]] = None) -> EvaluationResult:
        """
        Evaluate a single XML file with configurable validation modes

        Config options:
        - validate_tei: bool (default True)
        - validate_project: bool (default True)
        - quick_validation_first: bool (default True)
        - always_detailed: bool (default False)
        - auto_wrap_tei: bool (default False) - ADDED: automatically wrap non-TEI files
        - wrap_metadata: dict (default {}) - ADDED: metadata for TEI wrapping
        """
        config = config or {}
        all_errors = []
        file_path = Path(file_path)

        # Configuration defaults
        validate_tei = config.get('validate_tei', True)
        validate_project = config.get('validate_project', True)
        quick_validation_first = config.get('quick_validation_first', True)
        always_detailed = config.get('always_detailed', False)

        # ADDED: Wrapping configuration
        auto_wrap = config.get('auto_wrap_tei', self.auto_wrap_tei)
        wrap_metadata = config.get('wrap_metadata', {})

        # Check if file exists
        if not file_path.exists():
            error = Error(
                type=ErrorType.TEI_SCHEMA_VIOLATION,
                severity=10,
                location="File",
                message=f"File not found: {file_path}",
                raw_error=""
            )
            all_errors.append(error)

            return EvaluationResult(
                dimension=2,
                passed=False,
                score=0.0,
                errors=all_errors,
                metrics={"file_readable": False}
            )

        try:
            # Parse XML for validation
            xml_tree = None
            if auto_wrap:
                eval_file_path, was_wrapped = self._prepare_file_for_validation(str(file_path), auto_wrap, wrap_metadata)
                try:
                    xml_tree = etree.parse(eval_file_path)
                finally:
                    # Clean up if wrapped
                    if was_wrapped and eval_file_path != str(file_path):
                        try:
                            Path(eval_file_path).unlink()
                        except (OSError, PermissionError):
                            pass  # Ignore cleanup errors
            else:
                xml_tree = etree.parse(str(file_path))

            # Initialize validation results
            tei_validation = {"enabled": validate_tei, "valid": True, "error_count": 0, "jing_categories": {}}
            project_validation = {"enabled": validate_project, "valid": True, "error_count": 0, "jing_categories": {}}

            # PHASE 1: Quick validation (if enabled)
            if quick_validation_first and not always_detailed:
                self.logger.info("Quick validation...")
                if not self.quiet:
                    print("   [INFO] Quick validation...")

                quick_passed = True

                if validate_tei:
                    # MODIFIED: Pass wrapping parameters
                    tei_valid, tei_errors, tei_cats = self.validate_tei_schema(str(file_path), quick_only=True, auto_wrap=auto_wrap, wrap_metadata=wrap_metadata)
                    tei_validation["valid"] = tei_valid
                    if not tei_valid:
                        all_errors.extend(tei_errors)
                        tei_validation["error_count"] = len(tei_errors)
                        tei_validation["jing_categories"] = tei_cats
                        quick_passed = False

                if validate_project:
                    # MODIFIED: Pass wrapping parameters
                    project_valid, project_errors, project_cats = self.validate_project_schema(str(file_path), quick_only=True, auto_wrap=auto_wrap, wrap_metadata=wrap_metadata)
                    project_validation["valid"] = project_valid
                    if not project_valid:
                        all_errors.extend(project_errors)
                        project_validation["error_count"] = len(project_errors)
                        project_validation["jing_categories"] = project_cats
                        quick_passed = False

                # If quick validation passed and we're not forcing detailed analysis
                if quick_passed and not always_detailed:
                    self.logger.info("Quick validation passed - skipping detailed analysis")
                    if not self.quiet:
                        print("   [INFO] Quick validation passed - skipping detailed analysis")

                    overall_passed = True
                    overall_score = 100.0
                    # Determine evaluation mode string for reporter
                    if validate_project and not validate_tei:
                        eval_mode_str = 'project_validation_only'
                    elif validate_tei and not validate_project:
                        eval_mode_str = 'tei_validation_only'
                    else:
                        eval_mode_str = 'combined'

                    # Calculate component scores for quick validation
                    tei_score = 100.0 if tei_validation["valid"] else 0.0
                    project_score = 100.0 if project_validation["valid"] else 0.0

                    # ADDED: Validate reference file if available
                    reference_validation = self._validate_reference_file(str(file_path), validate_tei, validate_project)

                    # ADDED: Calculate validity matches
                    if reference_validation["tei_validity_reference"] is not None:
                        reference_validation["tei_validity_match"] = (
                            tei_validation["valid"] == reference_validation["tei_validity_reference"]
                        )
                    if reference_validation["project_validity_reference"] is not None:
                        reference_validation["project_validity_match"] = (
                            project_validation["valid"] == reference_validation["project_validity_reference"]
                        )

                    metrics = {
                        "file_readable": True,
                        "validation_mode": "quick_only",
                        # Include config so reporter can detect mode
                        "validation_config": {
                            "tei_enabled": validate_tei,
                            "project_enabled": validate_project
                        },
                        # Explicit reporter-facing evaluation mode
                        "evaluation_mode": eval_mode_str,
                        "overall_score": overall_score,
                        "tei_score": tei_score,
                        "project_score": project_score,
                        "tei_validation": tei_validation,
                        "project_validation": project_validation,
                        "tei_schema_valid": tei_validation["valid"],
                        "project_schema_valid": project_validation["valid"],
                        "tei_errors": tei_validation["error_count"],
                        "project_errors": project_validation["error_count"],
                        # ADDED: Reference validation metrics
                        **reference_validation
                    }

                    return EvaluationResult(
                        dimension=2,
                        passed=overall_passed,
                        score=overall_score,
                        errors=all_errors,
                        metrics=metrics
                    )

            # PHASE 2: Detailed validation
            self.logger.info("Detailed validation...")
            if not self.quiet:
                print("   [INFO] Detailed validation...")

            # Reset for detailed analysis
            all_errors = []

            # TEI Schema Validation
            if validate_tei:
                tei_valid, tei_errors, tei_cats = self.validate_tei_schema(str(file_path), quick_only=True)
                all_errors.extend(tei_errors)
                tei_validation.update({
                    "valid": tei_valid,
                    "error_count": len(tei_errors),
                    "jing_categories": tei_cats
                })

            # Project Schema Validation
            if validate_project:
                project_valid, project_errors, project_cats = self.validate_project_schema(str(file_path), quick_only=True)
                all_errors.extend(project_errors)
                project_validation.update({
                    "valid": project_valid,
                    "error_count": len(project_errors),
                    "jing_categories": project_cats
                })

            # Calculate scores
            overall_score = self.calculate_score(all_errors)

            tei_score = 100.0
            if validate_tei:
                tei_score = self.calculate_component_score(all_errors, [ErrorType.TEI_SCHEMA_VIOLATION])

            project_score = 100.0
            if validate_project:
                project_score = self.calculate_component_score(all_errors, [ErrorType.PROJECT_SCHEMA_VIOLATION])

            # Overall pass/fail
            overall_passed = (
                tei_validation["valid"] and
                project_validation["valid"]
            )

            # ADDED: Validate reference file if available
            reference_validation = self._validate_reference_file(str(file_path), validate_tei, validate_project)

            # ADDED: Calculate validity matches
            if reference_validation["tei_validity_reference"] is not None:
                reference_validation["tei_validity_match"] = (
                    tei_validation["valid"] == reference_validation["tei_validity_reference"]
                )
            if reference_validation["project_validity_reference"] is not None:
                reference_validation["project_validity_match"] = (
                    project_validation["valid"] == reference_validation["project_validity_reference"]
                )

            # Compile metrics
            if validate_project and not validate_tei:
                eval_mode_str = 'project_validation_only'
            elif validate_tei and not validate_project:
                eval_mode_str = 'tei_validation_only'
            else:
                eval_mode_str = 'combined'

            metrics = {
                "file_readable": True,
                "validation_mode": "detailed",
                # Explicit reporter-facing evaluation mode
                "evaluation_mode": eval_mode_str,
                "overall_score": overall_score,
                "tei_score": tei_score,
                "project_score": project_score,
                "total_errors": len(all_errors),
                "tei_validation": tei_validation,
                "project_validation": project_validation,
                "tei_schema_valid": tei_validation["valid"],
                "project_schema_valid": project_validation["valid"],
                "tei_errors": tei_validation["error_count"],
                "project_errors": project_validation["error_count"],
                "error_breakdown": {
                    error_type.value: len([e for e in all_errors if e.type == error_type])
                    for error_type in ErrorType
                },
                "validation_config": {
                    "tei_enabled": validate_tei,
                    "project_enabled": validate_project,
                    "auto_wrap_tei": auto_wrap,
                    "tei_schema_available": self.tei_schema_path is not None,
                    "project_schema_available": self.project_schema_path is not None,
                    "jing_validator_available": self.jing_validator.is_available()
                },
                # ADDED: Reference validation metrics
                **reference_validation
            }

            return EvaluationResult(
                dimension=2,
                passed=overall_passed,
                score=overall_score,
                errors=all_errors,
                metrics=metrics
            )

        except etree.XMLSyntaxError as e:
            # File is not well-formed XML
            error = Error(
                type=ErrorType.TEI_SCHEMA_VIOLATION,
                severity=10,
                location=f"Line {e.lineno}" if hasattr(e, 'lineno') else "Unknown",
                message="XML syntax error - file should pass Dimension 1 first",
                raw_error=str(e)
            )
            all_errors.append(error)

            return EvaluationResult(
                dimension=2,
                passed=False,
                score=0.0,
                errors=all_errors,
                metrics={
                    "file_readable": False,
                    "xml_parse_error": True
                }
            )
        except ImportError as e:
            # lxml is not available
            error = Error(
                type=ErrorType.TEI_SCHEMA_VIOLATION,
                severity=8,
                location="File",
                message="lxml library not available for XML parsing",
                raw_error=str(e)
            )
            all_errors.append(error)

            return EvaluationResult(
                dimension=2,
                passed=False,
                score=0.0,
                errors=all_errors,
                metrics={
                    "file_readable": False,
                    "lxml_not_available": True
                }
            )

        except (IOError, OSError, RuntimeError, ValueError) as e:
            error = Error(
                type=ErrorType.TEI_SCHEMA_VIOLATION,
                severity=8,
                location="File",
                message=f"Error during Dimension 2 evaluation: {str(e)}",
                raw_error=str(e)
            )
            all_errors.append(error)

            return EvaluationResult(
                dimension=2,
                passed=False,
                score=0.0,
                errors=all_errors,
                metrics={"evaluation_error": True}
            )

    def evaluate_file_with_auto_wrap(self, file_path: str, wrap_metadata: Optional[Dict[str, Any]] = None, **config_kwargs) -> EvaluationResult:
        """
        Convenience method to evaluate a file with automatic TEI wrapping enabled
        """
        config = config_kwargs.copy()
        config['auto_wrap_tei'] = True
        if wrap_metadata:
            config['wrap_metadata'] = wrap_metadata

        return self.evaluate_file(file_path, config)

    def evaluate_tei_only(self, xml_path: str, pattern: str = "*.xml") -> List[EvaluationResult]:
        """
        Evaluate only TEI schema compliance.

        This mode validates XML files against the TEI P5 standard schema only,
        without checking project-specific requirements.

        Args:
            xml_path: Directory path containing XML files or single XML file path
            pattern: Glob pattern to match files (default: "*.xml")

        Returns:
            List of EvaluationResult objects containing:
                - passed: Boolean indicating TEI schema validity
                - score: TEI quality score (100 - error penalties)
                - errors: List of TEI schema violation errors
                - metrics: TEI-specific metrics (tei_valid, tei_score, error_breakdown)

        Performance:
            Valid files (quick): ~50-100ms per file
            Invalid files (detailed): ~100-300ms per file
        """
        # Validate TEI schema is available
        if not self.tei_schema_path or not self.tei_schema_path.exists():
            self.logger.error("TEI schema not found at expected location: %s", self.tei_schema_path)
            if not self.quiet:
                print("[ERROR] TEI schema not found at expected location")
                print(f"        Expected: {self.tei_schema_path}")
                print("        Cannot proceed with TEI validation")
            return []

        config = {
            'validate_tei': True,
            'validate_project': False,
            'quick_validation_first': True,
            'always_detailed': False
        }
        return self.evaluate_batch(xml_path, pattern, config)

    def evaluate_project_only(self, xml_path: str, pattern: str = "*.xml") -> List[EvaluationResult]:
        """
        Evaluate only project schema compliance.

        This mode validates XML files against project-specific schema requirements only,
        without checking TEI P5 standard compliance.

        Args:
            xml_path: Directory path containing XML files or single XML file path
            pattern: Glob pattern to match files (default: "*.xml")

        Returns:
            List of EvaluationResult objects containing:
                - passed: Boolean indicating project schema validity
                - score: Project quality score (100 - error penalties)
                - errors: List of project schema violation errors
                - metrics: Project-specific metrics (project_valid, project_score, error_breakdown)

        Performance:
            Valid files (quick): ~50-100ms per file
            Invalid files (detailed): ~100-300ms per file
        """
        # Validate project schema is available
        if not self.project_schema_path or not self.project_schema_path.exists():
            self.logger.error("Project schema not found")
            if not self.quiet:
                print("[ERROR] Project schema not found")
                if self.project_schema_path:
                    print(f"        Expected location: {self.project_schema_path}")
                else:
                    print("        Expected location: data/schemas/*.rng (any RNG file)")
                print("        Please provide your project schema file")
                print("        See DATA_STRUCTURE.md for setup instructions")
            return []

        config = {
            'validate_tei': False,
            'validate_project': True,
            'quick_validation_first': True,
            'always_detailed': False
        }
        return self.evaluate_batch(xml_path, pattern, config)

    def evaluate_full(self, xml_path: str, pattern: str = "*.xml") -> List[EvaluationResult]:
        """
        Full evaluation: TEI schema + Project schema compliance.

        This mode performs comprehensive validation against both TEI P5 standard
        and project-specific schema requirements.

        Evaluation Steps:
        1. TEI Schema Validation:
           - Validates against TEI P5 RelaxNG schema
           - Checks standard TEI element usage
           - Calculates TEI quality score

        2. Project Schema Validation:
           - Validates against project RelaxNG schema
           - Checks project-specific requirements
           - Calculates project quality score

        3. Combined Assessment:
           - Separate scores for TEI and Project
           - Comprehensive error reporting for both
           - Pass requires BOTH validations to succeed

        Args:
            xml_path: Directory path containing XML files or single XML file path
            pattern: Glob pattern to match files (default: "*.xml")

        Returns:
            List of EvaluationResult objects containing:
                - passed: Boolean (True only if BOTH TEI and Project valid)
                - score: Overall score (combined penalties)
                - errors: Combined list of TEI and Project errors
                - metrics: Complete metrics including:
                    - tei_score: TEI quality score (0-100)
                    - project_score: Project quality score (0-100)
                    - tei_schema_valid: Boolean TEI validity
                    - project_schema_valid: Boolean Project validity
                    - error_breakdown: Errors by category

        Performance:
            Valid files (quick): ~100-200ms per file (both quick validations)
            Invalid files (detailed): ~200-600ms per file (both detailed validations)
        """
        # Validate TEI schema (required)
        if not self.tei_schema_path or not self.tei_schema_path.exists():
            self.logger.error("TEI schema not found at expected location: %s", self.tei_schema_path)
            if not self.quiet:
                print("[ERROR] TEI schema not found at expected location")
                print(f"        Expected: {self.tei_schema_path}")
                print("[ERROR] Cannot proceed with validation")
            return []

        # Check project schema availability
        if not self.project_schema_path or not self.project_schema_path.exists():
            self.logger.warning("Project schema not found")
            if not self.quiet:
                print("[WARNING] Project schema not found")
                print("          Expected location: data/schemas/*.rng (any RNG file)")
                print("          Full validation requires both TEI and Project schemas")
                print()

                response = input("Proceed with TEI validation only? (y/n): ").strip().lower()

                if response in ['y', 'yes']:
                    print("[INFO] Proceeding with TEI validation only")
                    return self.evaluate_tei_only(xml_path, pattern)
                else:
                    print("[INFO] Evaluation cancelled")
                    return []
            else:
                # In quiet mode, automatically proceed with TEI only
                self.logger.info("Proceeding with TEI validation only (quiet mode)")
                return self.evaluate_tei_only(xml_path, pattern)

        # Both schemas available - proceed with full validation
        config = {
            'validate_tei': True,
            'validate_project': True,
            'quick_validation_first': True,
            'always_detailed': False
        }
        return self.evaluate_batch(xml_path, pattern, config)

    def evaluate_batch(self, input_path: str, pattern: str = "*.xml", config: Optional[Dict[str, Any]] = None) -> List[EvaluationResult]:
        """
        Evaluate multiple TEI files with configurable validation modes
        """
        config = config or {}
        input_path = Path(input_path)
        results = []

        if not input_path.exists():
            self.logger.error("Directory not found: %s", input_path)
            if not self.quiet:
                print(f"[ERROR] Directory not found: {input_path}")
            return results

        # Find all matching files
        if input_path.is_file():
            xml_files = [input_path]
        else:
            xml_files = list(input_path.glob(pattern))

        if not xml_files:
            self.logger.error("No XML files found in %s with pattern '%s'", input_path, pattern)
            if not self.quiet:
                print(f"[ERROR] No XML files found in {input_path} with pattern '{pattern}'")
            return results

        # Display configuration
        validate_tei = config.get('validate_tei', True)
        validate_project = config.get('validate_project', True)
        auto_wrap = config.get('auto_wrap_tei', self.auto_wrap_tei)

        self.logger.info("Found %d XML files to evaluate", len(xml_files))
        if not self.quiet:
            print(f"Found {len(xml_files)} XML files to evaluate")
            print("Validation Configuration:")
            print(f"   TEI Schema: {'Enabled' if validate_tei else 'Disabled'}")
            print(f"   Project Schema: {'Enabled' if validate_project else 'Disabled'}")
            print(f"   Auto TEI Wrap: {'Enabled' if auto_wrap else 'Disabled'}")

            # Display available resources
            validator_info = self.jing_validator.get_validator_info()
            print("Available Resources:")
            print(f"   Jing validator: {'Available' if validator_info['jing_available'] else 'Not found'}")
            print(f"   Java: {'Available' if validator_info['java_available'] else 'Not found'}")
            print(f"   TEI Schema: {'Available' if self.tei_schema_path else 'Not found'}")
            print(f"   Project Schema: {'Available' if self.project_schema_path else 'Not found'}")
            print("=" * 60)

        for xml_file in sorted(xml_files):
            self.logger.info("Evaluating: %s", xml_file.name)
            if not self.quiet:
                print(f"Evaluating: {xml_file.name}")
            try:
                result = self.evaluate_file(str(xml_file), config)
                # Store the file path in metrics for later reference
                result.metrics['file_path'] = str(xml_file)
                result.metrics['file_name'] = xml_file.name
                results.append(result)

                # Quick summary for each file
                tei_validation = result.metrics.get('tei_validation', {})
                project_validation = result.metrics.get('project_validation', {})

                # Individual component status
                tei_status = "[VALID]" if tei_validation.get('valid', True) else "[INVALID]"
                project_status = "[VALID]" if project_validation.get('valid', True) else "[INVALID]"

                if not self.quiet:
                    if result.metrics.get('validation_mode') == 'quick_only':
                        print(f"   {tei_status} TEI | {project_status} Project | Quick validation only")
                    else:
                        # Detailed validation summary with individual statuses
                        summary_parts = []

                        if validate_tei:
                            tei_score = result.metrics.get('tei_score', 0)
                            summary_parts.append(f"{tei_status} TEI: {tei_score:5.1f}")

                        if validate_project:
                            project_score = result.metrics.get('project_score', 0)
                            summary_parts.append(f"{project_status} Project: {project_score:5.1f}")

                        summary_parts.append(f"Errors: {len(result.errors)}")

                        print(f"   {' | '.join(summary_parts)}")

            except (IOError, OSError, RuntimeError, ValueError) as e:
                self.logger.error("Error evaluating file: %s", str(e), exc_info=True)
                if not self.quiet:
                    print(f"   [ERROR]: {str(e)}")
                # Create error result for failed evaluation
                error_result = EvaluationResult(
                    dimension=2,
                    passed=False,
                    score=0.0,
                    errors=[Error(
                        type=ErrorType.TEI_SCHEMA_VIOLATION,
                        severity=10,
                        location="File",
                        message=f"Evaluation failed: {str(e)}",
                        raw_error=str(e)
                    )],
                    metrics={
                        "file_readable": False,
                        "evaluation_error": True,
                        "file_path": str(xml_file),
                        "file_name": xml_file.name
                    }
                )
                results.append(error_result)

        return results


    def generate_batch_summary(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Generate summary statistics for configurable batch evaluation results"""
        if not results:
            return {"total_files": 0, "message": "No results to summarize"}

        total_files = len(results)
        passed_files = sum(1 for r in results if r.passed)
        failed_files = total_files - passed_files

        scores = [r.score for r in results]
        avg_score = sum(scores) / len(scores) if scores else 0

        # Component-specific scores
        tei_scores = [r.metrics.get('tei_score', 0) for r in results if 'tei_score' in r.metrics]
        project_scores = [r.metrics.get('project_score', 0) for r in results if 'project_score' in r.metrics]

        avg_tei_score = sum(tei_scores) / len(tei_scores) if tei_scores else 0
        avg_project_score = sum(project_scores) / len(project_scores) if project_scores else 0

        # Validation mode statistics
        quick_only_count = sum(1 for r in results if r.metrics.get('validation_mode') == 'quick_only')
        detailed_count = sum(1 for r in results if r.metrics.get('validation_mode') == 'detailed')

        # FIXED: Component validation statistics using the correct fields
        # Count enabled files by checking if the validation section exists and has enabled=True
        tei_enabled_count = sum(1 for r in results if r.metrics.get('tei_validation', {}).get('enabled', False))
        project_enabled_count = sum(1 for r in results if r.metrics.get('project_validation', {}).get('enabled', False))

        # Component success rates using the correct fields
        tei_valid_count = sum(1 for r in results if r.metrics.get('tei_validation', {}).get('valid', False))
        project_valid_count = sum(1 for r in results if r.metrics.get('project_validation', {}).get('valid', False))

        # Error type breakdown
        all_errors = []
        for result in results:
            all_errors.extend(result.errors)

        error_breakdown = {}
        for error_type in ErrorType:
            count = len([e for e in all_errors if e.type == error_type])
            if count > 0:
                error_breakdown[error_type.value] = count

        # Jing category breakdown
        tei_jing_categories = {}
        project_jing_categories = {}
        bp_categories = {}

        for result in results:
            # TEI Jing categories
            tei_cats = result.metrics.get('tei_validation', {}).get('jing_categories', {})
            for cat, count in tei_cats.items():
                tei_jing_categories[cat] = tei_jing_categories.get(cat, 0) + count

            # Project Jing categories
            project_cats = result.metrics.get('project_validation', {}).get('jing_categories', {})
            for cat, count in project_cats.items():
                project_jing_categories[cat] = project_jing_categories.get(cat, 0) + count

        summary = {
            "total_files": total_files,
            "passed_files": passed_files,
            "failed_files": failed_files,
            "pass_rate": (passed_files / total_files * 100) if total_files > 0 else 0,
            "average_score": avg_score,
            "min_score": min(scores) if scores else 0,
            "max_score": max(scores) if scores else 0,

            # Component scores
            "average_tei_score": avg_tei_score,
            "average_project_score": avg_project_score,

            # Validation modes
            "quick_validation_only": quick_only_count,
            "detailed_validation": detailed_count,

            # FIXED: Component statistics using correct counts
            "tei_enabled_files": tei_enabled_count,
            "project_enabled_files": project_enabled_count,

            "tei_valid_files": tei_valid_count,
            "project_valid_files": project_valid_count,

            # Component success rates with proper calculation
            "tei_valid_rate": (tei_valid_count / tei_enabled_count * 100) if tei_enabled_count > 0 else 0,
            "project_valid_rate": (project_valid_count / project_enabled_count * 100) if project_enabled_count > 0 else 0,

            # Error analysis
            "total_errors": len(all_errors),
            "error_breakdown": error_breakdown,
            "tei_jing_categories": tei_jing_categories,
            "project_jing_categories": project_jing_categories,

            "detailed_results": [
                {
                    "file_name": result.metrics.get('file_name', f"File_{i+1}"),
                    "file_path": result.metrics.get('file_path', ''),
                    "passed": result.passed,
                    "score": result.score,
                    "validation_mode": result.metrics.get('validation_mode', 'unknown'),
                    "tei_score": result.metrics.get('tei_score', 0),
                    "project_score": result.metrics.get('project_score', 0),
                    "tei_valid": result.metrics.get('tei_validation', {}).get('valid', False),
                    "project_valid": result.metrics.get('project_validation', {}).get('valid', False),
                    "error_count": len(result.errors)
                }
                for i, result in enumerate(results)
            ]
        }

        return summary

    def print_batch_summary(self, results: List[EvaluationResult]):
        """
        Print a formatted summary of configurable batch evaluation results.

        Args:
            results: List of evaluation results to summarize
        """
        if self.quiet:
            return

        summary = self.generate_batch_summary(results)

        # Handle empty results
        if summary.get('total_files', 0) == 0:
            print("\n" + "="*70)
            print("DIMENSION 2 SCHEMA COMPLIANCE EVALUATION SUMMARY")
            print("="*70)
            print(f"[INFO] {summary.get('message', 'No results to summarize')}")
            print("="*70)
            return

        print("\n" + "="*70)
        print("DIMENSION 2 SCHEMA COMPLIANCE EVALUATION SUMMARY")
        print("="*70)
        print(f"Total Files: {summary['total_files']}")
        print(f"Passed: {summary['passed_files']} ({summary['pass_rate']:.1f}%)")
        print(f"Failed: {summary['failed_files']}")
        print(f"Average Overall Score: {summary['average_score']:.1f}/100")
        print(f"Score Range: {summary['min_score']:.1f} - {summary['max_score']:.1f}")

        # Validation mode breakdown
        print("\nVALIDATION MODES:")
        print(f"Quick validation only: {summary['quick_validation_only']} files")
        print(f"Detailed validation: {summary['detailed_validation']} files")

        # Component breakdown
        print("\nCOMPONENT ANALYSIS:")
        if summary['tei_enabled_files'] > 0:
            print(f"TEI Schema: {summary['tei_valid_files']}/{summary['tei_enabled_files']} valid ({summary['tei_valid_rate']:.1f}%) | Avg Score: {summary['average_tei_score']:.1f}")
        if summary['project_enabled_files'] > 0:
            print(f"Project Schema: {summary['project_valid_files']}/{summary['project_enabled_files']} valid ({summary['project_valid_rate']:.1f}%) | Avg Score: {summary['average_project_score']:.1f}")

        print(f"Total Errors: {summary['total_errors']}")

        # Jing categories breakdown
        if summary['tei_jing_categories']:
            print("\nTEI Jing Error Categories:")
            for category, count in summary['tei_jing_categories'].items():
                print(f"  - {category}: {count}")

        if summary['project_jing_categories']:
            print("\nProject Jing Error Categories:")
            for category, count in summary['project_jing_categories'].items():
                print(f"  - {category}: {count}")

        print("\nFile Details:")
        for detail in summary['detailed_results']:
            # Get individual component statuses
            tei_status = "[VALID]" if detail.get('tei_valid', True) else "[INVALID]"
            project_status = "[VALID]" if detail.get('project_valid', True) else "[INVALID]"

            mode_indicator = "Q" if detail['validation_mode'] == 'quick_only' else "D"

            if detail['validation_mode'] == 'quick_only':
                print(f"  [{mode_indicator}] {detail['file_name']:<25} {tei_status} TEI | {project_status} Project | Quick validation only")
            else:
                print(f"  [{mode_indicator}] {detail['file_name']:<20} {tei_status} TEI: {detail['tei_score']:5.1f} | {project_status} Project: {detail['project_score']:5.1f} | Errors: {detail['error_count']}")

        print("="*70)