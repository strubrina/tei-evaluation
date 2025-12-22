"""
Dimension 0: XML Well-formedness Evaluation.

This module evaluates the syntactic quality and well-formedness of XML/TEI output
from LLM-based encoding tasks.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..models import Error, ErrorType, EvaluationResult
from ..utils.xml_analysis import EnhancedXMLAnalyzer
from .base_evaluator import BaseEvaluator

class D0Syntactic(BaseEvaluator):
    """
    Dimension 0: XML Well-formedness Evaluation.

    Evaluates the syntactic quality and well-formedness of XML/TEI output from
    LLM-based encoding tasks. This dimension focuses exclusively on XML structural
    correctness:

    - Validates XML wellformedness using lxml parser
    - Checks tag nesting, closure, and syntax
    - Detects character encoding and escaping issues
    - Validates attribute syntax
    - Verifies document structure requirements

    Scoring:
        XML Score: 100 - (number of errors) - placeholder for future refinement
        Pass/Fail: Based on XML well-formedness

    Attributes:
        dimension: Evaluation dimension number (0)
        xml_analyzer: EnhancedXMLAnalyzer instance for XML analysis
        penalty_weights: Error type to penalty weight mapping
        logger: Logger instance for this evaluator
    """

    def __init__(self, quiet: bool = False):
        """
        Initialize the D0 Syntactic evaluator.

        Args:
            quiet: If True, suppress console output
        """
        super().__init__()
        self.dimension = 0
        self.xml_analyzer = EnhancedXMLAnalyzer()
        self.logger = logging.getLogger(__name__)
        self.quiet = quiet

        # Penalty weights for scoring (1 point per error - placeholder for future refinement)
        self.penalty_weights = {
            ErrorType.XML_MALFORMED: 1,
            ErrorType.CHARACTER_ENCODING: 1,
            ErrorType.TAG_STRUCTURE: 1,
            ErrorType.ATTRIBUTE_SYNTAX: 1,
            ErrorType.FILE_NOT_FOUND: 0
        }

    def calculate_score(self, errors: List[Error], base_score: float = 100.0) -> float:
        """
        Calculate XML well-formedness score (placeholder scoring - 1 point per error).

        Returns score from 0-100 based on error count. This is a placeholder
        scoring system for future refinement.

        Args:
            errors: List of Error objects
            base_score: Starting score (default: 100.0)

        Returns:
            Final score after applying penalties (0-100)
        """
        # Filter out non-XML errors
        xml_errors = [e for e in errors if e.type not in [ErrorType.CONTENT_PRESERVATION, ErrorType.FILE_NOT_FOUND]]

        total_penalty = 0
        for error in xml_errors:
            penalty_per_error = self.penalty_weights.get(error.type, 1)
            total_penalty += penalty_per_error

        # Cap maximum penalty at base_score
        total_penalty = min(total_penalty, base_score)
        final_score = max(0, base_score - total_penalty)
        return final_score

    def check_comprehensive_wellformedness(self, file_path: str) -> Tuple[bool, List[Error], Dict[str, Any]]:
        """
        Comprehensive XML analysis using enhanced analyzer.

        Args:
            file_path: Path to XML file to analyze

        Returns:
            Tuple containing:
                - is_well_formed: Whether XML is well-formed
                - errors: List of Error objects
                - analysis_result: Detailed analysis dictionary
        """
        analysis_result = self.xml_analyzer.analyze_file(file_path)

        # Convert to Error objects
        errors = self.xml_analyzer.convert_to_error_objects(analysis_result)

        # Determine if well-formed
        is_well_formed = analysis_result['well_formed']

        return is_well_formed, errors, analysis_result

    def evaluate_file(self, xml_file_path: str, config: Optional[Dict[str, Any]] = None) -> EvaluationResult:
        """
        Evaluate a single XML file for well-formedness.

        Args:
            xml_file_path: Path to XML file to evaluate
            config: Optional configuration dictionary (currently unused)

        Returns:
            EvaluationResult containing pass/fail status, score, errors, and metrics
        """
        all_errors = []
        xml_file_path = Path(xml_file_path)

        # Check if XML file exists
        if not xml_file_path.exists():
            error = Error(
                type=ErrorType.FILE_NOT_FOUND,
                severity=10,
                location="File",
                message=f"XML file not found: {xml_file_path}",
                raw_error=""
            )
            all_errors.append(error)

            return EvaluationResult(
                dimension=0,
                passed=False,
                score=0.0,
                errors=all_errors,
                metrics={"file_readable": False, "xml_file_missing": True}
            )

        # XML well-formedness analysis
        is_well_formed, wellformedness_errors, analysis_details = self.check_comprehensive_wellformedness(str(xml_file_path))
        all_errors.extend(wellformedness_errors)

        # Calculate XML score
        xml_score = self.calculate_score(all_errors)

        # Overall pass/fail: XML must be well-formed
        passed = is_well_formed

        # Build metrics
        metrics = {
            "file_readable": analysis_details.get('file_readable', True),
            "well_formed": is_well_formed,
            "xml_score": xml_score,
            "total_errors": len(all_errors),
            "xml_errors": len(wellformedness_errors),
            "critical_errors": len([e for e in wellformedness_errors if e.severity >= 8]),
            "error_breakdown": {
                error_type.value: len([e for e in all_errors if e.type == error_type])
                for error_type in ErrorType
            },
            "detailed_analysis": analysis_details.get('summary', {}),
            "error_categories": analysis_details.get('errors', {})
        }

        return EvaluationResult(
            dimension=0,
            passed=passed,
            score=xml_score,
            errors=all_errors,
            metrics=metrics
        )

    def evaluate_directory(self, xml_path: str, pattern: str = "*.xml") -> List[EvaluationResult]:
        """
        Evaluate all XML files in a directory for well-formedness.

        This mode is optimized for fast XML validation.
        Use this when you only need to verify structural correctness of XML output.

        Args:
            xml_path: Directory path containing XML files or single XML file path
            pattern: Glob pattern to match files (default: "*.xml")

        Returns:
            List of EvaluationResult objects containing:
                - passed: Boolean indicating XML well-formedness
                - score: XML quality score (100 - error penalties)
                - errors: List of XML-related errors only
                - metrics: XML-specific metrics (well_formed, xml_score, error_breakdown)

        Performance:
            Well-formed files: ~1ms per file (lxml validation only)
            Malformed files: ~10-50ms per file (includes comprehensive error analysis)
        """
        xml_files = self._find_xml_files(xml_path, pattern)
        if not xml_files:
            return []

        results = []
        self.logger.info("Found %d XML files to evaluate", len(xml_files))
        if not self.quiet:
            print(f"Found {len(xml_files)} XML files to evaluate")
            print("=" * 60)

        for xml_file in sorted(xml_files):
            self.logger.debug("Evaluating: %s", xml_file.name)
            if not self.quiet:
                print(f"Evaluating: {xml_file.name}")
            try:
                result = self.evaluate_file(str(xml_file))
                result.metrics['file_path'] = str(xml_file)
                result.metrics['file_name'] = xml_file.name

                status = "[VALID]" if result.passed else "[INVALID]"
                xml_score = result.metrics.get('xml_score', result.score)
                self.logger.info("%s: %s | XML: %.2f/100 | Errors: %d",
                               xml_file.name, status, xml_score, len(result.errors))
                if not self.quiet:
                    print(f"   {status} | XML: {xml_score:6.2f}/100 | Errors: {len(result.errors)}")

                results.append(result)
            except (IOError, OSError, RuntimeError, ValueError) as e:
                self.logger.error("Error evaluating %s: %s", xml_file.name, str(e), exc_info=True)
                print(f"   [ERROR]: {str(e)}")
                results.append(self._create_error_result(xml_file, str(e)))

        return results

    def _find_xml_files(self, xml_path: str, pattern: str) -> List[Path]:
        """
        Find and validate XML files.

        Args:
            xml_path: Path to directory or file
            pattern: Glob pattern for matching files

        Returns:
            List of Path objects for XML files
        """
        xml_path = Path(xml_path)

        if not xml_path.exists():
            self.logger.error("XML directory not found: %s", xml_path)
            print(f"[ERROR] XML directory not found: {xml_path}")
            return []

        # Find all matching XML files
        if xml_path.is_file():
            xml_files = [xml_path]
        else:
            xml_files = list(xml_path.glob(pattern))

        if not xml_files:
            self.logger.warning("No XML files found in %s with pattern '%s'", xml_path, pattern)
            print(f"[ERROR] No XML files found in {xml_path} with pattern '{pattern}'")
            return []

        return xml_files

    def _create_error_result(self, xml_file: Path, error_message: str) -> EvaluationResult:
        """
        Create an error result for failed evaluation.

        Args:
            xml_file: Path to the XML file
            error_message: Error message describing the failure

        Returns:
            EvaluationResult with error information
        """
        return EvaluationResult(
            dimension=0,
            passed=False,
            score=0.0,
            errors=[Error(
                type=ErrorType.XML_MALFORMED,
                severity=10,
                location="File",
                message=f"Evaluation failed: {error_message}",
                raw_error=error_message
            )],
            metrics={
                "file_readable": False,
                "evaluation_error": True,
                "file_path": str(xml_file),
                "file_name": xml_file.name
            }
        )

