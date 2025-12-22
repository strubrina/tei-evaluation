"""
Dimension 1: Source Fidelity Evaluation.

This module evaluates the fidelity of XML/TEI output to the original source text.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..models import Error, ErrorType, EvaluationResult
from ..utils.content_preservation import ContentPreservationAnalyzer
from .base_evaluator import BaseEvaluator


class D1Source(BaseEvaluator):
    """
    Dimension 1: Source Fidelity Evaluation.

    Evaluates the fidelity of XML/TEI output to the original source text.
    This dimension focuses exclusively on content preservation:

    - Compares original text with XML-encoded content
    - Measures text similarity (with/without whitespace)
    - Detects missing, extra, or modified content
    - Provides character-level difference analysis

    Scoring:
        Source Fidelity Score: Text similarity percentage (0-100)
        Pass/Fail: Must be 100% match (whitespace ignored)

    Attributes:
        dimension: Evaluation dimension number (1)
        content_analyzer: ContentPreservationAnalyzer instance
        logger: Logger instance for this evaluator
        quiet: Whether to suppress console output
    """

    def __init__(self, quiet: bool = False):
        """
        Initialize the D1 Source Fidelity evaluator.

        Args:
            quiet: If True, suppress console output
        """
        super().__init__()
        self.dimension = 1
        self.content_analyzer = ContentPreservationAnalyzer()
        self.logger = logging.getLogger(__name__)
        self.quiet = quiet

    def calculate_score(self, errors: List[Error], base_score: float = 100.0) -> float:
        """
        Calculate score based on errors (required by BaseEvaluator).

        For source fidelity, the score is primarily based on content similarity,
        not error count, so this is a placeholder implementation.

        Args:
            errors: List of Error objects (not used in D1)
            base_score: Starting score (default: 100.0)

        Returns:
            Base score (actual score comes from calculate_content_score())
        """
        # For D1Source, we don't use error-based scoring
        # The actual score comes from calculate_content_score()
        return base_score

    def check_content_preservation(self, xml_file_path: str, txt_file_path: str) -> Tuple[Dict[str, Any], List[Error]]:
        """
        Analyze content preservation between original text and XML output.

        Args:
            xml_file_path: Path to XML file
            txt_file_path: Path to original text file

        Returns:
            Tuple containing:
                - comparison_results: Dictionary with similarity metrics and differences
                - errors: List of Error objects
        """
        try:
            # Read both files
            with open(txt_file_path, 'r', encoding='utf-8') as f:
                original_text = f.read()

            with open(xml_file_path, 'r', encoding='utf-8') as f:
                xml_content = f.read()

            # Analyze content preservation
            comparison_results, errors = self.content_analyzer.analyze_content_preservation(original_text, xml_content)

            return comparison_results, errors

        except (IOError, OSError, UnicodeDecodeError, ValueError) as e:
            self.logger.error("Content preservation analysis failed: %s", str(e), exc_info=True)
            error = Error(
                type=ErrorType.CONTENT_PRESERVATION,
                severity=8,
                location="Content Analysis",
                message=f"Content preservation analysis failed: {str(e)}",
                raw_error=str(e)
            )
            return {"analysis_failed": True, "error": str(e)}, [error]

    def calculate_content_score(self, comparison_results: Dict[str, Any]) -> float:
        """
        Calculate content preservation score (0-100).

        Args:
            comparison_results: Dictionary containing comparison metrics

        Returns:
            Content preservation score (0-100)
        """
        if comparison_results.get("analysis_failed", False):
            return 0.0

        # If exact match without whitespace, perfect score
        if comparison_results.get("exact_match_without_whitespace", False):
            return 100.0

        # Otherwise use similarity score
        similarity = comparison_results.get("similarity_without_whitespace", 0.0)
        return similarity * 100

    def find_corresponding_txt_file(self, xml_file_path: str, txt_directory: str) -> Optional[str]:
        """
        Find the corresponding .txt file for a .xml file.

        Args:
            xml_file_path: Path to XML file
            txt_directory: Directory containing text files

        Returns:
            Path to corresponding text file, or None if not found
        """
        xml_path = Path(xml_file_path)
        txt_dir = Path(txt_directory)

        # Extract base name (e.g., "letter120" from "letter120.xml")
        base_name = xml_path.stem

        # Look for corresponding .txt file
        txt_file = txt_dir / f"{base_name}.txt"

        return str(txt_file) if txt_file.exists() else None

    def evaluate_file(self, xml_file_path: str, txt_file_path: str, config: Optional[Dict[str, Any]] = None) -> EvaluationResult:
        """
        Evaluate content preservation for a single XML file against its source text.

        Args:
            xml_file_path: Path to XML file
            txt_file_path: Path to original text file
            config: Optional configuration dictionary (currently unused)

        Returns:
            EvaluationResult containing pass/fail status, score, errors, and metrics
        """
        all_errors = []
        xml_file_path = Path(xml_file_path)
        txt_file_path = Path(txt_file_path)

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
                dimension=1,
                passed=False,
                score=0.0,
                errors=all_errors,
                metrics={"file_readable": False, "xml_file_missing": True}
            )

        # Check if TXT file exists
        if not txt_file_path.exists():
            error = Error(
                type=ErrorType.FILE_NOT_FOUND,
                severity=9,
                location="Content Analysis",
                message=f"Original text file not found: {txt_file_path}",
                raw_error=""
            )
            all_errors.append(error)

            return EvaluationResult(
                dimension=1,
                passed=False,
                score=0.0,
                errors=all_errors,
                metrics={"txt_file_missing": True}
            )

        # Content preservation analysis
        content_results, content_errors = self.check_content_preservation(str(xml_file_path), str(txt_file_path))
        all_errors.extend(content_errors)
        content_score = self.calculate_content_score(content_results)

        # For source fidelity, pass/fail requires 100% content match
        content_passed = content_score >= 100.0  # Must be exact match (whitespace ignored)

        # Build metrics
        metrics = {
            "file_readable": True,
            "content_score": content_score,
            "total_errors": len(all_errors),
            "content_errors": len(content_errors),
            "error_breakdown": {
                "content_preservation": len(content_errors)
            },
            "content_preservation": {
                "original_file_path": str(txt_file_path),
                "original_file_found": True,
                **content_results
            }
        }

        return EvaluationResult(
            dimension=1,
            passed=content_passed,
            score=content_score,
            errors=all_errors,
            metrics=metrics
        )

    def evaluate_directory(self, xml_path: str, txt_path: str, pattern: str = "*.xml") -> List[EvaluationResult]:
        """
        Evaluate source fidelity for all XML files in a directory.

        This mode focuses on comparing the text content of XML files
        against original source text files.

        Process:
        1. Extracts text content from XML (strips all markup)
        2. Compares with original text file
        3. Calculates similarity scores
        4. Identifies differences (missing, extra, or modified text)

        Args:
            xml_path: Directory path containing XML files or single XML file path
            txt_path: Directory path containing original text files
            pattern: Glob pattern to match files (default: "*.xml")

        Returns:
            List of EvaluationResult objects containing:
                - passed: Boolean (True if content_score >= 100.0)
                - score: Content preservation score (0-100, based on text similarity)
                - errors: List of content preservation errors only
                - metrics: Content-specific metrics (content_score, exact_match, similarity)

        Note:
            Corresponding text files are matched by filename (e.g., letter123.xml → letter123.txt)
        """
        xml_files = self._find_xml_files(xml_path, pattern)
        if not xml_files:
            return []

        if not self._validate_content_directory(txt_path):
            return []

        results = []
        self.logger.info("Found %d XML files to evaluate", len(xml_files))
        if not self.quiet:
            print(f"Found {len(xml_files)} XML files to evaluate")
            print(f"Mode: Source Fidelity (Content Preservation)")
            print(f"Comparing with text files in: {txt_path}")
            print("=" * 60)

        for xml_file in sorted(xml_files):
            self.logger.debug("Evaluating: %s", xml_file.name)
            if not self.quiet:
                print(f"Evaluating: {xml_file.name}")
            try:
                txt_file_path = self.find_corresponding_txt_file(str(xml_file), txt_path)
                if not txt_file_path:
                    txt_file_path = str(Path(txt_path) / f"{xml_file.stem}.txt")

                result = self.evaluate_file(str(xml_file), txt_file_path)
                result.metrics['file_path'] = str(xml_file)
                result.metrics['file_name'] = xml_file.name

                status = "[VALID]" if result.passed else "[INVALID]"
                content_score = result.metrics.get('content_score', result.score)
                self.logger.info("%s: %s | Content: %.2f/100 | Errors: %d",
                               xml_file.name, status, content_score, len(result.errors))
                if not self.quiet:
                    print(f"   {status} | Content: {content_score:6.2f}/100 | Errors: {len(result.errors)}")

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

    def _validate_content_directory(self, original_text_path: Optional[str]) -> bool:
        """
        Validate content directory and return whether content analysis should be performed.

        Args:
            original_text_path: Path to original text directory

        Returns:
            True if directory exists and is valid, False otherwise
        """
        if not original_text_path:
            return False

        txt_path = Path(original_text_path)
        if not txt_path.exists():
            self.logger.error("Original text directory not found: %s", txt_path)
            print(f"[ERROR] Original text directory not found: {txt_path}")
            return False

        return True

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
            dimension=1,
            passed=False,
            score=0.0,
            errors=[Error(
                type=ErrorType.CONTENT_PRESERVATION,
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

