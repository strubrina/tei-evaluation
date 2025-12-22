"""
Dimension 3: Structural Comparison Against Reference Files.

This module evaluates structural similarity of XML/TEI output against reference files,
comparing element composition and structural organization.
"""

import logging
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from lxml import etree

from ..models import EvaluationResult, Error, ErrorType
from .base_evaluator import BaseEvaluator
from ..utils.structural_comparator import StructuralComparator
from ..utils.structural_difference_analyzer import StructuralDifferenceAnalyzer


class D3Structural(BaseEvaluator):
    """
    Dimension 3: Structural Comparison Against Reference Files

    Evaluates structural similarity of XML/TEI output against reference files.
    This dimension compares the structural organization and element composition
    of generated TEI documents with validated reference documents.

    Comparison Methods:
    1. Quick Check: Element count comparison
    2. XMLDiff Analysis: Structural difference detection
    3. Tree Edit Distance: Quantitative structural similarity

    Scoring:
    - Graduated scoring based on structural similarity metrics
    - Element count similarity (40%)
    - Tree edit distance (40%)
    - XMLDiff structural differences (20%)
    """

    def __init__(self, reference_directory: Optional[str] = None, quiet: bool = False):
        """
        Initialize D3Structural evaluator.

        Args:
            reference_directory: Path to reference XML files directory (optional, uses config if not provided)
            quiet: If True, suppress print statements (logging still active)
        """
        super().__init__()
        self.dimension = 3
        self.quiet = quiet
        self.logger = logging.getLogger(__name__)
        self.structural_comparator = StructuralComparator()
        self.difference_analyzer = StructuralDifferenceAnalyzer()

        # Reference directory setup
        self.reference_directory = self._setup_reference_directory(reference_directory)

        # Penalty weights for scoring
        self.penalty_weights = {
            ErrorType.TEI_STRUCTURE: 10  # High penalty for structural mismatches
        }

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

    def calculate_score(self, errors: list, base_score: float = 100.0) -> float:
        """
        Calculate score based on errors (required by BaseEvaluator).
        Note: For D3, the actual score is calculated as average of completeness and structural scores.
        This method is kept for compatibility with the abstract base class.
        """
        if not errors:
            return base_score

        total_penalty = 0
        for error in errors:
            penalty = self.penalty_weights.get(error.type, 10)
            total_penalty += penalty

        # Cap maximum penalty at base_score
        total_penalty = min(total_penalty, base_score)
        return max(0, base_score - total_penalty)

    def evaluate_file(self, file_path: str, config: Optional[Dict[str, Any]] = None) -> EvaluationResult:
        """
        Evaluate a single XML file for structural similarity against reference

        Config options:
        - reference_file: str (optional, overrides auto-detection)
        - require_reference: bool (default True, fail if no reference found)
        """
        config = config or {}
        all_errors = []
        file_path = Path(file_path)

        # Check if file exists
        if not file_path.exists():
            error = Error(
                type=ErrorType.TEI_STRUCTURE,
                severity=10,
                location="File",
                message=f"File not found: {file_path}",
                raw_error=""
            )
            all_errors.append(error)

            return EvaluationResult(
                dimension=3,
                passed=False,
                score=0.0,
                errors=all_errors,
                metrics={"file_readable": False}
            )

        # Find reference file
        reference_file = config.get('reference_file')
        if not reference_file:
            reference_file = self.find_reference_file(str(file_path))

        require_reference = config.get('require_reference', True)

        if not reference_file and require_reference:
            error = Error(
                type=ErrorType.TEI_STRUCTURE,
                severity=8,
                location="Reference",
                message=f"No reference file found for {file_path.name}",
                raw_error=f"Looked in: {self.reference_directory}"
            )
            all_errors.append(error)

            return EvaluationResult(
                dimension=3,
                passed=False,
                score=0.0,
                errors=all_errors,
                metrics={
                    "file_readable": True,
                    "reference_found": False,
                    "reference_directory": str(self.reference_directory) if self.reference_directory else None
                }
            )

        if not reference_file:
            # No reference required, skip comparison
            return EvaluationResult(
                dimension=3,
                passed=True,
                score=100.0,
                errors=[],
                metrics={
                    "file_readable": True,
                    "reference_found": False,
                    "comparison_skipped": True,
                    "reference_directory": str(self.reference_directory) if self.reference_directory else None
                }
            )

        self.logger.info("Comparing with reference: %s", Path(reference_file).name)
        if not self.quiet:
            print(f"   [INFO] Comparing with reference: {Path(reference_file).name}")

        xml_body = None
        ref_body = None
        seq_xml: Optional[List[str]] = None
        seq_ref: Optional[List[str]] = None
        tree_lcs_length: Optional[int] = None
        tree_lcs_similarity: Optional[float] = None

        try:
            # Perform structural comparison
            is_match, comparison_details, comparison_errors = self.structural_comparator.compare_structure(
                str(file_path), reference_file
            )

            all_errors.extend(comparison_errors)

            # Initialize element analysis data
            element_analysis = {
                'xml_element_count': 0,
                'ref_element_count': 0,
                'element_count_diff': 0,
                'added_element_types': [],
                'removed_element_types': [],
                'analysis_available': False
            }

            analysis_available = False

            # Always perform element analysis (for both matches and differences)
            try:
                xml_tree = etree.parse(str(file_path))
                ref_tree = etree.parse(reference_file)

                xml_body = self.structural_comparator._extract_body_content(xml_tree)
                ref_body = self.structural_comparator._extract_body_content(ref_tree)

                # Track whether body elements were found
                xml_body_found = xml_body is not None
                ref_body_found = ref_body is not None

                # Fallback to root element if body not found
                if xml_body is None:
                    xml_body = xml_tree.getroot()
                if ref_body is None:
                    ref_body = ref_tree.getroot()

                # Always perform comparison (even if body is missing)
                if xml_body is not None and ref_body is not None:
                    # Warn if body element was not found
                    if not xml_body_found:
                        self.logger.warning("Body element not found in XML - comparing from root element")
                        if not self.quiet:
                            print("      [WARNING] Body element not found in XML - comparing from root element")
                    if not ref_body_found:
                        self.logger.warning("Body element not found in reference - comparing from root element")
                        if not self.quiet:
                            print("      [WARNING] Body element not found in reference - comparing from root element")

                    # Use difference analyzer to get detailed element analysis
                    self.logger.info("Running detailed element analysis...")
                    if not self.quiet:
                        print("      [INFO] Running detailed element analysis...")
                    detailed_diff = self.difference_analyzer.analyze_detailed_differences(xml_body, ref_body)
                    diff_summary = detailed_diff.get('summary', {})

                    # Extract element counts and type changes
                    element_count_changes = diff_summary.get('element_count_changes', {})
                    added_types = diff_summary.get('added_element_types', [])
                    removed_types = diff_summary.get('removed_element_types', [])

                    # Store element analysis data
                    element_analysis.update({
                        'xml_element_count': diff_summary.get('xml_element_count', 0),
                        'ref_element_count': diff_summary.get('ref_element_count', 0),
                        'element_count_diff': diff_summary.get('element_count_diff', 0),
                        'added_element_types': added_types,
                        'removed_element_types': removed_types,
                        'element_count_changes': element_count_changes,  # Per-type counts!
                        'xml_body_found': xml_body_found,
                        'ref_body_found': ref_body_found,
                        'analysis_available': True
                    })
                    analysis_available = True

                    # Print analysis (only for differences)
                    if not is_match:
                        if added_types:
                            self.logger.info("Added element types: %s", ', '.join(sorted(added_types)))
                            if not self.quiet:
                                print(f"      Added: {', '.join(sorted(added_types))}")
                        if removed_types:
                            self.logger.info("Removed element types: %s", ', '.join(sorted(removed_types)))
                            if not self.quiet:
                                print(f"      Removed: {', '.join(sorted(removed_types))}")

                    # For perfect matches, ensure XMLDiff operations is 0 if not calculated
                    if is_match and 'xmldiff' in comparison_details and comparison_details['xmldiff'].get('available', False):
                        if 'total_operations' not in comparison_details['xmldiff'] and 'total_differences' not in comparison_details['xmldiff']:
                            comparison_details['xmldiff']['total_operations'] = 0
                            comparison_details['xmldiff']['total_differences'] = 0

                    seq_xml = self._get_preorder_sequence(xml_body)
                    seq_ref = self._get_preorder_sequence(ref_body)

            except (etree.XMLSyntaxError, AttributeError, ValueError, RuntimeError) as e:
                # If analysis fails, continue with default values
                self.logger.warning("Element analysis failed: %s", str(e))
                if not self.quiet:
                    print(f"      Warning: Element analysis failed: {e}")

            if seq_xml is not None and seq_ref is not None:
                tree_lcs_length, tree_lcs_similarity = self._calculate_tree_lcs(seq_xml, seq_ref)

            if isinstance(comparison_details, dict):
                comparison_details['analysis_available'] = analysis_available

            # Determine tree edit distance from XMLDiff results if available
            tree_edit_distance = None
            normalized_ted = None
            ted_similarity = None
            if isinstance(comparison_details, dict):
                xmldiff_info = comparison_details.get('xmldiff', {})
                if isinstance(xmldiff_info, dict) and xmldiff_info.get('available'):
                    tree_edit_distance = xmldiff_info.get('structural_differences')
                    if tree_edit_distance is None:
                        tree_edit_distance = xmldiff_info.get('total_differences')
                    if tree_edit_distance is not None:
                        xml_size = element_analysis.get('xml_element_count', 0)
                        ref_size = element_analysis.get('ref_element_count', 0)
                        sum_size = xml_size + ref_size
                        if sum_size > 0:
                            normalized_ted = tree_edit_distance / sum_size
                            ted_similarity = max(0.0, 1.0 - normalized_ted)
                        else:
                            normalized_ted = 0.0
                            ted_similarity = 1.0

            if tree_edit_distance is None:
                tree_edit_distance = 0 if is_match else None
            if normalized_ted is None:
                normalized_ted = 0.0 if tree_edit_distance == 0 else None
            if ted_similarity is None and normalized_ted is not None:
                ted_similarity = max(0.0, 1.0 - normalized_ted) if normalized_ted is not None else None

            if tree_lcs_length is None:
                if seq_xml is not None and seq_ref is not None:
                    tree_lcs_length = 0
                    tree_lcs_similarity = 1.0 if is_match else 0.0
                elif is_match:
                    tree_lcs_length = 0
                    tree_lcs_similarity = 1.0

            if isinstance(comparison_details, dict):
                comparison_details['tree_lcs'] = {
                    "length": tree_lcs_length,
                    "similarity": tree_lcs_similarity
                }

            # Calculate Completeness Score and Structural Score
            completeness_score = self._calculate_completeness_score(element_analysis)
            completeness_metrics = element_analysis.get('completeness_metrics', {})
            xmldiff_score = self._calculate_xmldiff_score(comparison_details, element_analysis)

            # Structural score now reflects LCS similarity (0-100)
            if tree_lcs_similarity is not None:
                structural_score = tree_lcs_similarity * 100.0
            elif ted_similarity is not None:
                structural_score = ted_similarity * 100.0
            else:
                structural_score = 0.0

            # Calculate overall score as average of completeness and structural scores
            score = (completeness_score + structural_score) / 2

            # Determine structural match based on 100% LCS similarity
            lcs_match = tree_lcs_similarity is not None and tree_lcs_similarity >= 1.0

            # Determine pass/fail based on LCS match AND no errors
            passed = lcs_match and len(all_errors) == 0

            # Compile metrics with element analysis data
            metrics = {
                "file_readable": True,
                "reference_found": True,
                "reference_file": reference_file,
                "structural_match": lcs_match,
                "completeness_score": completeness_score,
                "structural_score": structural_score,
                "xmldiff_score": xmldiff_score,
                "tree_edit_distance": tree_edit_distance,
                "tree_edit_normalized": normalized_ted,
                "tree_similarity": ted_similarity,
                "tree_lcs_length": tree_lcs_length,
                "tree_lcs_similarity": tree_lcs_similarity,
                "completeness_precision": completeness_metrics.get('precision', 0.0),
                "completeness_recall": completeness_metrics.get('recall', 0.0),
                "completeness_f1": completeness_metrics.get('f1', completeness_score),
                "completeness_macro_f1": completeness_metrics.get('macro_f1', 0.0),
                "completeness_micro_f1": completeness_metrics.get('micro_f1', completeness_score),
                "completeness_macro_precision": completeness_metrics.get('macro_precision', 0.0),
                "completeness_macro_recall": completeness_metrics.get('macro_recall', 0.0),
                "completeness_micro_precision": completeness_metrics.get('micro_precision', completeness_metrics.get('precision', 0.0)),
                "completeness_micro_recall": completeness_metrics.get('micro_recall', completeness_metrics.get('recall', 0.0)),
                "comparison_details": comparison_details,
                "detailed_analysis": {
                    "summary": element_analysis  # Store element analysis here for CSV access
                },
                "total_errors": len(all_errors),
                "error_breakdown": {
                    error_type.value: len([e for e in all_errors if e.type == error_type])
                    for error_type in ErrorType
                },
                "analysis_available": analysis_available,
                "dependencies": self.structural_comparator.get_required_dependencies()
            }

            return EvaluationResult(
                dimension=3,
                passed=passed,
                score=score,
                errors=all_errors,
                metrics=metrics
            )

        except (IOError, OSError, RuntimeError, ValueError, etree.XMLSyntaxError) as e:
            error = Error(
                type=ErrorType.TEI_STRUCTURE,
                severity=8,
                location="Evaluation",
                message=f"Error during Dimension 3 evaluation: {str(e)}",
                raw_error=str(e)
            )
            all_errors.append(error)

            return EvaluationResult(
                dimension=3,
                passed=False,
                score=0.0,
                errors=all_errors,
                metrics={"evaluation_error": True}
            )
    def evaluate_batch(self, input_path: str, pattern: str = "*.xml", config: Optional[Dict[str, Any]] = None) -> List[EvaluationResult]:
        """
        Evaluate multiple XML files for structural similarity
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

        # Display setup information
        deps = self.structural_comparator.get_required_dependencies()
        self.logger.info("Found %d XML files to evaluate", len(xml_files))
        if not self.quiet:
            print(f"Found {len(xml_files)} XML files to evaluate")
            print(f"Reference Directory: {'[FOUND]' if self.reference_directory else '[NOT SET]'}")
            if self.reference_directory:
                print(f"   Path: {self.reference_directory}")
            if not all(deps.values()):
                print(f"[INFO] {self.structural_comparator.install_dependencies_message()}")
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
                if result.metrics.get('comparison_skipped'):
                    self.logger.info("Skipped - no reference file available")
                    if not self.quiet:
                        print("   [SKIP] No reference file available")
                else:
                    status = "[MATCH]" if result.metrics.get('structural_match') else "[DIFF]"
                    self.logger.info("%s | Score: %.1f/100", status.strip('[]'), result.score)
                    if not self.quiet:
                        print(f"   {status} | Score: {result.score:5.1f}/100")

            except (IOError, OSError, RuntimeError, ValueError, etree.XMLSyntaxError) as e:
                self.logger.error("Error evaluating file: %s", str(e), exc_info=True)
                if not self.quiet:
                    print(f"   [ERROR] {str(e)}")
                # Create error result for failed evaluation
                error_result = EvaluationResult(
                    dimension=3,
                    passed=False,
                    score=0.0,
                    errors=[Error(
                        type=ErrorType.TEI_STRUCTURE,
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
        """Generate summary statistics for batch evaluation results"""
        if not results:
            return {"total_files": 0, "message": "No results to summarize"}

        total_files = len(results)

        # Count different result types
        matched_files = sum(1 for r in results if r.metrics.get('structural_match', False))
        diff_files = sum(1 for r in results if not r.metrics.get('structural_match', False) and not r.metrics.get('comparison_skipped', False))
        skipped_files = sum(1 for r in results if r.metrics.get('comparison_skipped', False))
        error_files = sum(1 for r in results if r.metrics.get('evaluation_error', False))

        # Completeness metrics
        completeness_matches = 0
        completeness_scores = []
        structural_scores = []

        for r in results:
            if not r.metrics.get('comparison_skipped', False):
                completeness_scores.append(r.metrics.get('completeness_score', 0))
                structural_scores.append(r.metrics.get('structural_score', 0))

                # Check completeness match (simple version - just check if completeness_score is 100)
                if r.metrics.get('completeness_score', 0) == 100.0:
                    completeness_matches += 1

        completeness_diffs = (total_files - skipped_files) - completeness_matches

        # Score statistics
        scores = [r.score for r in results if not r.metrics.get('comparison_skipped', False)]
        avg_score = sum(scores) / len(scores) if scores else 0
        avg_completeness = sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0
        avg_structural = sum(structural_scores) / len(structural_scores) if structural_scores else 0

        # Reference file statistics
        ref_found_count = sum(1 for r in results if r.metrics.get('reference_found', False))

        # Error statistics
        all_errors = []
        for result in results:
            all_errors.extend(result.errors)

        error_breakdown = {}
        for error_type in ErrorType:
            count = len([e for e in all_errors if e.type == error_type])
            if count > 0:
                error_breakdown[error_type.value] = count

        # Dependencies check
        deps_available = all(self.structural_comparator.get_required_dependencies().values())

        summary = {
            "total_files": total_files,
            "matched_files": matched_files,
            "diff_files": diff_files,
            "skipped_files": skipped_files,
            "error_files": error_files,
            "completeness_matches": completeness_matches,
            "completeness_diffs": completeness_diffs,
            "avg_completeness_score": avg_completeness,
            "avg_structural_score": avg_structural,
            "match_rate": (matched_files / (total_files - skipped_files) * 100) if (total_files - skipped_files) > 0 else 0,
            "reference_found_count": ref_found_count,
            "reference_found_rate": (ref_found_count / total_files * 100) if total_files > 0 else 0,
            "average_score": avg_score,
            "min_score": min(scores) if scores else 0,
            "max_score": max(scores) if scores else 0,
            "total_errors": len(all_errors),
            "error_breakdown": error_breakdown,
            "dependencies_available": deps_available,
            "dependencies": self.structural_comparator.get_required_dependencies(),
            "detailed_results": [
                {
                    "file_name": result.metrics.get('file_name', f"File_{i+1}"),
                    "file_path": result.metrics.get('file_path', ''),
                    "structural_match": result.metrics.get('structural_match', False),
                    "reference_found": result.metrics.get('reference_found', False),
                    "comparison_skipped": result.metrics.get('comparison_skipped', False),
                    "score": result.score,
                    "error_count": len(result.errors)
                }
                for i, result in enumerate(results)
            ]
        }

        return summary

    def _get_element_types(self, element: etree._Element) -> set:
        """Get set of all element types in tree"""
        elements = set()

        def collect_elements(elem):
            elem_name = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
            if elem_name not in self.structural_comparator.ignore_elements:
                elements.add(elem_name)
            for child in elem:
                collect_elements(child)

        collect_elements(element)
        return elements

    def _calculate_completeness_score(self, element_analysis: Dict[str, Any]) -> float:
        """
        Calculate Completeness Score (0-100%) using micro-averaged F1 across element types.
        Also stores detailed precision/recall metrics on the analysis summary.
        """
        metrics = self._calculate_completeness_metrics(element_analysis)
        element_analysis['completeness_metrics'] = metrics

        if not metrics.get('available', False):
            return 0.0

        return metrics.get('micro_f1', 0.0)

    def _calculate_completeness_metrics(self, element_analysis: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate precision/recall/F1 metrics for element completeness.
        Returns values as percentages (0-100).
        """
        metrics = {
            "available": element_analysis.get('analysis_available', False),
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "micro_precision": 0.0,
            "micro_recall": 0.0,
            "micro_f1": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0
        }

        if not metrics["available"]:
            return metrics

        element_count_changes = element_analysis.get('element_count_changes', {})
        if not element_count_changes:
            # No elements present – treat as perfect match
            metrics.update({
                "precision": 100.0,
                "recall": 100.0,
                "f1": 100.0,
                "micro_precision": 100.0,
                "micro_recall": 100.0,
                "micro_f1": 100.0,
                "macro_precision": 100.0,
                "macro_recall": 100.0,
                "macro_f1": 100.0
            })
            return metrics

        tp_sum = fp_sum = fn_sum = 0
        per_type_precisions = []
        per_type_recalls = []
        per_type_f1 = []

        for change_data in element_count_changes.values():
            xml_count = change_data.get('xml_count', 0)
            ref_count = change_data.get('ref_count', 0)

            if xml_count == 0 and ref_count == 0:
                continue

            tp = min(xml_count, ref_count)
            fp = max(xml_count - ref_count, 0)
            fn = max(ref_count - xml_count, 0)

            tp_sum += tp
            fp_sum += fp
            fn_sum += fn

            precision_den = tp + fp
            recall_den = tp + fn

            if precision_den > 0:
                precision_t = tp / precision_den
            else:
                precision_t = 1.0 if recall_den == 0 else 0.0

            if recall_den > 0:
                recall_t = tp / recall_den
            else:
                recall_t = 1.0 if precision_den == 0 else 0.0

            if precision_t + recall_t > 0:
                f1_t = (2 * precision_t * recall_t) / (precision_t + recall_t)
            else:
                f1_t = 0.0

            per_type_precisions.append(precision_t)
            per_type_recalls.append(recall_t)
            per_type_f1.append(f1_t)

        # Micro averages
        if tp_sum == 0 and fp_sum == 0 and fn_sum == 0:
            micro_precision = micro_recall = micro_f1 = 1.0
        else:
            micro_precision = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0
            micro_recall = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0.0
            if micro_precision + micro_recall > 0:
                micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall)
            else:
                micro_f1 = 0.0

        # Macro averages
        if per_type_precisions:
            macro_precision = sum(per_type_precisions) / len(per_type_precisions)
            macro_recall = sum(per_type_recalls) / len(per_type_recalls)
            macro_f1 = sum(per_type_f1) / len(per_type_f1)
        else:
            macro_precision = macro_recall = macro_f1 = 1.0

        # Store both standard and micro values (same for now)
        metrics.update({
            "precision": micro_precision * 100,
            "recall": micro_recall * 100,
            "f1": micro_f1 * 100,
            "micro_precision": micro_precision * 100,
            "micro_recall": micro_recall * 100,
            "micro_f1": micro_f1 * 100,
            "macro_precision": macro_precision * 100,
            "macro_recall": macro_recall * 100,
            "macro_f1": macro_f1 * 100
        })

        return metrics

    def _calculate_xmldiff_score(self, comparison_details: Dict[str, Any], element_analysis: Dict[str, Any]) -> float:
        """
        Calculate XMLDiff-based structural score (0-100%) using structural operations.

        Formula: (1 - Structural_XMLDiff_Ops / Total_Elements) * 100

        Structural operations: ALL structural changes (InsertNode, DeleteNode, MoveNode,
        RenameNode, attribute changes)
        Excludes ONLY: text content changes (UpdateTextIn, UpdateTextAfter)

        Perfect structural match = 100%
        """
        if not element_analysis.get('analysis_available', False):
            return 0.0

        if not isinstance(comparison_details, dict):
            comparison_details = {}

        xmldiff_data = comparison_details.get('xmldiff', {})
        if not comparison_details.get('analysis_available', False) and not xmldiff_data.get('available', False):
            return 0.0

        # Get total element count (use reference as baseline)
        ref_count = element_analysis.get('ref_element_count', 0)

        # Edge case: no elements
        if ref_count == 0:
            return 100.0

        # Count structural operations (all except text changes)
        structural_ops = 0

        if xmldiff_data.get('available', False):
            # Get detailed operations
            detailed_ops = xmldiff_data.get('detailed_operations', [])

            for op in detailed_ops:
                is_structural = op.get('is_structural', False)

                # Count ALL structural operations (is_structural already excludes text changes)
                if is_structural:
                    structural_ops += 1

        # Calculate score
        score = (1 - (structural_ops / ref_count)) * 100

        # Cap at 0-100 range
        return max(0.0, min(100.0, score))

    def _get_preorder_sequence(self, element: etree._Element) -> Optional[List[str]]:
        """Return preorder sequence of element tags, ignoring comparator's ignored elements."""
        if element is None:
            return None

        sequence: List[str] = []
        ignore = set()
        if hasattr(self.structural_comparator, "ignore_elements"):
            ignore = set(self.structural_comparator.ignore_elements)

        def traverse(node: etree._Element):
            if not hasattr(node, "tag") or not isinstance(node.tag, str):
                for child in node:
                    traverse(child)
                return
            tag_str = node.tag.split('}', 1)[-1] if '}' in node.tag else node.tag
            if tag_str not in ignore:
                sequence.append(tag_str)
            for child in node:
                traverse(child)

        traverse(element)
        return sequence

    def _calculate_tree_lcs(self, seq1: List[str], seq2: List[str]) -> Tuple[int, float]:
        """Calculate LCS length and similarity ratio for two sequences."""
        if seq1 is None or seq2 is None:
            return 0, 0.0
        len1 = len(seq1)
        len2 = len(seq2)
        if len1 == 0 and len2 == 0:
            return 0, 1.0
        dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
        for i in range(len1):
            a = seq1[i]
            for j in range(len2):
                if a == seq2[j]:
                    dp[i + 1][j + 1] = dp[i][j] + 1
                else:
                    dp[i + 1][j + 1] = dp[i + 1][j] if dp[i + 1][j] >= dp[i][j + 1] else dp[i][j + 1]
        lcs_len = dp[len1][len2]
        denom = max(len1, len2)
        similarity = (lcs_len / denom) if denom > 0 else 1.0
        return lcs_len, similarity

    def print_batch_summary(self, results: List[EvaluationResult]):
        """
        Print a formatted summary of batch evaluation results.

        Args:
            results: List of evaluation results to summarize
        """
        if self.quiet:
            return

        summary = self.generate_batch_summary(results)

        print("\n" + "="*60)
        print("DIMENSION 3 STRUCTURAL COMPARISON SUMMARY")
        print("="*60)
        print(f"Total Files: {summary['total_files']}")
        print(f"Skipped (No Reference): {summary['skipped_files']}")
        print()
        print(f"Completeness Matches: {summary.get('completeness_matches', 0)}")
        print(f"Completeness Differences: {summary.get('completeness_diffs', 0)}")
        print(f"Avg Completeness Score: {summary.get('avg_completeness_score', 0):.1f}%")
        print()
        print(f"Structural Matches: {summary['matched_files']}")
        print(f"Structural Differences: {summary['diff_files']}")
        print(f"Avg Structural Score: {summary.get('avg_structural_score', 0):.1f}%")
        print()
        print(f"Errors: {summary['error_files']}")


        # Dependencies status
        deps = summary['dependencies']
        print(f"\nDependencies: xmldiff={'[OK]' if deps['xmldiff'] else '[MISSING]'}")
        if not summary['dependencies_available']:
            print(f"[INFO] {self.structural_comparator.install_dependencies_message()}")

        if summary['error_breakdown']:
            print(f"\nError Breakdown:")
            for error_type, count in summary['error_breakdown'].items():
                print(f"  - {error_type}: {count}")

        print(f"\nFile Details:")
        for detail in summary['detailed_results']:
            if detail['comparison_skipped']:
                print(f"  [SKIP] {detail['file_name']:<30} No reference available")
            else:
                # Determine status and result text
                is_match = detail['structural_match']
                status = "[MATCH]" if is_match else "[DIFF]"
                result_text = "Perfect match" if is_match else f"Differences: {detail['error_count']}"

                # Print the result
                print(f"  {status} {detail['file_name']:<30} Score: {detail['score']:5.1f} | {result_text}")

        print("="*60)