"""
Dimension 4: Semantic Content Matching Against Reference Files.

This module evaluates semantic content accuracy by comparing TEI element content
against validated reference files, focusing on text content within structural elements.
"""

import json
import logging
import re
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any

from lxml import etree

from ..models import EvaluationResult, Error, ErrorType
from .base_evaluator import BaseEvaluator
from ..utils.tei_wrapper import TEIWrapper
from ..utils.whitespace_normalizer import normalize_whitespace


class D4Semantic(BaseEvaluator):
    """
    Dimension 4: Semantic Content Matching Against Reference Files

    Evaluates semantic content accuracy by comparing TEI element content
    against validated reference files. This dimension focuses on verifying
    that the actual text content within structural elements matches the
    expected reference content.

    Comparison Methods:
    1. Exact Match: Content is identical after normalization
    2. Over-inclusion: LLM content includes complete reference + extra text
       - Reports inclusion_ratio: what % of LLM content is the reference
    3. Under-inclusion: LLM content is substring of reference (partial capture)
       - Reports coverage_ratio: what % of reference was captured
    4. No Match: No substring relationship exists

    Practical Match Strategy (NO THRESHOLDS):
    - All substring relationships (over/under-inclusion) count as practical matches
    - Ratios are calculated and reported for quantitative analysis
    - Quality interpretation happens in unified reporting using D1 context:
      * If D1 content preserved → practical matches weighted 0.8 (likely boundary issues)
      * If D1 NOT preserved → practical matches weighted 0.6 (possible hallucinations)

    Scoring:
    - Graduated scoring based on match quality (exact vs practical)
    - Error penalties for missing, extra, or completely mismatched content
    - Minimum score of 20/100 to indicate partial success
    """

    def __init__(self, reference_directory: Optional[str] = None, auto_wrap_tei: bool = True,
                 content_elements: Optional[Any] = None, quiet: bool = False):
        """
        Initialize Dimension 4 Semantic Content Matching evaluator.

        Args:
            reference_directory: Path to directory containing reference XML files
            auto_wrap_tei: Whether to automatically wrap non-TEI XML with TEI structure
            content_elements: Element types to compare. Can be:
                - None (default): Auto-discover all text-bearing elements (genre-agnostic)
                - List[str]: Custom list of element types (e.g., ['speaker', 'stage', 'l'])
                - 'correspondence': Use predefined correspondence elements
                - 'auto': Same as None, auto-discover (explicit)
            quiet: If True, suppress print statements (logging still active)

        Examples:
            # Default mode (auto-discovery - genre-agnostic)
            evaluator = D4Semantic()

            # Custom elements for dramas
            evaluator = D4Semantic(content_elements=['speaker', 'stage', 'l', 'p'])

            # Correspondence-specific mode
            evaluator = D4Semantic(content_elements='correspondence')

            # Explicit auto-discovery mode
            evaluator = D4Semantic(content_elements='auto')
        """
        super().__init__()
        self.dimension = 4
        self.quiet = quiet
        self.logger = logging.getLogger(__name__)

        # Reference directory setup
        self.reference_directory = self._setup_reference_directory(reference_directory)

        # Initialize TEI wrapper
        self.tei_wrapper = TEIWrapper()
        self.auto_wrap_tei = auto_wrap_tei

        # Elements to compare for content matching
        if content_elements == 'correspondence':
            # Correspondence-specific mode: use predefined list
            self.content_elements = [
                'dateline', 'salute', 'signed', 'opener', 'closer',
                'p', 'address', 'persName', 'placeName', 'date'
            ]
        elif content_elements is None or content_elements == 'auto':
            # Default: Auto-discovery mode (genre-agnostic)
            self.content_elements = 'auto'
        else:
            # Custom list provided by user
            self.content_elements = content_elements

        # =============================================================================
        # PRACTICAL MATCH STRATEGY:
        # =============================================================================
        # No thresholds are applied for practical matches in D4 individual evaluation.
        #
        # Rationale:
        # 1. D4 detects WHETHER a substring relationship exists (over/under-inclusion)
        # 2. Ratios (inclusion_ratio, coverage_ratio) are calculated and reported
        # 3. The ADJUSTED SCORE in unified reporting uses D1 context to interpret quality:
        #    - If D1 content preserved → practical matches weighted 0.8 (likely boundary issues)
        #    - If D1 NOT preserved → practical matches weighted 0.6 (possible hallucinations)
        #
        # This two-stage approach:
        # - Keeps D4 flexible and domain-agnostic
        # - Provides quantitative ratio data for analysis
        # - Uses cross-dimensional context for quality interpretation
        # =============================================================================

        # No thresholds applied - all substring relationships are considered practical matches
        # Ratios are calculated and reported for analysis
        self.default_thresholds = None

        # Penalty weights for scoring
        self.penalty_weights = {
            ErrorType.ENTITY_RECOGNITION: 0,    # Content mismatch - penalties removed
            ErrorType.TEMPORAL_PROCESSING: 2,   # Date content issues (reduced from 4)
            ErrorType.EDITORIAL_CONVERSION: 1,  # Minor content differences (reduced from 3)
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

    def _prepare_file_for_evaluation(self, file_path: str, auto_wrap: bool = None, wrap_metadata: Dict[str, Any] = None) -> Tuple[str, bool]:
        """
        Prepare file for evaluation, wrapping with TEI if needed

        Returns:
            Tuple of (actual_file_path, was_wrapped)
        """
        # Use instance setting if not specified
        if auto_wrap is None:
            auto_wrap = self.auto_wrap_tei

        if not auto_wrap:
            return file_path, False

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

                # Create temporary unescaped file
                file_path_obj = Path(file_path)
                temp_unescaped_path = file_path_obj.parent / f"{file_path_obj.stem}_unescaped_temp.xml"

                with open(temp_unescaped_path, 'w', encoding='utf-8') as f:
                    f.write(unescaped_content)

                self.logger.info("Created temporary unescaped file")
                if not self.quiet:
                    print("   [INFO] Created temporary unescaped file")

                # Check if the unescaped content is TEI XML
                if self.tei_wrapper._is_tei_xml(unescaped_content):
                    self.logger.info("Unescaped content is TEI XML - using temporary file")
                    if not self.quiet:
                        print("   [INFO] Unescaped content is TEI XML - using temporary file")
                    return str(temp_unescaped_path), True

                # If not TEI, continue with wrapping the unescaped content
                content = unescaped_content
                file_path = str(temp_unescaped_path)

            # Check if it's already TEI XML (after potential unescaping)
            if self.tei_wrapper._is_tei_xml(content):
                return file_path, False

            # File needs wrapping
            self.logger.info("File needs TEI wrapping - proceeding...")
            if not self.quiet:
                print("   [INFO] File needs TEI wrapping - proceeding...")
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
                print(f"   [WARNING] Could not wrap file with TEI: {e}")
            return file_path, False

    def detailed_substring_analysis(self, llm_text: str, ref_text: str, element_type: str = None) -> Dict[str, Any]:
        """
        Advanced content comparison using substring analysis.

        NO THRESHOLDS APPLIED: All substring relationships are considered practical matches.
        Ratios are calculated and reported for quantitative analysis.
        Quality interpretation happens in the unified adjusted score using D1 context.

        Args:
            llm_text: Text from LLM-generated element
            ref_text: Text from reference element
            element_type: Type of element (for potential future use, currently not used for thresholds)

        Returns:
            Dictionary with match analysis including:
            - match_type: 'exact', 'over_inclusion', 'under_inclusion', 'no_match'
            - exact_match: Boolean
            - practical_match: Boolean (True for any substring relationship)
            - inclusion_ratio: What % of LLM content is the reference (for over-inclusion)
            - coverage_ratio: What % of reference was captured by LLM (for under-inclusion)
        """
        llm_norm = self.normalize_text(llm_text)
        ref_norm = self.normalize_text(ref_text)

        # Handle empty content
        if not ref_norm and not llm_norm:
            return {
                'match_type': 'both_empty',
                'exact_match': True,
                'practical_match': True,
                'analysis': 'Both elements are empty'
            }

        if not ref_norm:
            return {
                'match_type': 'reference_empty',
                'exact_match': False,
                'practical_match': False,
                'analysis': 'Reference element is empty, LLM has content'
            }

        if not llm_norm:
            return {
                'match_type': 'llm_empty',
                'exact_match': False,
                'practical_match': False,
                'analysis': 'LLM element is empty, reference has content'
            }

        # Exact match
        if llm_norm == ref_norm:
            return {
                'match_type': 'exact',
                'exact_match': True,
                'practical_match': True,
                'inclusion_ratio': 1.0,
                'coverage_ratio': 1.0,
                'analysis': 'Perfect match'
            }

        # Over-inclusion: reference content found in LLM content
        # LLM has the complete reference but added extra text
        if ref_norm in llm_norm:
            start_pos = llm_norm.find(ref_norm)
            end_pos = start_pos + len(ref_norm)

            prefix = llm_norm[:start_pos]
            suffix = llm_norm[end_pos:]
            inclusion_ratio = len(ref_norm) / len(llm_norm)

            # NO THRESHOLD - all over-inclusions are practical matches
            # Ratio provides quantitative measure of match quality
            practical_match = True

            return {
                'match_type': 'over_inclusion',
                'exact_match': False,
                'practical_match': practical_match,
                'inclusion_ratio': round(inclusion_ratio, 3),
                'coverage_ratio': 1.0,
                'extra_prefix': prefix,
                'extra_suffix': suffix,
                'analysis': f'LLM included {len(prefix)} chars before, {len(suffix)} chars after'
            }

        # Under-inclusion: LLM content found in reference content
        # LLM captured part of the reference but is missing some content
        if llm_norm in ref_norm:
            coverage_ratio = len(llm_norm) / len(ref_norm)

            # NO THRESHOLD - all under-inclusions are practical matches
            # Ratio provides quantitative measure of how much was captured
            practical_match = True

            return {
                'match_type': 'under_inclusion',
                'exact_match': False,
                'practical_match': practical_match,
                'inclusion_ratio': 1.0,
                'coverage_ratio': round(coverage_ratio, 3),
                'analysis': f'LLM captured {coverage_ratio:.1%} of reference content'
            }

        # No substring relationship
        return {
            'match_type': 'no_match',
            'exact_match': False,
            'practical_match': False,
            'inclusion_ratio': 0.0,
            'coverage_ratio': 0.0,
            'analysis': 'No substring relationship found'
        }

    def find_best_content_match(self, llm_element, ref_elements: List, used_indices: set, element_type: str = None) -> tuple:
        """
        Find the best matching reference element for an LLM element based on content.

        Args:
            llm_element: The LLM element to match
            ref_elements: List of all reference elements of same type
            used_indices: Set of already-matched reference indices
            element_type: Type of element for threshold selection

        Returns:
            Tuple of (best_match_index, analysis_dict) or (None, None) if no good match
        """
        if not ref_elements:
            return None, None

        llm_content = self.extract_element_content(llm_element)
        llm_norm = self.normalize_text(llm_content)

        best_match_idx = None
        best_analysis = None
        best_score = -1

        for idx, ref_element in enumerate(ref_elements):
            # Skip already-used reference elements
            if idx in used_indices:
                continue

            ref_content = self.extract_element_content(ref_element)
            ref_norm = self.normalize_text(ref_content)

            # Exact match gets highest priority
            if llm_norm == ref_norm:
                analysis = self.detailed_substring_analysis(llm_content, ref_content, element_type)
                return idx, analysis

            # Calculate match quality based on substring relationships
            analysis = self.detailed_substring_analysis(llm_content, ref_content, element_type)

            # Score based on match type priority
            score = 0
            if analysis.get('match_type') == 'exact':
                score = 100
            elif analysis.get('match_type') == 'over_inclusion':
                # Prioritize by inclusion ratio
                score = 50 + (analysis.get('inclusion_ratio', 0) * 40)
            elif analysis.get('match_type') == 'under_inclusion':
                # Prioritize by coverage ratio
                score = 30 + (analysis.get('coverage_ratio', 0) * 20)
            elif analysis.get('match_type') == 'no_match':
                score = 0

            if score > best_score:
                best_score = score
                best_match_idx = idx
                best_analysis = analysis

        # Only return a match if we found something reasonable (score > 0)
        if best_score > 0:
            return best_match_idx, best_analysis

        return None, None

    def get_namespaced_xpath(self, xml_tree: etree._ElementTree, xpath_query: str) -> List:
        """Helper method to handle namespaced XPath queries"""
        root = xml_tree.getroot()
        namespaces = root.nsmap

        if namespaces and namespaces.get(None) == "http://www.tei-c.org/ns/1.0":
            # TEI namespace - convert XPath to use namespace prefix
            ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
            # Convert //element to //tei:element
            namespaced_query = xpath_query.replace('//', '//tei:').replace('tei:tei:', 'tei:')
            return xml_tree.xpath(namespaced_query, namespaces=ns)
        else:
            # No namespace - use regular XPath
            return xml_tree.xpath(xpath_query)

    def extract_element_content(self, element: etree._Element) -> str:
        """Extract only direct text content from element, excluding child element text"""
        if element is None:
            return ""

        # Get only direct text content (element.text and element.tail), excluding child element text
        content_parts = []
        if element.text:
            content_parts.append(element.text.strip())

        # Include tail text (text after the element)
        if element.tail:
            content_parts.append(element.tail.strip())

        # Join with single spaces and normalize
        full_content = " ".join(part for part in content_parts if part)
        return self.normalize_text(full_content)

    def has_only_child_elements(self, element: etree._Element) -> bool:
        """
        Check if element only contains child elements without any direct text content.

        Returns True if element should be SKIPPED (wrapper only or empty).
        Returns False if element should be COMPARED (has direct text).

        Examples:
            <opener><dateline>Text</dateline></opener>  -> True (skip, wrapper only)
            <p>Some text <persName>John</persName></p>  -> False (compare, has direct text)
            <p></p>                                      -> True (skip, empty)
        """
        if element is None:
            return True  # Skip None elements

        # Check if element has any direct text content (ignoring whitespace)
        has_direct_text = bool(element.text and element.text.strip())

        if has_direct_text:
            # Has direct text → should be compared
            return False

        # No direct text - check if it has children
        has_children = len(element) > 0

        if has_children:
            # No direct text but has children → wrapper only → skip
            return True
        else:
            # No direct text and no children → empty element → skip
            return True

    def get_element_info(self, element: etree._Element) -> Dict[str, Any]:
        """
        Get information about an element including child element names.
        """
        if element is None:
            return {'has_children': False, 'child_names': []}

        has_children = len(element) > 0
        child_names = []

        if has_children:
            # Get unique child element names (strip namespace)
            for child in element:
                tag = child.tag
                # Remove namespace if present
                if '}' in tag:
                    tag = tag.split('}')[1]
                if tag not in child_names:
                    child_names.append(tag)

        return {
            'has_children': has_children,
            'child_names': child_names
        }

    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison - preserves case, normalizes whitespace"""
        # Using unified whitespace normalizer
        return normalize_whitespace(text, mode="text")


    def discover_text_bearing_elements(self, llm_tree: etree._ElementTree,
                                       ref_tree: etree._ElementTree) -> List[str]:
        """
        Discover all element types in body that contain direct text content.

        This method scans both the LLM output and reference documents to find
        all unique element types that have direct text nodes (not just child text).
        Elements that are pure wrappers (contain only child elements) are excluded.

        Args:
            llm_tree: The LLM-generated XML tree
            ref_tree: The reference XML tree

        Returns:
            Sorted list of unique element type names that have direct text content

        Examples:
            <p>Text here</p>                              -> 'p' included
            <persName>John <surname>Doe</surname></persName> -> 'persName' included
            <opener><dateline>Date</dateline></opener>    -> only 'dateline' included (opener skipped)
        """
        element_types = set()

        # Search both trees
        for tree in [llm_tree, ref_tree]:
            if tree is None:
                continue

            # Get body element
            body_elements = self.get_namespaced_xpath(tree, "//body")
            if not body_elements:
                continue

            body_elem = body_elements[0]

            # Iterate through all descendants of body
            for elem in body_elem.iter():
                # Get element tag name (strip namespace if present)
                tag = elem.tag
                if isinstance(tag, str):  # Skip comments and processing instructions
                    if '}' in tag:
                        tag = tag.split('}')[1]  # Remove namespace

                    # Skip the body element itself
                    if tag == 'body':
                        continue

                    # Check if element has direct text (not just child text)
                    has_direct_text = bool(elem.text and elem.text.strip())

                    if has_direct_text:
                        element_types.add(tag)

        return sorted(list(element_types))

    def compare_elements_by_xpath(self, llm_tree: etree._ElementTree,
                                ref_tree: etree._ElementTree) -> List[Dict[str, Any]]:
        """
        Compare elements between LLM and reference trees - only inside body element.

        Supports two modes:
        1. Explicit list: Compare only specified element types (self.content_elements)
        2. Auto-discovery: Discover and compare all text-bearing elements (self.content_elements == 'auto')
        """
        comparison_results = []

        # Determine which element types to compare
        if self.content_elements == 'auto':
            # Auto-discovery mode: find all text-bearing elements
            element_types = self.discover_text_bearing_elements(llm_tree, ref_tree)
            self.logger.info("AUTO-DISCOVERY: Found %d text-bearing element types: %s", len(element_types), ', '.join(element_types))
            if not self.quiet:
                print(f"   [AUTO-DISCOVERY] Found {len(element_types)} text-bearing element types: {', '.join(element_types)}")
        else:
            # Explicit list mode: use provided element types
            element_types = self.content_elements

        for element_type in element_types:
            # Only search for elements inside the body element
            xpath_query = f"//body//{element_type}"

            # Get elements from both trees
            llm_elements_raw = self.get_namespaced_xpath(llm_tree, xpath_query)
            ref_elements_raw = self.get_namespaced_xpath(ref_tree, xpath_query)

            # Filter out elements that only contain child elements (no text content)
            llm_elements = [elem for elem in llm_elements_raw if not self.has_only_child_elements(elem)]
            ref_elements = [elem for elem in ref_elements_raw if not self.has_only_child_elements(elem)]

            # Compare element counts
            llm_count = len(llm_elements)
            ref_count = len(ref_elements)

            element_comparison = {
                'element_type': element_type,
                'llm_count': llm_count,
                'ref_count': ref_count,
                'count_match': llm_count == ref_count,
                'positional_comparisons': [],  # Position-based matching (strict)
                'content_comparisons': []       # Content-based matching (best-match)
            }

            # PART 1: POSITIONAL MATCHING (strict - same index must match)
            max_elements = max(llm_count, ref_count)
            for i in range(max_elements):
                llm_element = llm_elements[i] if i < llm_count else None
                ref_element = ref_elements[i] if i < ref_count else None

                if llm_element is not None and ref_element is not None:
                    # Both elements exist at same position - compare content
                    llm_content = self.extract_element_content(llm_element)
                    ref_content = self.extract_element_content(ref_element)

                    content_analysis = self.detailed_substring_analysis(llm_content, ref_content, element_type)
                    content_analysis.update({
                        'element_index': i,
                        'llm_content': llm_content,
                        'ref_content': ref_content,
                        'element_present': 'both',
                        'llm_element_info': self.get_element_info(llm_element)
                    })

                elif llm_element is not None:
                    # LLM has element, reference doesn't at this position
                    llm_content = self.extract_element_content(llm_element)
                    content_analysis = {
                        'element_index': i,
                        'llm_content': llm_content,
                        'ref_content': '',
                        'element_present': 'llm_only',
                        'match_type': 'extra_element',
                        'exact_match': False,
                        'fuzzy_match': False,
                        'practical_match': False,
                        'analysis': 'Element exists in LLM but not in reference at this position',
                        'llm_element_info': self.get_element_info(llm_element)
                    }

                else:
                    # Reference has element, LLM doesn't at this position
                    ref_content = self.extract_element_content(ref_element)
                    content_analysis = {
                        'element_index': i,
                        'llm_content': '',
                        'ref_content': ref_content,
                        'element_present': 'ref_only',
                        'match_type': 'missing_element',
                        'exact_match': False,
                        'fuzzy_match': False,
                        'practical_match': False,
                        'analysis': 'Element exists in reference but not in LLM at this position'
                    }

                element_comparison['positional_comparisons'].append(content_analysis)

            # PART 2: CONTENT-BASED MATCHING (find best match regardless of position)
            used_ref_indices = set()

            for i, llm_element in enumerate(llm_elements):
                llm_content = self.extract_element_content(llm_element)

                # Try to find best matching reference element
                best_ref_idx, content_analysis = self.find_best_content_match(
                    llm_element, ref_elements, used_ref_indices, element_type
                )

                if best_ref_idx is not None:
                    # Found a match
                    used_ref_indices.add(best_ref_idx)
                    ref_element = ref_elements[best_ref_idx]
                    ref_content = self.extract_element_content(ref_element)

                    content_analysis.update({
                        'llm_index': i,
                        'ref_index': best_ref_idx,
                        'position_match': i == best_ref_idx,  # Track if positions also matched
                        'llm_content': llm_content,
                        'ref_content': ref_content,
                        'element_present': 'both',
                        'llm_element_info': self.get_element_info(llm_element)
                    })
                else:
                    # No match found - LLM element is extra
                    content_analysis = {
                        'llm_index': i,
                        'ref_index': None,
                        'position_match': False,
                        'llm_content': llm_content,
                        'ref_content': '',
                        'element_present': 'llm_only',
                        'match_type': 'extra_element',
                        'exact_match': False,
                        'fuzzy_match': False,
                        'practical_match': False,
                        'analysis': 'Element exists in LLM but no matching content in reference',
                        'llm_element_info': self.get_element_info(llm_element)
                    }

                element_comparison['content_comparisons'].append(content_analysis)

            # Add unmatched reference elements
            for i, ref_element in enumerate(ref_elements):
                if i not in used_ref_indices:
                    ref_content = self.extract_element_content(ref_element)
                    content_analysis = {
                        'llm_index': None,
                        'ref_index': i,
                        'position_match': False,
                        'llm_content': '',
                        'ref_content': ref_content,
                        'element_present': 'ref_only',
                        'match_type': 'missing_element',
                        'exact_match': False,
                        'fuzzy_match': False,
                        'practical_match': False,
                        'analysis': 'Element exists in reference but no matching content in LLM'
                    }
                    element_comparison['content_comparisons'].append(content_analysis)

            comparison_results.append(element_comparison)

        return comparison_results

    def generate_errors_from_comparison(self, comparison_results: List[Dict[str, Any]]) -> List[Error]:
        """Generate Error objects from comparison results"""
        errors = []

        for element_comparison in comparison_results:
            element_type = element_comparison['element_type']

            # NOTE: Count mismatches are NOT penalized in D4
            # Element count differences are evaluated in Dimension 3 (Structural Comparison)
            # D4 focuses only on content quality of matched elements

            # Check content mismatches
            for content_comp in element_comparison['content_comparisons']:
                if not content_comp.get('exact_match', True):
                    # Use llm_index or ref_index depending on which is available
                    element_index = content_comp.get('llm_index') if content_comp.get('llm_index') is not None else content_comp.get('ref_index')
                    match_type = content_comp.get('match_type', 'unknown')

                    if match_type == 'missing_element':
                        errors.append(Error(
                            type=ErrorType.ENTITY_RECOGNITION,
                            severity=6,
                            location=f"{element_type}[{element_index}]",
                            message=f"Missing {element_type} element in LLM output",
                            raw_error=f"missing_{element_type}_{element_index}"
                        ))

                    elif match_type == 'extra_element':
                        errors.append(Error(
                            type=ErrorType.ENTITY_RECOGNITION,
                            severity=4,
                            location=f"{element_type}[{element_index}]",
                            message=f"Extra {element_type} element in LLM output",
                            raw_error=f"extra_{element_type}_{element_index}"
                        ))

                    elif not content_comp.get('practical_match', False):
                        # NOTE: With no thresholds, practical_match is True for all substring relationships
                        # This block only executes for 'no_match' cases (no substring relationship found)
                        # Over-inclusion and under-inclusion no longer generate errors regardless of ratio

                        if match_type == 'over_inclusion':
                            # This block is now unreachable (over-inclusion always has practical_match=True)
                            # Kept for code structure consistency
                            inclusion_ratio = content_comp.get('inclusion_ratio', 0)
                            if inclusion_ratio < 0.3:
                                errors.append(Error(
                                    type=ErrorType.EDITORIAL_CONVERSION,
                                    severity=3,
                                    location=f"{element_type}[{element_index}]",
                                    message=f"Over-inclusion in {element_type}: {inclusion_ratio:.1%} content is correct",
                                    raw_error=f"over_inclusion_{element_type}_{element_index}"
                                ))

                        elif match_type == 'under_inclusion':
                            # This block is now unreachable (under-inclusion always has practical_match=True)
                            # Kept for code structure consistency
                            coverage_ratio = content_comp.get('coverage_ratio', 0)
                            if coverage_ratio < 0.3:
                                errors.append(Error(
                                    type=ErrorType.EDITORIAL_CONVERSION,
                                    severity=4,
                                    location=f"{element_type}[{element_index}]",
                                    message=f"Under-inclusion in {element_type}: {coverage_ratio:.1%} of reference captured",
                                    raw_error=f"under_inclusion_{element_type}_{element_index}"
                                ))

                        elif match_type == 'no_match':
                            # This is the only case that generates errors now
                            # No substring relationship exists between LLM and reference content
                            errors.append(Error(
                                type=ErrorType.ENTITY_RECOGNITION,
                                severity=7,
                                location=f"{element_type}[{element_index}]",
                                message=f"Content mismatch in {element_type}: no similarity found",
                                raw_error=f"content_mismatch_{element_type}_{element_index}"
                            ))

        return errors

    def calculate_score(self, errors: List[Error], comparison_results: List[Dict[str, Any]],
                       base_score: float = 100.0, d1_content_preserved: Optional[bool] = None) -> float:
        """
        Calculate score based on content matching results using Macro F1.

        Uses D1 content preservation status to determine practical match weight:
        - If D1 content preserved: practical_weight = 0.8 (likely boundary issues)
        - If D1 content NOT preserved: practical_weight = 0.6 (possible hallucinations)
        - If D1 status unknown: practical_weight = 0.6 (default, conservative)

        The score is calculated as Macro F1 from content-based matching metrics,
        which treats each element type equally and uses weighted TP for practical matches.

        Args:
            errors: List of errors found during evaluation (not used in scoring)
            comparison_results: List of element comparison results
            base_score: Base score (default 100.0)
            d1_content_preserved: Whether D1 content preservation passed (None if unknown)

        Returns:
            Final score (0-100) based on Macro F1 only
        """
        # Determine practical match weight based on D1 content preservation
        if d1_content_preserved is True:
            practical_weight = 0.8  # Higher weight - likely boundary issues
        else:
            practical_weight = 0.6  # Lower weight - possible hallucinations or unknown D1 status

        # Generate summary stats with context-aware practical weight
        summary_stats = self._generate_summary_stats(comparison_results, practical_weight=practical_weight)

        # Get content-based Macro F1 (this is what we use for scoring)
        content_metrics = summary_stats.get('content', {}).get('metrics', {})
        macro_f1 = content_metrics.get('macro_f1', 0.0)  # Already in 0-100 range

        # Get comparison statistics
        content_stats = summary_stats.get('content', {})
        compared_elements = content_stats.get('total_elements', 0)
        missing_elements = content_stats.get('missing_elements', 0)
        extra_elements = content_stats.get('extra_elements', 0)

        # Use Macro F1 as the match score
        # First, handle the case where no comparisons were made at all
        if compared_elements == 0 and missing_elements == 0 and extra_elements == 0:
            # No elements to compare - might be no comparison data available
            # This could happen if both files are empty or have no comparable elements
            return base_score

        # If 0 elements compared but there are missing/extra elements,
        # elements exist but none matched - F1 should be 0 (not base_score)
        # This ensures we return 0.0 in this case, even if macro_f1 calculation
        # has some edge case that returns a non-zero value
        if compared_elements == 0 and (missing_elements > 0 or extra_elements > 0):
            # Elements exist in reference and/or LLM output, but no matches
            # This means TP=0, FP>0 or FN>0, so F1 should definitely be 0
            return 0.0

        # Normal case: use calculated Macro F1
        if macro_f1 > 0:
            return macro_f1
        else:
            # Macro F1 is 0 - could be because elements were compared but none matched
            return 0.0

    def evaluate_file(self, file_path: str, config: Optional[Dict[str, Any]] = None) -> EvaluationResult:
        """Evaluate a single XML file for Dimension 4 content matching"""
        config = config or {}
        all_errors = []
        file_path = Path(file_path)

        # Check if file exists
        if not file_path.exists():
            error = Error(
                type=ErrorType.ENTITY_RECOGNITION,
                severity=10,
                location="File",
                message=f"File not found: {file_path}",
                raw_error=""
            )
            all_errors.append(error)

            return EvaluationResult(
                dimension=4,
                passed=False,
                score=0.0,
                errors=all_errors,
                metrics={"file_readable": False}
            )

        # Find reference file
        reference_file = config.get('reference_file')
        if not reference_file:
            # Use original file path for reference matching, not the wrapped temp file
            reference_file = self.find_reference_file(str(file_path))

        if not reference_file:
            return EvaluationResult(
                dimension=4,
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

        self.logger.info("Comparing content with reference: %s", Path(reference_file).name)
        if not self.quiet:
            print(f"   Comparing content with reference: {Path(reference_file).name}")

        llm_file_to_parse = str(file_path)
        llm_was_wrapped = False

        try:
            # Prepare LLM file (wrap if needed)
            auto_wrap = config.get('auto_wrap_tei', self.auto_wrap_tei)
            if auto_wrap:
                llm_file_to_parse, llm_was_wrapped = self._prepare_file_for_evaluation(
                    str(file_path),
                    auto_wrap=True,
                    wrap_metadata={'title': file_path.stem.replace('_', ' ').title()}
                )

            # Parse both files
            llm_tree = etree.parse(llm_file_to_parse)
            ref_tree = etree.parse(reference_file)

            # Perform content comparison
            comparison_results = self.compare_elements_by_xpath(llm_tree, ref_tree)

            # Generate errors from comparison
            comparison_errors = self.generate_errors_from_comparison(comparison_results)
            all_errors.extend(comparison_errors)

            # Get D1 content preservation status from config (if available)
            d1_content_preserved = config.get('d1_content_preserved', None)

            # Calculate score with D1-aware weighting
            score = self.calculate_score(all_errors, comparison_results, d1_content_preserved=d1_content_preserved)

            # Determine pass/fail
            passed = len(all_errors) == 0

            # Generate summary statistics (use same practical_weight as in calculate_score)
            d1_content_preserved = config.get('d1_content_preserved', None)
            practical_weight = 0.8 if d1_content_preserved is True else 0.6
            summary_stats = self._generate_summary_stats(comparison_results, practical_weight=practical_weight)

            # Compile metrics
            metrics = {
                "file_readable": True,
                "reference_found": True,
                "reference_file": reference_file,
                "comparison_skipped": False,
                "comparison_results": comparison_results,
                "summary_stats": summary_stats,
                "total_errors": len(all_errors),
                "error_breakdown": {
                    error_type.value: len([e for e in all_errors if e.type == error_type])
                    for error_type in ErrorType
                }
            }

            return EvaluationResult(
                dimension=4,
                passed=passed,
                score=score,
                errors=all_errors,
                metrics=metrics
            )

        except (IOError, OSError, RuntimeError, ValueError, etree.XMLSyntaxError) as e:
            error = Error(
                type=ErrorType.ENTITY_RECOGNITION,
                severity=8,
                location="Evaluation",
                message=f"Error during Dimension 4 evaluation: {str(e)}",
                raw_error=str(e)
            )
            all_errors.append(error)

            return EvaluationResult(
                dimension=4,
                passed=False,
                score=0.0,
                errors=all_errors,
                metrics={"evaluation_error": True}
            )

        finally:
            # Clean up temporary wrapped file if created
            if llm_was_wrapped and llm_file_to_parse != str(file_path):
                try:
                    temp_file = Path(llm_file_to_parse)
                    if temp_file.exists():
                        temp_file.unlink()
                except (OSError, PermissionError):
                    pass  # Ignore cleanup errors


    def _generate_summary_stats(self, comparison_results: List[Dict[str, Any]],
                                practical_weight: float = 0.6) -> Dict[str, Any]:
        """
        Generate summary statistics from comparison results.

        Args:
            comparison_results: List of element comparison results
            practical_weight: Weight for practical matches (0.6 if D1 not preserved, 0.8 if preserved)
        """
        # Positional matching stats (strict - same index)
        pos_compared_elements = 0
        pos_exact_matches = 0
        pos_practical_matches = 0
        pos_missing_elements = 0
        pos_extra_elements = 0

        # Content-based matching stats (best match regardless of position)
        content_compared_elements = 0
        content_exact_matches = 0
        content_practical_matches = 0
        content_missing_elements = 0
        content_extra_elements = 0
        content_reordered_elements = 0  # Elements matched but at different positions

        element_stats = {}
        positional_type_counts: Dict[str, Dict[str, float]] = {}
        content_type_counts: Dict[str, Dict[str, float]] = {}

        for element_comparison in comparison_results:
            element_type = element_comparison['element_type']
            element_stats[element_type] = {
                'positional': {
                    'compared': 0,
                    'exact_matches': 0,
                    'practical_matches': 0
                },
                'content': {
                    'compared': 0,
                    'exact_matches': 0,
                    'practical_matches': 0,
                    'reordered': 0
                }
            }
            positional_type_counts.setdefault(element_type, {'tp': 0.0, 'fp': 0.0, 'fn': 0.0})
            content_type_counts.setdefault(element_type, {'tp': 0.0, 'fp': 0.0, 'fn': 0.0})

            # Process positional comparisons
            for pos_comp in element_comparison.get('positional_comparisons', []):
                if pos_comp.get('element_present') == 'both':
                    pos_compared_elements += 1
                    element_stats[element_type]['positional']['compared'] += 1
                    pos_counts = positional_type_counts[element_type]

                    # Count exact matches and practical matches as mutually exclusive
                    if pos_comp.get('exact_match', False):
                        pos_exact_matches += 1
                        element_stats[element_type]['positional']['exact_matches'] += 1
                        pos_counts['tp'] += 1
                    elif pos_comp.get('practical_match', False):
                        pos_practical_matches += 1
                        element_stats[element_type]['positional']['practical_matches'] += 1
                        # Practical matches use weighted TP: TP = weight, FP = 1-weight, FN = 1-weight
                        pos_counts['tp'] += practical_weight
                        pos_counts['fp'] += (1.0 - practical_weight)
                        pos_counts['fn'] += (1.0 - practical_weight)
                    else:
                        pos_counts['fp'] += 1
                        pos_counts['fn'] += 1

                elif pos_comp.get('element_present') == 'ref_only':
                    pos_missing_elements += 1
                    positional_type_counts[element_type]['fn'] += 1
                elif pos_comp.get('element_present') == 'llm_only':
                    pos_extra_elements += 1
                    positional_type_counts[element_type]['fp'] += 1

            # Process content-based comparisons
            for content_comp in element_comparison.get('content_comparisons', []):
                if content_comp.get('element_present') == 'both':
                    content_compared_elements += 1
                    element_stats[element_type]['content']['compared'] += 1
                    content_counts = content_type_counts[element_type]

                    # Count exact matches and practical matches as mutually exclusive
                    if content_comp.get('exact_match', False):
                        content_exact_matches += 1
                        element_stats[element_type]['content']['exact_matches'] += 1
                        content_counts['tp'] += 1
                    elif content_comp.get('practical_match', False):
                        content_practical_matches += 1
                        element_stats[element_type]['content']['practical_matches'] += 1
                        # Practical matches use weighted TP: TP = weight, FP = 1-weight, FN = 1-weight
                        content_counts['tp'] += practical_weight
                        content_counts['fp'] += (1.0 - practical_weight)
                        content_counts['fn'] += (1.0 - practical_weight)
                    else:
                        content_counts['fp'] += 1
                        content_counts['fn'] += 1

                    # Check if element was reordered (matched but not at same position)
                    if not content_comp.get('position_match', True):
                        content_reordered_elements += 1
                        element_stats[element_type]['content']['reordered'] += 1

                elif content_comp.get('element_present') == 'ref_only':
                    content_missing_elements += 1
                    content_type_counts[element_type]['fn'] += 1
                elif content_comp.get('element_present') == 'llm_only':
                    content_extra_elements += 1
                    content_type_counts[element_type]['fp'] += 1

        positional_metrics = self._calculate_match_metrics(positional_type_counts)
        content_metrics = self._calculate_match_metrics(content_type_counts)

        return {
            # Positional matching (strict)
            'positional': {
                'total_elements': pos_compared_elements,
                'exact_matches': pos_exact_matches,
                'practical_matches': pos_practical_matches,
                'missing_elements': pos_missing_elements,
                'extra_elements': pos_extra_elements,
                'precision': positional_metrics.get('precision', 0),
                'recall': positional_metrics.get('recall', 0),
                'f1': positional_metrics.get('f1', 0),
                'micro_precision': positional_metrics.get('micro_precision', 0),
                'micro_recall': positional_metrics.get('micro_recall', 0),
                'micro_f1': positional_metrics.get('micro_f1', 0),
                'macro_precision': positional_metrics.get('macro_precision', 0),
                'macro_recall': positional_metrics.get('macro_recall', 0),
                'macro_f1': positional_metrics.get('macro_f1', 0),
                'metrics': positional_metrics
            },
            # Content-based matching (best match)
            'content': {
                'total_elements': content_compared_elements,
                'exact_matches': content_exact_matches,
                'practical_matches': content_practical_matches,
                'missing_elements': content_missing_elements,
                'extra_elements': content_extra_elements,
                'reordered_elements': content_reordered_elements,
                'precision': content_metrics.get('precision', 0),
                'recall': content_metrics.get('recall', 0),
                'f1': content_metrics.get('f1', 0),
                'micro_precision': content_metrics.get('micro_precision', 0),
                'micro_recall': content_metrics.get('micro_recall', 0),
                'micro_f1': content_metrics.get('micro_f1', 0),
                'macro_precision': content_metrics.get('macro_precision', 0),
                'macro_recall': content_metrics.get('macro_recall', 0),
                'macro_f1': content_metrics.get('macro_f1', 0),
                'metrics': content_metrics
            },
            # Backward compatibility - use positional stats as default
            'total_elements': pos_compared_elements,
            'exact_matches': pos_exact_matches,
            'practical_matches': pos_practical_matches,
            'missing_elements': pos_missing_elements,
            'extra_elements': pos_extra_elements,
            'exact_match_rate': (pos_exact_matches / pos_compared_elements * 100) if pos_compared_elements > 0 else 0,
            'practical_match_rate': (pos_practical_matches / pos_compared_elements * 100) if pos_compared_elements > 0 else 0,
            'element_stats': element_stats,
            'positional_metrics': positional_metrics,
            'content_metrics': content_metrics
        }

    def _calculate_match_metrics(self, per_type_counts: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Calculate precision/recall/F1 metrics (micro & macro) from per-type counts.

        Args:
            per_type_counts: Dictionary mapping element types to TP/FP/FN counts (can be floats for weighted counts)
        """
        metrics = {
            "available": False,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "micro_precision": 0.0,
            "micro_recall": 0.0,
            "micro_f1": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "counts": {"tp": 0, "fp": 0, "fn": 0}
        }

        if not per_type_counts:
            return metrics

        metrics["available"] = True
        tp_sum = fp_sum = fn_sum = 0
        per_type_precisions = []
        per_type_recalls = []
        per_type_f1 = []

        for counts in per_type_counts.values():
            tp = float(counts.get('tp', 0))
            fp = float(counts.get('fp', 0))
            fn = float(counts.get('fn', 0))

            tp_sum += tp
            fp_sum += fp
            fn_sum += fn

            precision_den = tp + fp
            recall_den = tp + fn

            precision = tp / precision_den if precision_den > 0 else (1.0 if recall_den == 0 else 0.0)
            recall = tp / recall_den if recall_den > 0 else (1.0 if precision_den == 0 else 0.0)
            if precision + recall > 0:
                f1 = (2 * precision * recall) / (precision + recall)
            else:
                f1 = 0.0

            per_type_precisions.append(precision)
            per_type_recalls.append(recall)
            per_type_f1.append(f1)

        # Micro averages
        if tp_sum == 0 and fp_sum == 0 and fn_sum == 0:
            micro_precision = micro_recall = micro_f1 = 1.0
        else:
            micro_precision = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0
            micro_recall = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0.0
            micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)) if (micro_precision + micro_recall) > 0 else 0.0

        # Macro averages
        if per_type_precisions:
            macro_precision = sum(per_type_precisions) / len(per_type_precisions)
            macro_recall = sum(per_type_recalls) / len(per_type_recalls)
            macro_f1 = sum(per_type_f1) / len(per_type_f1)
        else:
            macro_precision = macro_recall = macro_f1 = 1.0

        metrics.update({
            "precision": micro_precision * 100,
            "recall": micro_recall * 100,
            "f1": micro_f1 * 100,
            "micro_precision": micro_precision * 100,
            "micro_recall": micro_recall * 100,
            "micro_f1": micro_f1 * 100,
            "macro_precision": macro_precision * 100,
            "macro_recall": macro_recall * 100,
            "macro_f1": macro_f1 * 100,
            "counts": {"tp": tp_sum, "fp": fp_sum, "fn": fn_sum}
        })

        return metrics

    def _load_d1_content_preservation_map(self, config: Optional[Dict[str, Any]]) -> Dict[str, bool]:
        """
        Load D1 content preservation status from D1 evaluation results.

        Looks for d1_source_fidelity_report.json in the same directory structure
        as the evaluation output. This allows D4 to use context-aware weighting.

        Args:
            config: Evaluation config that may contain output_directory or d1_results_path

        Returns:
            Dictionary mapping file_name -> content_preserved (bool)
        """
        d1_map = {}

        # Try to find D1 results file
        d1_results_path = None

        # Option 1: Check if config specifies D1 results path
        if config and 'd1_results_path' in config:
            d1_results_path = Path(config['d1_results_path'])
        # Option 2: Look for D1 results in output directory (same level as D4 output)
        elif config and 'output_directory' in config:
            output_dir = Path(config['output_directory'])
            # Try same directory first
            d1_results_path = output_dir / "d1_source_fidelity_report.json"
            # If not found, try parent directory (for unified evaluation structure)
            if not d1_results_path.exists():
                parent_dir = output_dir.parent
                d1_results_path = parent_dir / "d1_source_fidelity_report.json"

        if d1_results_path and d1_results_path.exists():
            try:
                with open(d1_results_path, 'r', encoding='utf-8') as f:
                    d1_data = json.load(f)

                # Extract content_preserved status for each file
                if 'files' in d1_data:
                    for file_data in d1_data['files']:
                        if isinstance(file_data, str):  # Skip section headers
                            continue
                        file_name = file_data.get('file_name', '')
                        if file_name:
                            d1_map[file_name] = file_data.get('content_preserved', False)

                if d1_map:
                    preserved_count = sum(1 for v in d1_map.values() if v)
                    self.logger.info("Loaded D1 content preservation data for %d files", len(d1_map))
                    self.logger.info("Files with preserved content: %d (will use 0.8 weight)", preserved_count)
                    self.logger.info("Files without preserved content: %d (will use 0.6 weight)", len(d1_map) - preserved_count)
                    if not self.quiet:
                        print(f"[INFO] Loaded D1 content preservation data for {len(d1_map)} files")
                        print(f"[INFO]   - Files with preserved content: {preserved_count} (will use 0.8 weight)")
                        print(f"[INFO]   - Files without preserved content: {len(d1_map) - preserved_count} (will use 0.6 weight)")
            except (IOError, OSError, json.JSONDecodeError, KeyError):
                # Silently fail - D1 data is optional
                pass

        return d1_map

    def print_batch_summary(self, results):
        """
        Print a summary of batch evaluation results.

        Args:
            results: List of evaluation results to summarize
        """
        if self.quiet:
            return

        summary = self.generate_batch_summary(results)

        print(f"\n{'='*60}")
        print("DIMENSION 4 BATCH EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total Files: {summary['total_files']}")
        print(f"Passed: {summary['passed_files']}")
        print(f"Failed: {summary['failed_files']}")
        print(f"Skipped: {summary['skipped_files']}")
        print(f"Files with References: {summary['files_with_references']}")
        print(f"Average Score: {summary['average_score']:.1f}/100")
        print(f"Content Match Rate: {summary['content_match_rate']:.1f}%")

        if summary['content_stats']['total_comparisons'] > 0:
            print(f"\nCONTENT MATCHING DETAILS:")
            print(f"Total Elements: {summary['content_stats']['total_comparisons']}")
            print(f"Exact Matches: {summary['content_stats']['exact_matches']} ({summary['content_stats']['exact_match_rate']:.1f}%)")
            print(f"Practical Matches: {summary['content_stats']['practical_matches']} ({summary['content_stats']['practical_match_rate']:.1f}%)")
        print(f"{'='*60}")

    def generate_batch_summary(self, results):
        """Generate summary statistics for batch evaluation results"""
        if not results:
            return {"total_files": 0, "message": "No results to summarize"}

        total_files = len(results)
        passed_files = sum(1 for r in results if r.passed)
        failed_files = total_files - passed_files
        skipped_files = sum(1 for r in results if r.metrics.get('comparison_skipped', False))
        files_with_references = sum(1 for r in results if r.metrics.get('reference_found', False))

        # Score statistics
        scores = [r.score for r in results if not r.metrics.get('comparison_skipped', False)]
        avg_score = sum(scores) / len(scores) if scores else 0

        # Content matching statistics
        content_stats = {
            "total_comparisons": 0,
            "exact_matches": 0,
            "practical_matches": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0
        }

        macro_precision_values = []
        macro_recall_values = []
        macro_f1_values = []

        for result in results:
            if not result.metrics.get('comparison_skipped', False):
                summary_stats = result.metrics.get('summary_stats', {})
                content_stats["total_comparisons"] += summary_stats.get('total_elements', 0)
                content_stats["exact_matches"] += summary_stats.get('exact_matches', 0)
                content_stats["practical_matches"] += summary_stats.get('practical_matches', 0)
                content_metrics = summary_stats.get('content_metrics', {})
                counts = content_metrics.get('counts', {})
                content_stats["tp"] += counts.get('tp', 0)
                content_stats["fp"] += counts.get('fp', 0)
                content_stats["fn"] += counts.get('fn', 0)
                if content_metrics.get('available'):
                    macro_precision_values.append(content_metrics.get('macro_precision', 0))
                    macro_recall_values.append(content_metrics.get('macro_recall', 0))
                    macro_f1_values.append(content_metrics.get('macro_f1', 0))

        # Calculate rates
        exact_match_rate = (content_stats["exact_matches"] / content_stats["total_comparisons"] * 100) if content_stats["total_comparisons"] > 0 else 0
        practical_match_rate = (content_stats["practical_matches"] / content_stats["total_comparisons"] * 100) if content_stats["total_comparisons"] > 0 else 0
        content_match_rate = (passed_files / files_with_references * 100) if files_with_references > 0 else 0

        precision = (content_stats["tp"] / (content_stats["tp"] + content_stats["fp"]) * 100) if (content_stats["tp"] + content_stats["fp"]) > 0 else (100.0 if content_stats["tp"] == 0 and content_stats["fp"] == 0 else 0.0)
        recall = (content_stats["tp"] / (content_stats["tp"] + content_stats["fn"]) * 100) if (content_stats["tp"] + content_stats["fn"]) > 0 else (100.0 if content_stats["tp"] == 0 and content_stats["fn"] == 0 else 0.0)

        content_stats.update({
            "exact_match_rate": exact_match_rate,
            "practical_match_rate": practical_match_rate,
            "precision": precision,
            "recall": recall
        })

        if content_stats["precision"] + content_stats["recall"] > 0:
            micro_f1 = (2 * content_stats["precision"] * content_stats["recall"]) / (content_stats["precision"] + content_stats["recall"])
        else:
            micro_f1 = 0.0
        content_stats["micro_f1"] = micro_f1
        content_stats["micro_precision"] = precision
        content_stats["micro_recall"] = recall
        content_stats["f1"] = micro_f1

        if macro_precision_values:
            content_stats["macro_precision"] = sum(macro_precision_values) / len(macro_precision_values)
            content_stats["macro_recall"] = sum(macro_recall_values) / len(macro_recall_values)
            content_stats["macro_f1"] = sum(macro_f1_values) / len(macro_f1_values)
        else:
            content_stats["macro_precision"] = 0.0
            content_stats["macro_recall"] = 0.0
            content_stats["macro_f1"] = 0.0

        return {
            "total_files": total_files,
            "passed_files": passed_files,
            "failed_files": failed_files,
            "skipped_files": skipped_files,
            "files_with_references": files_with_references,
            "average_score": avg_score,
            "content_match_rate": content_match_rate,
            "content_stats": content_stats
        }

    def evaluate_batch(self, input_path: str, pattern: str = "*.xml", config=None):
        """Evaluate multiple XML files for content matching"""
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

        self.logger.info("Found %d XML files to evaluate", len(xml_files))
        if not self.quiet:
            print(f"Found {len(xml_files)} XML files to evaluate")
            if self.reference_directory:
                print(f"Reference Directory: {self.reference_directory}")
            else:
                print("[WARNING] No reference directory configured")
            print("=" * 60)

        # Try to load D1 results for context-aware weighting
        d1_content_map = self._load_d1_content_preservation_map(config)

        for xml_file in sorted(xml_files):
            self.logger.info("Evaluating: %s", xml_file.name)
            if not self.quiet:
                print(f"Evaluating: {xml_file.name}")
            try:
                # Create per-file config with D1 data if available
                file_config = dict(config) if config else {}
                file_name = xml_file.name
                if file_name in d1_content_map:
                    file_config['d1_content_preserved'] = d1_content_map[file_name]

                result = self.evaluate_file(str(xml_file), file_config)
                # Store the file path in metrics for later reference
                result.metrics['file_path'] = str(xml_file)
                result.metrics['file_name'] = xml_file.name
                results.append(result)

                # Quick summary for each file
                if result.metrics.get('comparison_skipped'):
                    self.logger.info("Skipped - no reference file available")
                    if not self.quiet:
                        print("   SKIP | No reference file available")
                else:
                    status = "PASS" if result.passed else "FAIL"
                    self.logger.info("%s | Score: %.1f/100", status, result.score)
                    if not self.quiet:
                        print(f"   {status} | Score: {result.score:5.1f}/100")

                        # Show quick content matching details
                        summary_stats = result.metrics.get('summary_stats', {})
                        if summary_stats.get('total_elements', 0) > 0:
                            exact_rate = summary_stats.get('exact_match_rate', 0)
                            practical_rate = summary_stats.get('practical_match_rate', 0)
                            print(f"      Content: {exact_rate:.0f}% exact, {practical_rate:.0f}% practical matches")

            except (IOError, OSError, RuntimeError, ValueError, etree.XMLSyntaxError) as e:
                self.logger.error("Error evaluating file: %s", str(e), exc_info=True)
                if not self.quiet:
                    print(f"   ERROR: {str(e)}")
                # Create error result for failed evaluation
                error_result = EvaluationResult(
                    dimension=4,
                    passed=False,
                    score=0.0,
                    errors=[Error(
                        type=ErrorType.ENTITY_RECOGNITION,
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
