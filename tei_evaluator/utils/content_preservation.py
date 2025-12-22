# content_preservation.py

import re
import unicodedata
from typing import Tuple, List, Dict, Any
from lxml import etree
from ..models import Error, ErrorType
from .whitespace_normalizer import normalize_whitespace_completely, normalize_whitespace

class ContentPreservationAnalyzer:
    """
    Analyzes content fidelity between source transcription and XML output,
    implementing normalized string comparison with whitespace-aware and whitespace-agnostic modes.
    """

    def __init__(self):
        # XML entities that should be normalized back to their character equivalents
        self.xml_entities = {
            '&amp;': '&',
            '&lt;': '<',
            '&gt;': '>',
            '&quot;': '"',
            '&apos;': "'"
        }

        # Numeric character references pattern
        self.numeric_entities_pattern = re.compile(r'&#(\d+);|&#x([0-9a-fA-F]+);')

    def analyze_content_preservation(self, original_text: str, xml_output: str) -> Tuple[Dict[str, Any], List[Error]]:
        """
        Compare original transcription with XML output for content fidelity.

        Args:
            original_text: Source transcription text
            xml_output: Generated XML content

        Returns:
            Tuple of (comparison_results, errors_list)
        """
        try:
            # Step 1: Extract text content from XML
            extracted_text = self._extract_text_from_xml(xml_output)

            # Step 2: Normalize entities and Unicode (preserve whitespace structure for now)
            entities_normalized_original = self._normalize_entities_only(original_text)
            entities_normalized_extracted = self._normalize_entities_only(extracted_text)

            # Step 3: Create whitespace-normalized versions (normalize to single spaces, preserve word boundaries)
            # Using unified whitespace normalizer
            normalized_original = normalize_whitespace(entities_normalized_original, mode="text")
            normalized_extracted = normalize_whitespace(entities_normalized_extracted, mode="text")

            # Step 4: Create whitespace-agnostic versions (remove all whitespace)
            # Using unified whitespace normalizer
            no_ws_original = normalize_whitespace_completely(normalized_original)
            no_ws_extracted = normalize_whitespace_completely(normalized_extracted)

            # Step 5: Perform comparisons
            exact_match_with_ws = (normalized_original == normalized_extracted)
            exact_match_without_ws = (no_ws_original == no_ws_extracted)

            # Step 6: Calculate similarity scores if not exact matches
            similarity_with_ws = 1.0 if exact_match_with_ws else self._calculate_similarity(normalized_original, normalized_extracted)
            similarity_without_ws = 1.0 if exact_match_without_ws else self._calculate_similarity(no_ws_original, no_ws_extracted)

            # Step 7: Generate diff information ONLY if actual content differs (not just whitespace)
            diff_info = None
            if not exact_match_without_ws:
                # Content actually differs - generate diff from normalized versions for readable context
                diff_info = self._generate_context_aware_diff(normalized_original, normalized_extracted)
                # Filter out whitespace-only changes
                diff_info = self._filter_whitespace_only_diffs(diff_info, no_ws_original, no_ws_extracted)

            results = {
                "exact_match_with_whitespace": exact_match_with_ws,
                "exact_match_without_whitespace": exact_match_without_ws,
                "similarity_with_whitespace": similarity_with_ws,
                "similarity_without_whitespace": similarity_without_ws,
                "original_length": len(original_text),
                "extracted_length": len(extracted_text),
                "normalized_original_length": len(normalized_original),
                "normalized_extracted_length": len(normalized_extracted),
                "diff_info": diff_info
            }

            # Generate errors for significant issues
            errors = self._generate_errors(results)

            return results, errors

        except (etree.XMLSyntaxError, IOError, ValueError, UnicodeDecodeError) as e:
            error = Error(
                type=ErrorType.CONTENT_PRESERVATION,
                severity=8,
                location="Content Analysis",
                message=f"Content preservation analysis failed: {str(e)}",
                raw_error=str(e)
            )
            return {"analysis_error": str(e)}, [error]

    def _extract_text_from_xml(self, xml_content: str) -> str:
        """
        Extract text content from XML, preserving original spacing.
        """
        try:
            # Try to parse with lxml first for proper handling
            root = etree.fromstring(xml_content.encode('utf-8'))

            # Extract all text content, preserving original whitespace
            text_parts = []
            self._extract_text_improved_recursive(root, text_parts)

            # Join the text parts and normalize only excessive whitespace
            extracted = ''.join(text_parts)

            # Only normalize line breaks and excessive whitespace, preserve single spaces
            # Replace multiple whitespace chars with single space, but preserve word boundaries
            extracted = re.sub(r'[ \t]+', ' ', extracted)  # Multiple spaces/tabs to single space
            extracted = re.sub(r'\n\s*', ' ', extracted)    # Newlines with possible leading whitespace to space
            extracted = extracted.strip()

            return extracted

        except etree.XMLSyntaxError:
            # If XML is malformed, use regex fallback
            return self._extract_text_regex_fallback(xml_content)

    def _extract_text_improved_recursive(self, element, text_parts: List[str]):
        """
        Recursive text extraction - extract ALL text content.
        """
        # Add element text if present
        if element.text:
            text_parts.append(element.text)

        # Process child elements
        for child in element:
            self._extract_text_improved_recursive(child, text_parts)

            # Add tail text after child element
            if child.tail:
                text_parts.append(child.tail)

    def _extract_text_regex_fallback(self, xml_content: str) -> str:
        """
        Fallback text extraction using regex when XML parsing fails.
        """
        text = xml_content

        # Remove XML comments
        text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)

        # Remove processing instructions
        text = re.sub(r'<\?.*?\?>', '', text)

        # Remove TEI header completely
        text = re.sub(r'<teiHeader.*?</teiHeader>', '', text, flags=re.DOTALL)

        # Remove all XML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Clean up whitespace - replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        return text

    def _normalize_entities_only(self, text: str) -> str:
        """
        Normalize only XML entities and Unicode, preserve whitespace structure.
        """
        # Step 1: Resolve XML entities
        normalized = text
        for entity, char in self.xml_entities.items():
            normalized = normalized.replace(entity, char)

        # Resolve numeric character references
        def replace_numeric_entity(match):
            if match.group(1):  # Decimal
                code_point = int(match.group(1))
            else:  # Hexadecimal
                code_point = int(match.group(2), 16)
            try:
                return chr(code_point)
            except ValueError:
                return match.group(0)  # Keep original if invalid

        normalized = self.numeric_entities_pattern.sub(replace_numeric_entity, normalized)

        # Step 2: Unicode normalization to NFC form
        normalized = unicodedata.normalize('NFC', normalized)

        return normalized

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity using difflib.SequenceMatcher.
        """
        if not text1 and not text2:
            return 1.0
        if not text1 or not text2:
            return 0.0

        from difflib import SequenceMatcher
        matcher = SequenceMatcher(None, text1, text2)
        return matcher.ratio()

    def _generate_context_aware_diff(self, original: str, extracted: str) -> List[Dict[str, Any]]:
        """
        Generate context-aware diff information showing changes with surrounding context.
        """
        from difflib import SequenceMatcher

        matcher = SequenceMatcher(None, original, extracted)
        changes = []

        for opcode in matcher.get_opcodes():
            operation, i1, i2, j1, j2 = opcode

            if operation == 'replace':
                original_text = original[i1:i2]
                extracted_text = extracted[j1:j2]
                context = self._get_context(original, i1, i2)

                changes.append({
                    'type': 'replace',
                    'change': f'{original_text} → {extracted_text}',
                    'original_text': original_text,
                    'extracted_text': extracted_text,
                    'context': context,
                    'position': i1
                })

            elif operation == 'delete':
                deleted_text = original[i1:i2]
                context = self._get_context(original, i1, i2)

                changes.append({
                    'type': 'delete',
                    'change': f'deleted: {deleted_text}',
                    'original_text': deleted_text,
                    'extracted_text': '',
                    'context': context,
                    'position': i1
                })

            elif operation == 'insert':
                inserted_text = extracted[j1:j2]
                # For inserts, we need to find where it would be in the original
                context = self._get_context(original, i1, i1)  # Point insertion

                changes.append({
                    'type': 'insert',
                    'change': f'inserted: {inserted_text}',
                    'original_text': '',
                    'extracted_text': inserted_text,
                    'context': context,
                    'position': i1
                })

        return changes

    def _get_context(self, text: str, start: int, end: int, context_size: int = 20) -> str:
        """
        Get surrounding context for a change, showing text before and after.
        """
        before_start = max(0, start - context_size)
        after_end = min(len(text), end + context_size)

        before = text[before_start:start]
        changed = text[start:end]
        after = text[end:after_end]

        # Add ellipsis if we truncated
        if before_start > 0:
            before = '...' + before
        if after_end < len(text):
            after = after + '...'

        return f'{before}[{changed}]{after}'

    def _filter_whitespace_only_diffs(self, diff_info: List[Dict[str, Any]], no_ws_original: str, no_ws_extracted: str) -> List[Dict[str, Any]]:
        """
        Filter out differences that are purely whitespace changes.

        If the texts match when whitespace is removed, any reported differences
        are just whitespace variations (newlines, spaces, tabs) which are not
        meaningful for content preservation evaluation.

        Args:
            diff_info: List of difference dictionaries from difflib
            no_ws_original: Original text with all whitespace removed
            no_ws_extracted: Extracted text with all whitespace removed

        Returns:
            Filtered list containing only actual content differences
        """
        if not diff_info:
            return diff_info

        # If texts match without whitespace, all diffs are whitespace-only - remove them all
        if no_ws_original == no_ws_extracted:
            return []

        # Otherwise, filter individual diffs that are whitespace-only
        filtered_diffs = []
        for diff in diff_info:
            orig_text = diff.get('original_text', '')
            extr_text = diff.get('extracted_text', '')

            # Remove all whitespace from this specific diff
            orig_no_ws = re.sub(r'\s', '', orig_text)
            extr_no_ws = re.sub(r'\s', '', extr_text)

            # Keep this diff only if there's an actual content difference
            if orig_no_ws != extr_no_ws:
                filtered_diffs.append(diff)

        return filtered_diffs

    def _generate_errors(self, results: Dict[str, Any]) -> List[Error]:
        """
        Generate error objects based on comparison results.
        """
        errors = []

        # Check if whitespace-agnostic comparison fails (indicates missing/added content)
        if not results.get('exact_match_without_whitespace', False):
            similarity = results.get('similarity_without_whitespace', 0.0)

            if similarity < 0.95:  # Less than 95% similarity
                errors.append(Error(
                    type=ErrorType.CONTENT_PRESERVATION,
                    severity=7,
                    location="Content Comparison",
                    message=f"Significant content differences detected (similarity: {similarity:.1%})",
                    raw_error=f"Original length: {results.get('original_length', 0)}, Extracted length: {results.get('extracted_length', 0)}"
                ))

        # Check for significant length differences (potential hallucination or content loss)
        original_len = results.get('normalized_original_length', 1)
        extracted_len = results.get('normalized_extracted_length', 0)

        if extracted_len > original_len * 1.2:  # More than 20% longer
            errors.append(Error(
                type=ErrorType.CONTENT_PRESERVATION,
                severity=8,
                location="Content Analysis",
                message=f"Possible content hallucination detected (extracted text {extracted_len/original_len:.1f}x longer)",
                raw_error=f"Original: {original_len} chars, Extracted: {extracted_len} chars"
            ))

        if extracted_len < original_len * 0.8:  # More than 20% shorter
            errors.append(Error(
                type=ErrorType.CONTENT_PRESERVATION,
                severity=8,
                location="Content Analysis",
                message=f"Significant content loss detected (extracted text {extracted_len/original_len:.1f}x shorter)",
                raw_error=f"Original: {original_len} chars, Extracted: {extracted_len} chars"
            ))

        return errors