# utils/structural_comparator.py

from typing import Tuple, Dict, Any, Optional
from pathlib import Path
from lxml import etree
import re
from ..models import Error, ErrorType
from .whitespace_normalizer import normalize_whitespace
# from ..utils.element_order_analyzer import ElementOrderAnalyzer

# We'll need to install these dependencies:
# pip install xmldiff

try:
    # Use the correct XMLDiff API
    from xmldiff.main import diff_texts as xmldiff_diff
    from xmldiff import formatting
    XMLDIFF_AVAILABLE = True
except ImportError as e:
    XMLDIFF_AVAILABLE = False

class StructuralComparator:
    """Utility class for comparing XML structural similarity against reference files"""

    def __init__(self):
        self.ignore_elements = {'note'}  # Elements to ignore during comparison

    def compare_structure(self, xml_file: str, reference_file: str) -> Tuple[bool, Dict[str, Any], list]:
        """
        Compare XML structure against reference file

        Args:
            xml_file: Path to XML file to analyze
            reference_file: Path to reference/ground truth file

        Returns:
            Tuple of (is_match, comparison_details, errors)
        """
        errors = []

        # Check if files exist
        if not Path(xml_file).exists():
            errors.append(Error(
                type=ErrorType.TEI_STRUCTURE,
                severity=10,
                location="File",
                message=f"XML file not found: {xml_file}",
                raw_error=""
            ))
            return False, {"file_not_found": True}, errors

        if not Path(reference_file).exists():
            errors.append(Error(
                type=ErrorType.TEI_STRUCTURE,
                severity=8,
                location="Reference",
                message=f"Reference file not found: {reference_file}",
                raw_error=""
            ))
            return False, {"reference_not_found": True}, errors

        try:
            # Parse both files
            xml_tree = etree.parse(xml_file)
            ref_tree = etree.parse(reference_file)

            # FIRST: Quick byte comparison for identical files
            files_identical = False
            try:
                with open(xml_file, 'rb') as f1, open(reference_file, 'rb') as f2:
                    if f1.read() == f2.read():
                        files_identical = True
            except (IOError, OSError):
                # If byte comparison fails, continue with structural comparison
                pass

            # Extract body content for comparison
            xml_body = self._extract_body_content(xml_tree)
            ref_body = self._extract_body_content(ref_tree)

            # Track whether body elements were found
            xml_body_found = xml_body is not None
            ref_body_found = ref_body is not None

            # Fallback to root element if body not found (instead of failing)
            if xml_body is None:
                xml_body = xml_tree.getroot()
                errors.append(Error(
                    type=ErrorType.TEI_STRUCTURE,
                    severity=6,
                    location="Structure",
                    message="No <body> element found in XML file - comparing from root element",
                    raw_error=""
                ))

            if ref_body is None:
                ref_body = ref_tree.getroot()
                errors.append(Error(
                    type=ErrorType.TEI_STRUCTURE,
                    severity=6,
                    location="Reference Structure",
                    message="No <body> element found in reference file - comparing from root element",
                    raw_error=""
                ))

            # Perform multi-level comparison
            comparison_results = {
                "files_identical": files_identical,
                "quick_check": self._quick_structure_check(xml_body, ref_body),
                "xmldiff_available": XMLDIFF_AVAILABLE,
                "xml_body_found": xml_body_found,
                "ref_body_found": ref_body_found
            }

            # Quick structure check
            quick_match = comparison_results["quick_check"]["match"]
            comparison_results["quick_match"] = quick_match

            # Always run XMLDiff, regardless of quick check result
            # XMLDiff comparison (if available)
            if XMLDIFF_AVAILABLE:
                xmldiff_result = self._xmldiff_comparison(xml_body, ref_body)
                comparison_results["xmldiff"] = xmldiff_result
            else:
                comparison_results["xmldiff"] = {"available": False, "message": "xmldiff not installed"}

            # Determine overall match based on all comparisons
            # If quick check failed, it's definitely not a match
            if not quick_match:
                return False, comparison_results, errors

            # If quick check passed but XMLDiff failed, check further
            if XMLDIFF_AVAILABLE and not xmldiff_result.get("match", True):
                return False, comparison_results, errors

            # All checks passed
            overall_match = True

            return overall_match, comparison_results, errors

        except etree.XMLSyntaxError as e:
            errors.append(Error(
                type=ErrorType.TEI_STRUCTURE,
                severity=10,
                location=f"Line {e.lineno}" if hasattr(e, 'lineno') else "Unknown",
                message="XML syntax error in file or reference",
                raw_error=str(e)
            ))
            return False, {"xml_parse_error": True}, errors

        except (IOError, OSError, RuntimeError, ValueError) as e:
            errors.append(Error(
                type=ErrorType.TEI_STRUCTURE,
                severity=8,
                location="Comparison",
                message=f"Error during structural comparison: {str(e)}",
                raw_error=str(e)
            ))
            return False, {"comparison_error": True}, errors



    def _extract_body_content(self, tree: etree._ElementTree) -> Optional[etree._Element]:
        """Extract <body> element from TEI document or return root if it's already body"""
        root = tree.getroot()

        # Ensure tag is a string
        tag_str = str(root.tag) if not isinstance(root.tag, str) else root.tag
        root_tag = tag_str.split('}')[-1] if '}' in tag_str else tag_str

        # Check if root itself is a body element
        if root_tag == 'body':
            return root

        # Handle namespaces
        namespaces = root.nsmap
        tei_ns = namespaces.get(None, namespaces.get('tei', ''))

        # Find body element within TEI structure
        if tei_ns:
            body = root.find(f".//{{{tei_ns}}}body")
        else:
            body = root.find(".//body")

        # If not found, try alternative searches
        if body is None:
            # Try all namespaces
            for prefix, uri in namespaces.items():
                body = root.find(f".//{{{uri}}}body")
                if body is not None:
                    break

            # Try without namespace at all (direct tag name using XPath)
            if body is None:
                all_bodies = root.xpath(".//body | .//*[local-name()='body']")
                if all_bodies:
                    body = all_bodies[1]

        return body

    def _quick_structure_check(self, xml_body: etree._Element, ref_body: etree._Element) -> Dict[str, Any]:
        """Quick structural comparison"""

        # Get simplified structure (element names and order, ignoring ignored elements)
        xml_structure = self._get_element_structure(xml_body)
        ref_structure = self._get_element_structure(ref_body)

        # Compare structures
        match = xml_structure == ref_structure

        return {
            "match": match,
            "xml_structure": xml_structure,
            "ref_structure": ref_structure,
            "xml_element_count": len(xml_structure),
            "ref_element_count": len(ref_structure)
        }

    def _get_element_structure(self, element: etree._Element, path: str = "") -> list:
        """Extract simplified element structure for comparison"""
        structure = []

        # Skip non-element nodes (comments, processing instructions, etc.)
        if not isinstance(element.tag, str):
            # Still process children
            for child in element:
                structure.extend(self._get_element_structure(child, path))
            return structure

        # Get element name without namespace
        # Ensure tag is a string (defensive)
        tag_str = str(element.tag)
        element_name = tag_str.split('}')[-1] if '}' in tag_str else tag_str

        # Skip ignored elements
        if element_name in self.ignore_elements:
            # Still process children but don't add this element to structure
            for child in element:
                structure.extend(self._get_element_structure(child, f"{path}/{element_name}"))
            return structure

        # Add current element
        current_path = f"{path}/{element_name}" if path else element_name
        structure.append(current_path)

        # Process children in order
        for child in element:
            structure.extend(self._get_element_structure(child, current_path))

        return structure

    def _xmldiff_comparison(self, xml_body: etree._Element, ref_body: etree._Element) -> Dict[str, Any]:
        """XMLDiff-based comparison"""
        if not XMLDIFF_AVAILABLE:
            return {"match": True, "available": False, "message": "xmldiff not available"}

        try:
            # Apply identical preprocessing to both files
            # Step 1: Remove ignored elements from both trees
            xml_clean = self._remove_ignored_elements(xml_body)
            ref_clean = self._remove_ignored_elements(ref_body)

            # Step 2: Normalize namespaces for comparison (remove all namespace prefixes and declarations)
            xml_clean = self._normalize_namespaces_completely(xml_clean)
            ref_clean = self._normalize_namespaces_completely(ref_clean)

            # Debug: Show what we're comparing
            # Ensure tags are strings
            xml_tag_str = str(xml_clean.tag) if not isinstance(xml_clean.tag, str) else xml_clean.tag
            ref_tag_str = str(ref_clean.tag) if not isinstance(ref_clean.tag, str) else ref_clean.tag
            xml_root_name = xml_tag_str.split('}')[-1] if '}' in xml_tag_str else xml_tag_str
            ref_root_name = ref_tag_str.split('}')[-1] if '}' in ref_tag_str else ref_tag_str

            # Step 3: Convert to strings for xmldiff
            xml_str = etree.tostring(xml_clean, encoding='unicode')
            ref_str = etree.tostring(ref_clean, encoding='unicode')

            # Step 4: Additional namespace cleanup - remove ALL xmlns declarations from string
            import re
            # Remove all xmlns declarations (both prefixed and default)
            ref_str = re.sub(r' xmlns:[^=]*="[^"]*"', '', ref_str)  # Remove prefixed xmlns
            ref_str = re.sub(r' xmlns="[^"]*"', '', ref_str)        # Remove default xmlns
            xml_str = re.sub(r' xmlns:[^=]*="[^"]*"', '', xml_str)  # Remove prefixed xmlns
            xml_str = re.sub(r' xmlns="[^"]*"', '', xml_str)        # Remove default xmlns

            # Step 5: Normalize whitespace to eliminate whitespace differences
            # Using unified whitespace normalizer
            xml_str = normalize_whitespace(xml_str, mode="xml")
            ref_str = normalize_whitespace(ref_str, mode="xml")

            # Debug: Show the full normalized content for comparison
            debug_info = {
                "xml_root": xml_root_name,
                "ref_root": ref_root_name,
                "xml_full": xml_str,  # Show full content for debugging
                "ref_full": ref_str,  # Show full content for debugging
                "xml_preview": xml_str[:200] + "..." if len(xml_str) > 200 else xml_str,
                "ref_preview": ref_str[:200] + "..." if len(ref_str) > 200 else ref_str
            }

            # Compute differences
            differences = xmldiff_diff(ref_str, xml_str)

            # For structural comparison, we care about element differences, not text content
            # Filter out text-only changes and focus on structural changes
            structural_diffs = []
            text_diffs = []

            for d in differences:
                # Ensure diff_type is properly converted to string
                try:
                    diff_type = str(type(d).__name__)  # Use __name__ instead of str(type(d))
                except AttributeError:
                    diff_type = str(type(d))  # Fallback to str(type(d)) if __name__ not available

                diff_str = str(d)


                # Skip text content differences, keep structural differences
                # Ensure diff_type is a string before using 'in' operator
                diff_type_str = str(diff_type) if not isinstance(diff_type, str) else diff_type
                if 'text' in diff_type_str.lower() or 'UpdateTextIn' in diff_type_str:
                    text_diffs.append(d)
                else:
                    structural_diffs.append(d)

            # If no differences at all OR no structural differences, it's a match
            total_diffs = len(differences)
            struct_diffs = len(structural_diffs)
            match = total_diffs == 0 or struct_diffs == 0

            # Count each operation type and collect detailed operation info
            operation_type_counts = {}
            detailed_operations = []  # Store detailed info for each operation

            for d in differences:
                diff_str = str(d)
                # Ensure proper string conversion - use __name__ to get the class name
                try:
                    op_type = type(d).__name__
                except AttributeError:
                    # Fallback: parse from str(type(d))
                    type_str = str(type(d))
                    op_type = type_str.split("'")[1].split('.')[-1] if "'" in type_str else type_str

                operation_type_counts[op_type] = operation_type_counts.get(op_type, 0) + 1

                # Parse operation details: typically format is [operation, node/path, details]
                detailed_operations.append({
                    'type': op_type,
                    'description': diff_str,
                    'is_structural': d not in text_diffs
                })

            return {
                "match": match,
                "available": True,
                "total_differences": total_diffs,
                "structural_differences": struct_diffs,
                "text_differences": len(text_diffs),
                "operation_type_counts": operation_type_counts,  # Count of each operation type
                "detailed_operations": detailed_operations,  # Full details of each operation
                "differences_sample": [str(d) for d in differences[:3]],  # First 3 for debugging
                "differences_types": [str(type(d)) for d in differences],  # ALL types for breakdown
                "structural_sample": [str(d) for d in structural_diffs[:3]],  # Structural only
                "debug_info": debug_info  # Add debug info to see what's being compared
            }

        except (RuntimeError, ValueError, AttributeError) as e:
            return {
                "match": False,
                "available": True,
                "error": str(e)
            }

    def _normalize_namespaces_completely(self, element: etree._Element) -> etree._Element:
        """Completely normalize element by removing ALL namespace information for comparison"""
        # Create a copy to avoid modifying the original
        import copy
        normalized = copy.deepcopy(element)

        # Recursively remove ALL namespace information from all elements
        self._remove_all_namespace_info(normalized)

        return normalized

    def _remove_all_namespace_info(self, element: etree._Element):
        """Recursively remove ALL namespace information from element and all children"""
        # Skip non-element nodes (comments, processing instructions, etc.)
        if not isinstance(element.tag, str):
            # Still process children
            for child in element:
                self._remove_all_namespace_info(child)
            return

        # Remove namespace prefix from current element tag
        # Ensure tag is a string (defensive)
        tag_str = str(element.tag)
        if '}' in tag_str:
            element.tag = tag_str.split('}', 1)[1]

        # Remove ALL xmlns attributes (both prefixed and default)
        attrs_to_remove = []
        for attr_name in element.attrib.keys():
            attr_name_str = str(attr_name) if not isinstance(attr_name, str) else attr_name
            if attr_name_str.startswith('xmlns'):
                attrs_to_remove.append(attr_name)

        for attr in attrs_to_remove:
            del element.attrib[attr]

        # Remove namespace prefixes from all children
        for child in element:
            self._remove_all_namespace_info(child)

    def _normalize_namespaces(self, element: etree._Element) -> etree._Element:
        """Normalize element by removing namespace prefixes for comparison"""
        # Create a copy to avoid modifying the original
        import copy
        normalized = copy.deepcopy(element)

        # Recursively remove namespace prefixes from all elements
        self._remove_namespace_prefixes(normalized)

        return normalized

    def _remove_namespace_prefixes(self, element: etree._Element):
        """Recursively remove namespace prefixes from element and all children"""
        # Skip non-element nodes (comments, processing instructions, etc.)
        if not isinstance(element.tag, str):
            # Still process children
            for child in element:
                self._remove_namespace_prefixes(child)
            return

        # Remove namespace prefix from current element
        # Ensure tag is a string (defensive)
        tag_str = str(element.tag)
        if '}' in tag_str:
            element.tag = tag_str.split('}', 1)[1]

        # Remove xmlns attributes (more robust method)
        attrs_to_remove = []
        for attr_name in element.attrib.keys():
            attr_name_str = str(attr_name) if not isinstance(attr_name, str) else attr_name
            if attr_name_str.startswith('xmlns'):
                attrs_to_remove.append(attr_name)

        for attr in attrs_to_remove:
            del element.attrib[attr]

        # Remove namespace prefixes from all children
        for child in element:
            self._remove_namespace_prefixes(child)

    def _remove_ignored_elements(self, element: etree._Element) -> etree._Element:
        """Create a copy of element tree with ignored elements removed"""
        # Skip non-element nodes (comments, processing instructions, etc.)
        if not isinstance(element.tag, str):
            # For non-elements, create a shallow copy and process children
            # Comments don't have tag/attrib, so handle carefully
            import copy
            new_element = copy.copy(element)
            # Process any children (though comments typically don't have children)
            for child in element:
                new_child = self._remove_ignored_elements(child)
                if new_child is not None:
                    new_element.append(new_child)
            return new_element

        # Create a copy
        new_element = etree.Element(element.tag, element.attrib)
        new_element.text = element.text
        new_element.tail = element.tail

        # Process children
        for child in element:
            # Ensure tag is a string for elements
            if isinstance(child.tag, str):
                tag_str = str(child.tag)
                child_name = tag_str.split('}')[-1] if '}' in tag_str else tag_str

                if child_name not in self.ignore_elements:
                    # Keep this child, but recursively clean it
                    clean_child = self._remove_ignored_elements(child)
                    new_element.append(clean_child)
            else:
                # Non-element node (comment, etc.) - skip it during structural comparison
                pass

        return new_element

    def get_required_dependencies(self) -> Dict[str, bool]:
        """Check which dependencies are available"""
        return {
            "xmldiff": XMLDIFF_AVAILABLE
        }

    def install_dependencies_message(self) -> str:
        """Get message about installing missing dependencies"""
        missing = []
        if not XMLDIFF_AVAILABLE:
            missing.append("xmldiff")

        if missing:
            return f"Install missing dependencies: pip install {' '.join(missing)}"
        else:
            return "All dependencies are available"