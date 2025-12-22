# utils/structural_difference_analyzer.py

from typing import Dict, List, Any
from lxml import etree

class StructuralDifferenceAnalyzer:
    """Simple version of structural difference analyzer"""

    def __init__(self):
        self.ignore_elements = {'note'}

    def analyze_detailed_differences(self, xml_body: etree._Element, ref_body: etree._Element) -> Dict[str, Any]:
        """Analyze structural differences between XML and reference"""

        # Get element lists (all occurrences)
        xml_elements = self._get_element_list(xml_body)
        ref_elements = self._get_element_list(ref_body)

        # Get element types (unique types)
        xml_types = set(xml_elements)
        ref_types = set(ref_elements)

        added_types = xml_types - ref_types
        removed_types = ref_types - xml_types

        # Also get element counts per type for more detailed analysis
        xml_counts = {}
        ref_counts = {}

        for elem in xml_elements:
            xml_counts[elem] = xml_counts.get(elem, 0) + 1
        for elem in ref_elements:
            ref_counts[elem] = ref_counts.get(elem, 0) + 1

        # Get counts for ALL element types (including unchanged ones)
        count_changes = {}
        all_types = xml_types | ref_types
        for elem_type in all_types:
            xml_count = xml_counts.get(elem_type, 0)
            ref_count = ref_counts.get(elem_type, 0)
            # Include ALL types, not just changed ones
            count_changes[elem_type] = {
                'xml_count': xml_count,
                'ref_count': ref_count,
                'difference': xml_count - ref_count
            }

        # Basic summary
        summary = {
            "element_count_diff": len(xml_elements) - len(ref_elements),
            "xml_element_count": len(xml_elements),
            "ref_element_count": len(ref_elements),
            "added_element_types": list(added_types),
            "removed_element_types": list(removed_types),
            "element_count_changes": count_changes
        }

        return {
            "summary": summary,
            "xmldiff_details": {"available": False, "message": "Simplified analyzer"},
            "tree_edit_details": {"available": False, "message": "Simplified analyzer"}
        }

    def _get_element_list(self, element: etree._Element) -> List[str]:
        """Get list of all element names"""
        elements = []

        def collect(elem):
            # Skip non-element nodes (comments, processing instructions, etc.)
            # Check if it's actually an Element (has a string tag)
            if not isinstance(elem.tag, str):
                # This is likely a Comment or ProcessingInstruction - skip it
                # But still process any children (though comments typically don't have children)
                for child in elem:
                    collect(child)
                return

            # Ensure tag is a string (defensive programming)
            tag_str = str(elem.tag)
            name = tag_str.split('}')[-1] if '}' in tag_str else tag_str
            if name not in self.ignore_elements:
                elements.append(name)
            for child in elem:
                collect(child)

        collect(element)
        return elements

    def format_analysis_report(self, analysis: Dict[str, Any]) -> str:
        """Format analysis into readable report"""
        summary = analysis["summary"]

        report = []
        report.append("[DATA] BASIC STRUCTURAL ANALYSIS:")
        report.append(f"Element Count: XML={summary['xml_element_count']}, Reference={summary['ref_element_count']}")

        if summary['element_count_diff'] != 0:
            diff_type = "more" if summary['element_count_diff'] > 0 else "fewer"
            report.append(f"Difference: {abs(summary['element_count_diff'])} {diff_type} elements in XML")

        if summary['added_element_types']:
            report.append(f"Added element types: {', '.join(summary['added_element_types'])}")

        if summary['removed_element_types']:
            report.append(f"Removed element types: {', '.join(summary['removed_element_types'])}")

        # Show element count changes
        count_changes = summary.get('element_count_changes', {})
        if count_changes:
            report.append(f"\nElement count changes:")
            for elem_type, change in count_changes.items():
                xml_count = change['xml_count']
                ref_count = change['ref_count']
                diff = change['difference']
                if diff > 0:
                    report.append(f"  {elem_type}: {ref_count} → {xml_count} (+{diff})")
                else:
                    report.append(f"  {elem_type}: {ref_count} → {xml_count} ({diff})")

        return "\n".join(report)