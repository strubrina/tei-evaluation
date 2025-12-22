import re
from typing import Dict, List

class DocumentStructureAnalyzer:
    """Analyze XML document structure issues"""

    def __init__(self):
        self.error_categories = {}

    def analyze_document_structure(self, xml_string: str) -> Dict[str, int]:
        """
        Analyze overall document structure

        Args:
            xml_string: XML content to analyze

        Returns:
            Dictionary of error categories with counts
        """
        self.error_categories.clear()

        # Check for multiple root elements
        root_count = self._count_root_elements(xml_string)
        if root_count > 1:
            self.error_categories['multiple_roots'] = root_count

        return self.error_categories

    def _count_root_elements(self, xml_string: str) -> int:
        """Count the number of root elements in the document"""
        # Remove comments, processing instructions, and whitespace
        cleaned = re.sub(r'<!--.*?-->', '', xml_string, flags=re.DOTALL)
        cleaned = re.sub(r'<\?.*?\?>', '', cleaned)
        cleaned = re.sub(r'^\s+', '', cleaned, flags=re.MULTILINE)

        # Find root elements (tags that are not nested within other tags)
        root_elements = []
        lines = cleaned.split('\n')
        tag_depth = 0

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Count opening and closing tags
            opening_tags = re.findall(r'<([a-zA-Z][^/>]*[^/])>', line)
            closing_tags = re.findall(r'</[a-zA-Z][^>]*>', line)

            # If we're at depth 0 and find an opening tag, it's a root element
            for tag in opening_tags:
                if tag_depth == 0:
                    # Extract just the tag name (before any attributes)
                    tag_name = tag.split()[0] if ' ' in tag else tag
                    root_elements.append(tag_name)
                tag_depth += 1

            tag_depth -= len(closing_tags)

        return len(root_elements)