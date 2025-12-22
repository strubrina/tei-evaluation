import re
from typing import Dict, List
from collections import defaultdict

class AttributeAnalyzer:
    """Analyze XML attribute syntax issues"""

    def __init__(self):
        self.error_categories = defaultdict(list)
        self.attr_pattern = r'(\w+)\s*=\s*(?:(["\'])([^"\']*?)\2|([^"\'\s>]+))'

    def analyze_attributes(self, xml_string: str) -> Dict[str, List[str]]:
        """
        Analyze attribute syntax issues

        Args:
            xml_string: XML content to analyze

        Returns:
            Dictionary of error categories with error messages
        """
        self.error_categories.clear()
        lines = xml_string.split('\n')

        for line_num, line in enumerate(lines, 1):
            self._analyze_line_attributes(line, line_num)

        return dict(self.error_categories)

    def _analyze_line_attributes(self, line: str, line_num: int):
        """Analyze attributes in a single line"""
        # Find all opening tags (excluding comments and processing instructions)
        for tag_match in re.finditer(r'<([^/!?][^>\s]*)((?:\s[^>]*)?)\s*/?>', line):
            tag_name = tag_match.group(1)
            attr_section = tag_match.group(2)

            if not attr_section.strip():
                continue

            self._analyze_tag_attributes(attr_section, tag_name, line_num)

    def _analyze_tag_attributes(self, attr_section: str, tag_name: str, line_num: int):
        """Analyze attributes within a tag"""
        attributes = {}

        for attr_match in re.finditer(self.attr_pattern, attr_section):
            attr_name = attr_match.group(1)
            quote_char = attr_match.group(2)  # " or ' or None
            quoted_value = attr_match.group(3)  # Value within quotes
            unquoted_value = attr_match.group(4)  # Unquoted value

            attr_value = quoted_value if quote_char else unquoted_value

            # Check for unquoted attributes (XML requires quotes)
            if not quote_char:
                self.error_categories['unquoted'].append(
                    f"Line {line_num}: Unquoted attribute '{attr_name}={attr_value}' in <{tag_name}> - XML requires quoted attribute values"
                )

            # Check for duplicate attributes
            if attr_name in attributes:
                self.error_categories['duplicate'].append(
                    f"Line {line_num}: Duplicate attribute '{attr_name}' in <{tag_name}>"
                )

            attributes[attr_name] = attr_value