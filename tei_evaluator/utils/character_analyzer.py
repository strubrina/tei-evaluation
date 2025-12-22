import re
from typing import Dict, List
from collections import defaultdict

class CharacterAnalyzer:
    """Analyze XML character encoding and entity issues"""

    def __init__(self):
        self.error_categories = defaultdict(list)
        self.standard_entities = {'&amp;', '&lt;', '&gt;', '&quot;', '&apos;'}

    def analyze_characters(self, xml_content: str) -> Dict[str, List[str]]:
        """
        Analyze character encoding and escaping issues

        Args:
            xml_content: XML content to analyze

        Returns:
            Dictionary of error categories with error messages
        """
        self.error_categories.clear()
        lines = xml_content.split('\n')

        for line_num, line in enumerate(lines, 1):
            # Skip XML declarations and comments entirely
            stripped_line = line.strip()
            if stripped_line.startswith('<?xml') or stripped_line.startswith('<!--'):
                continue

            self._analyze_line_contexts(line, line_num)

        return dict(self.error_categories)

    def _analyze_line_contexts(self, line: str, line_num: int):
        """Analyze a line by parsing its different contexts (tags, attributes, text content)"""
        i = 0
        while i < len(line):
            if line[i] == '<':
                i = self._handle_tag_context(line, line_num, i)
            elif line[i] == '&':
                i = self._handle_entity_context(line, line_num, i)
            else:
                i = self._handle_text_context(line, line_num, i)

    def _handle_tag_context(self, line: str, line_num: int, start_pos: int) -> int:
        """Handle context within a tag"""
        tag_end = line.find('>', start_pos)
        if tag_end == -1:
            # Malformed tag - missing closing bracket
            self.error_categories['malformed_tags'].append(
                f"Line {line_num}, column {start_pos+1}: Tag missing closing bracket"
            )
            return len(line)

        # Extract and analyze the tag
        tag_content = line[start_pos:tag_end+1]
        self._analyze_tag_content(tag_content, line_num, start_pos)
        return tag_end + 1

    def _handle_entity_context(self, line: str, line_num: int, start_pos: int) -> int:
        """Handle entity reference context"""
        entity_end = line.find(';', start_pos)
        if entity_end == -1:
            # Unescaped ampersand - no semicolon found
            self.error_categories['unescaped_ampersand'].append(
                f"Line {line_num}, column {start_pos+1}: Unescaped '&' - missing semicolon"
            )
            return start_pos + 1
        else:
            # Check if it's a valid entity reference
            entity = line[start_pos:entity_end+1]
            if not self._is_valid_entity(entity):
                self.error_categories['unescaped_ampersand'].append(
                    f"Line {line_num}, column {start_pos+1}: Invalid entity reference '{entity}'"
                )
            return entity_end + 1

    def _handle_text_context(self, line: str, line_num: int, start_pos: int) -> int:
        """Handle text content context"""
        # Check for invalid characters
        char_code = ord(line[start_pos])
        if self._is_invalid_xml_character(char_code):
            self.error_categories['invalid_characters'].append(
                f"Line {line_num}, column {start_pos+1}: Invalid XML character (U+{char_code:04X})"
            )
        return start_pos + 1

    def _analyze_tag_content(self, tag_content: str, line_num: int, start_col: int):
        """Analyze the content within a tag"""
        # Skip comments, processing instructions, and DOCTYPE
        if tag_content.startswith('<!--') or tag_content.startswith('<?') or tag_content.startswith('<!'):
            return

        # Check for unescaped < within the tag content (this would be malformed)
        inner_content = tag_content[1:-1]  # Remove < and >
        if '<' in inner_content:
            self.error_categories['malformed_tags'].append(
                f"Line {line_num}, column {start_col+1}: Unescaped '<' within tag"
            )

    def _is_valid_entity(self, entity: str) -> bool:
        """Check if an entity reference is valid"""
        # Standard XML entities
        if entity in self.standard_entities:
            return True

        # Numeric character references: &#123; or &#x1F;
        if re.match(r'&#\d+;$', entity) or re.match(r'&#x[0-9a-fA-F]+;$', entity):
            return True

        # Named character entities (this is more complex in real XML)
        # For simplicity, we'll accept any entity that follows the pattern &name;
        if re.match(r'&[a-zA-Z][a-zA-Z0-9]*;$', entity):
            return True

        return False

    def _is_invalid_xml_character(self, char_code: int) -> bool:
        """Check if character code is invalid in XML"""
        # Invalid XML character (control characters except tab, newline, carriage return)
        return (char_code in range(0x00, 0x09) or
                char_code in range(0x0B, 0x0C) or
                char_code in range(0x0E, 0x20) or
                char_code == 0x7F)