import re
from typing import List, Tuple, Dict, Any
from collections import defaultdict

class TagStructureAnalyzer:
    """Analyze XML tag structure and nesting issues"""

    def __init__(self):
        self.error_categories = defaultdict(list)
        self.tag_pattern = r'<(/?)([a-zA-Z][a-zA-Z0-9]*(?::[a-zA-Z][a-zA-Z0-9]*)?)[^>]*?(/?)>'

    def analyze_tag_structure(self, xml_string: str) -> Dict[str, List[str]]:
        """
        Analyze tag nesting and closure issues

        Args:
            xml_string: XML content to analyze

        Returns:
            Dictionary of error categories with error messages
        """
        self.error_categories.clear()
        tag_stack = []
        lines = xml_string.split('\n')

        for line_num, line in enumerate(lines, 1):
            self._analyze_line_tags(line, line_num, tag_stack)

        # Report any remaining unclosed tags
        self._report_unclosed_tags(tag_stack)

        return dict(self.error_categories)

    def _analyze_line_tags(self, line: str, line_num: int, tag_stack: List[Tuple[str, int, int]]):
        """Analyze tags in a single line"""
        # Skip XML declarations and comments
        if line.strip().startswith('<?xml') or line.strip().startswith('<!--'):
            return

        for match in re.finditer(self.tag_pattern, line):
            self._process_tag_match(match, line_num, tag_stack)

    def _process_tag_match(self, match, line_num: int, tag_stack: List[Tuple[str, int, int]]):
        """Process a single tag match"""
        is_closing = match.group(1) == '/'
        tag_name = match.group(2)
        is_self_closing = match.group(3) == '/'
        col_num = match.start() + 1

        # Skip processing instructions and DOCTYPE declarations
        if tag_name.startswith('?') or tag_name.startswith('!'):
            return

        if is_self_closing:
            return
        elif not is_closing:
            tag_stack.append((tag_name, line_num, col_num))
        else:
            self._handle_closing_tag(tag_name, line_num, col_num, tag_stack)

    def _handle_closing_tag(self, tag_name: str, line_num: int, col_num: int, tag_stack: List[Tuple[str, int, int]]):
        """Handle closing tag logic"""
        if not tag_stack:
            self._add_unexpected_closing_error(tag_name, line_num, col_num)
        elif tag_stack[-1][0] != tag_name:
            self._handle_mismatched_tag(tag_name, line_num, col_num, tag_stack)
        else:
            tag_stack.pop()

    def _add_unexpected_closing_error(self, tag_name: str, line_num: int, col_num: int):
        """Add error for unexpected closing tag"""
        self.error_categories['unexpected_closing'].append(
            f"Line {line_num}, column {col_num}: Unexpected closing tag '</{tag_name}>' - no matching opening tag"
        )

    def _handle_mismatched_tag(self, tag_name: str, line_num: int, col_num: int, tag_stack: List[Tuple[str, int, int]]):
        """Handle mismatched tag scenario"""
        expected_tag, expected_line, expected_col = tag_stack[-1]

        self.error_categories['mismatched_tags'].append(
            f"Line {line_num}, column {col_num}: Expected '</{expected_tag}>' (opened at line {expected_line}), found '</{tag_name}>'"
        )

        # Try to find matching tag in stack
        found_match = self._find_matching_tag_in_stack(tag_name, tag_stack)

        if not found_match:
            self._add_unexpected_closing_error(tag_name, line_num, col_num)

    def _find_matching_tag_in_stack(self, tag_name: str, tag_stack: List[Tuple[str, int, int]]) -> bool:
        """Find matching tag in stack and handle unclosed tags"""
        for i in range(len(tag_stack) - 1, -1, -1):
            if tag_stack[i][0] == tag_name:
                # Report unclosed tags between the match and the end
                for j in range(i + 1, len(tag_stack)):
                    unclosed_tag, unclosed_line, unclosed_col = tag_stack[j]
                    self.error_categories['unclosed_tags'].append(
                        f"Line {unclosed_line}, column {unclosed_col}: Unclosed tag '<{unclosed_tag}>' (closed out of order)"
                    )

                # Remove tags from stack up to the match
                tag_stack[:] = tag_stack[:i]
                return True
        return False

    def _report_unclosed_tags(self, tag_stack: List[Tuple[str, int, int]]):
        """Report any remaining unclosed tags"""
        for tag_name, line_num, col_num in tag_stack:
            self.error_categories['unclosed_tags'].append(
                f"Line {line_num}, column {col_num}: Unclosed tag '<{tag_name}>'"
            )