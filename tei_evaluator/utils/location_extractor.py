import re
from typing import Optional, Tuple

from lxml import etree


class LocationExtractor:
    """Utility class for extracting and formatting location information from XML errors"""

    # Regex patterns for extracting location information
    LINE_COL_PATTERN = r'Line (\d+)(?:, col(?:umn)? (\d+))?'
    LINE_ONLY_PATTERN = r'Line (\d+)'
    COL_ONLY_PATTERN = r'col(?:umn)? (\d+)'

    @staticmethod
    def extract_location(message: str) -> str:
        """
        Extract location information from error message

        Args:
            message: Error message that may contain location information

        Returns:
            Formatted location string or "Unknown" if not found
        """
        location_match = re.search(LocationExtractor.LINE_COL_PATTERN, message)
        if location_match:
            return location_match.group(0)
        return "Unknown"

    @staticmethod
    def extract_line_column(message: str) -> Tuple[Optional[int], Optional[int]]:
        """
        Extract line and column numbers from error message

        Args:
            message: Error message containing location information

        Returns:
            Tuple of (line_number, column_number) or (None, None) if not found
        """
        match = re.search(LocationExtractor.LINE_COL_PATTERN, message)
        if match:
            line_num = int(match.group(1)) if match.group(1) else None
            col_num = int(match.group(2)) if match.group(2) else None
            return line_num, col_num
        return None, None

    @staticmethod
    def format_location(line_num: int, col_num: Optional[int] = None) -> str:
        """
        Format location information into a standard string

        Args:
            line_num: Line number
            col_num: Column number (optional)

        Returns:
            Formatted location string
        """
        if col_num is not None:
            return f"Line {line_num}, column {col_num}"
        return f"Line {line_num}"

    @staticmethod
    def extract_from_lxml_error(error: etree.XMLSyntaxError) -> str:
        """
        Extract location from lxml XMLSyntaxError

        Args:
            error: lxml XMLSyntaxError object

        Returns:
            Formatted location string
        """
        line_no = getattr(error, 'lineno', None)
        col_no = getattr(error, 'offset', None)

        if line_no is not None:
            return LocationExtractor.format_location(line_no, col_no)
        return "Unknown"