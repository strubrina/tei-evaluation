"""
XML Error Formatter Utility.

This module converts XML analysis results into Error objects.
"""

from typing import Any, Dict, List

from ..models.error_types import Error, ErrorType
from .location_extractor import LocationExtractor
from .xml_error_config import XMLErrorConfig


class XMLErrorFormatter:
    """
    Convert XML analysis results into Error objects.

    This class transforms raw analysis results into structured Error objects
    with appropriate types, severities, and locations.
    """

    def __init__(self):
        self.config = XMLErrorConfig()
        self.location_extractor = LocationExtractor()

    def convert_analysis_to_errors(self, analysis_result: Dict[str, Any]) -> List[Error]:
        """
        Convert analysis results to Error objects.

        Args:
            analysis_result: Dictionary containing analysis results from XML analyzer

        Returns:
            List of Error objects with categorized errors
        """
        errors = []

        # Handle file reading errors
        if not analysis_result.get('file_readable', True):
            errors.append(self._create_file_error(analysis_result))
            return errors

        # Convert each error category
        for category, subcategories in analysis_result.get('errors', {}).items():
            for subcat, error_list in subcategories.items():
                if isinstance(error_list, list):
                    for error_msg in error_list:
                        errors.append(self._create_error_object(category, subcat, error_msg))
                elif error_list > 0:  # Handle integer counts
                    errors.append(self._create_error_object(category, subcat, f"{subcat}: {error_list}"))

        return errors

    def _create_file_error(self, analysis_result: Dict[str, Any]) -> Error:
        """
        Create error object for file reading issues.

        Args:
            analysis_result: Analysis result dictionary with error information

        Returns:
            Error object for file reading failure
        """
        return Error(
            type=ErrorType.CHARACTER_ENCODING,
            severity=10,
            location="File",
            message=f"Could not read file: {analysis_result.get('read_error', 'Unknown error')}",
            raw_error=analysis_result.get('read_error', '')
        )

    def _create_error_object(self, category: str, subcategory: str, message: str) -> Error:
        """
        Create Error object from analysis results.

        Args:
            category: Error category (e.g., 'tag_structure', 'character_encoding')
            subcategory: Error subcategory (e.g., 'unclosed_tags', 'unescaped_ampersand')
            message: Error message

        Returns:
            Error object with appropriate type, severity, and location
        """
        error_type = self.config.ERROR_TYPE_MAPPING.get(category, ErrorType.XML_MALFORMED)
        severity = self.config.SEVERITY_MAPPING.get(subcategory, self.config.DEFAULT_SEVERITY)
        location = self.location_extractor.extract_location(message)

        return Error(
            type=error_type,
            severity=severity,
            location=location,
            message=message,
            raw_error=message
        )

    def create_lxml_error(self, error, category: str, subcategory: str) -> Error:
        """
        Create Error object from lxml XMLSyntaxError.

        Args:
            error: lxml XMLSyntaxError object
            category: Error category
            subcategory: Error subcategory

        Returns:
            Error object with extracted location and appropriate severity
        """
        location = self.location_extractor.extract_from_lxml_error(error)
        error_type = self.config.ERROR_TYPE_MAPPING.get(category, ErrorType.XML_MALFORMED)
        severity = self.config.SEVERITY_MAPPING.get(subcategory, self.config.DEFAULT_SEVERITY)

        return Error(
            type=error_type,
            severity=severity,
            location=location,
            message=str(error),
            raw_error=str(error)
        )