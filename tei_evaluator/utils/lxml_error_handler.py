"""
LXML Error Handler Utility.

This module handles and categorizes lxml XMLSyntaxError messages.
"""

from collections import defaultdict
from typing import Dict, List

from lxml import etree

from .location_extractor import LocationExtractor


class LXMLErrorHandler:
    """
    Handle and categorize lxml XMLSyntaxError messages.

    This class parses lxml error messages and categorizes them into
    specific error types for better reporting and analysis.
    """

    def __init__(self):
        self.error_categories = defaultdict(list)
        self.location_extractor = LocationExtractor()

    def categorize_lxml_error(self, error: etree.XMLSyntaxError) -> Dict[str, List[str]]:
        """
        Categorize lxml XMLSyntaxError messages.

        Args:
            error: lxml XMLSyntaxError object

        Returns:
            Dictionary mapping error categories to lists of error messages
        """
        self.error_categories.clear()
        error_msg = str(error).lower()
        location = self.location_extractor.extract_from_lxml_error(error)

        # Categorize based on error message content
        if 'mismatched tag' in error_msg or ('expected' in error_msg and 'found' in error_msg):
            self.error_categories['mismatched_tags'].append(f"{location}: {error}")
        elif 'unclosed token' in error_msg or 'unclosed' in error_msg:
            self.error_categories['unclosed_tags'].append(f"{location}: {error}")
        elif 'not well-formed' in error_msg or 'well-formed' in error_msg:
            self.error_categories['malformed'].append(f"{location}: {error}")
        elif 'invalid character' in error_msg or 'illegal character' in error_msg:
            self.error_categories['invalid_characters'].append(f"{location}: {error}")
        elif 'unescaped' in error_msg:
            self._handle_unescaped_error(error_msg, location, error)
        elif 'attribute' in error_msg:
            self._handle_attribute_error(error_msg, location, error)
        elif 'multiple' in error_msg and 'root' in error_msg:
            self.error_categories['multiple_roots'].append(f"{location}: {error}")
        elif 'tag name' in error_msg or 'invalid tag' in error_msg:
            self.error_categories['invalid_tag_names'].append(f"{location}: {error}")
        else:
            # Catch-all for other syntax errors
            self.error_categories['lxml_syntax_error'].append(f"{location}: {error}")

        return dict(self.error_categories)

    def _handle_unescaped_error(self, error_msg: str, location: str, error: etree.XMLSyntaxError):
        """
        Handle unescaped character errors.

        Args:
            error_msg: Error message string
            location: Location string
            error: Original XMLSyntaxError object
        """
        if '&' in error_msg:
            self.error_categories['unescaped_ampersand'].append(f"{location}: {error}")
        elif '<' in error_msg:
            self.error_categories['unescaped_less_than'].append(f"{location}: {error}")
        else:
            self.error_categories['unescaped_ampersand'].append(f"{location}: {error}")

    def _handle_attribute_error(self, error_msg: str, location: str, error: etree.XMLSyntaxError):
        """
        Handle attribute-related errors.

        Args:
            error_msg: Error message string
            location: Location string
            error: Original XMLSyntaxError object
        """
        if 'duplicate' in error_msg:
            self.error_categories['duplicate'].append(f"{location}: {error}")
        elif 'unquoted' in error_msg or 'quote' in error_msg:
            self.error_categories['unquoted'].append(f"{location}: {error}")
        else:
            self.error_categories['malformed'].append(f"{location}: {error}")