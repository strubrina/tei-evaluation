"""
XML Error Configuration.

This module provides centralized configuration for XML error handling
and categorization.
"""

from ..models.error_types import ErrorType


class XMLErrorConfig:
    """
    Centralized configuration for XML error handling and categorization.

    This class defines mappings between error categories, types, and severity
    levels used throughout the XML analysis process.
    """

    # Mapping from analysis categories to ErrorType
    ERROR_TYPE_MAPPING = {
        'tag_structure': ErrorType.TAG_STRUCTURE,
        'character_encoding': ErrorType.CHARACTER_ENCODING,
        'attributes': ErrorType.ATTRIBUTE_SYNTAX,
        'document_structure': ErrorType.XML_MALFORMED,
        'syntax': ErrorType.XML_MALFORMED
    }

    # Severity mapping based on subcategory
    SEVERITY_MAPPING = {
        'unclosed_tags': 9,
        'mismatched_tags': 9,
        'unexpected_closing': 8,
        'malformed_tags': 9,
        'invalid_tag_names': 8,
        'multiple_roots': 10,
        'unescaped_ampersand': 6,
        'unescaped_less_than': 7,
        'invalid_characters': 8,
        'unquoted': 8,
        'duplicate': 6,
        'malformed': 8,
        'other': 7,
        'lxml_syntax_error': 9
    }

    # Default severity for unknown error types
    DEFAULT_SEVERITY = 5

    # Standard XML entities
    STANDARD_ENTITIES = {'&amp;', '&lt;', '&gt;', '&quot;', '&apos;'}

    # Error category initialization template
    ERROR_CATEGORIES_TEMPLATE = {
        'tag_structure': {},
        'character_encoding': {},
        'attributes': {},
        'document_structure': {},
        'syntax': {}
    }