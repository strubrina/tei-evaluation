"""
Unified whitespace normalization utilities for TEI evaluation.

This module provides consistent whitespace normalization across all evaluation dimensions.
"""

import re
from typing import Literal


def normalize_whitespace(text: str, mode: Literal["text", "xml"] = "text") -> str:
    """
    Normalize whitespace in text or XML strings.

    This function provides consistent whitespace normalization across all evaluation
    dimensions. It preserves word boundaries while ignoring formatting variations
    (multiple spaces, tabs, newlines, etc.).

    Args:
        text: The text or XML string to normalize
        mode: Normalization mode
            - "text": For plain text content (normalizes all whitespace sequences to single space)
            - "xml": For XML strings (also removes whitespace between tags)

    Returns:
        Normalized string with whitespace sequences collapsed to single spaces

    Examples:
        >>> normalize_whitespace("Hello    World\\n\\tTest")
        'Hello World Test'
        >>> normalize_whitespace("<tag>  </tag>  <other>text</other>", mode="xml")
        '<tag></tag> <other>text</other>'
    """
    if not text:
        return ""

    if mode == "xml":
        # For XML strings: first remove whitespace between tags
        text = re.sub(r'>\s+<', '><', text)

    # Normalize all whitespace sequences to single space
    normalized = re.sub(r'\s+', ' ', text)

    # Remove leading/trailing whitespace
    normalized = normalized.strip()

    return normalized


def normalize_whitespace_completely(text: str) -> str:
    """
    Remove ALL whitespace characters completely.

    This is a legacy function for Dimension 1's whitespace-agnostic comparison.
    It removes all whitespace to create a version with zero whitespace for exact
    content matching (ignoring all formatting).

    Note: This function loses word boundaries. For most use cases, prefer
    normalize_whitespace() which preserves word boundaries.

    Args:
        text: The text to process

    Returns:
        String with all whitespace removed

    Examples:
        >>> normalize_whitespace_completely("Hello    World\\nTest")
        'HelloWorldTest'
    """
    if not text:
        return ""

    # Remove ALL whitespace characters: spaces, tabs, newlines, carriage returns, etc.
    # \s matches [ \t\n\r\f\v] and any Unicode whitespace
    normalized = re.sub(r'\s', '', text)

    return normalized

