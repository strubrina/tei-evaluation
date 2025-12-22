"""
Enhanced XML Analysis Module.

This module provides comprehensive XML well-formedness analysis with
detailed error detection and categorization.
"""

from typing import Any, Dict, List

from lxml import etree

from ..models import Error
from .attribute_analyzer import AttributeAnalyzer
from .character_analyzer import CharacterAnalyzer
from .document_structure_analyzer import DocumentStructureAnalyzer
from .lxml_error_handler import LXMLErrorHandler
from .tag_structure_analyzer import TagStructureAnalyzer
from .xml_error_config import XMLErrorConfig
from .xml_error_formatter import XMLErrorFormatter
from .xml_file_reader import XMLFileReader

class EnhancedXMLAnalyzer:
    """
    Enhanced XML well-formedness analyzer with comprehensive error detection and categorization.

    Architecture:
    This analyzer uses a two-stage approach combining industry-standard validation
    with specialized error detection:

    1. Primary Validation (lxml):
       - Fast, battle-tested XML parser
       - Strict mode validation (no error recovery)
       - Returns pass/fail immediately for well-formed documents

    2. Supplementary Analysis (Custom Analyzers):
       - Only runs if lxml detects errors
       - Provides comprehensive error scanning (not just first error)
       - Categorizes errors into specific types for detailed reporting

    Components:
    - XMLErrorConfig: Error categories, severity levels, and type mappings
    - XMLFileReader: Safe file reading with encoding detection
    - XMLErrorFormatter: Converts analysis results to Error objects
    - LXMLErrorHandler: Parses and categorizes lxml error messages
    - TagStructureAnalyzer: Detects tag nesting, closure, and syntax issues
    - CharacterAnalyzer: Finds unescaped characters and encoding problems
    - AttributeAnalyzer: Validates attribute syntax and quoting
    - DocumentStructureAnalyzer: Checks document-level requirements (root element, etc.)

    Error Categories:
    - tag_structure: Tag nesting, closure, and syntax errors
    - character_encoding: Unescaped characters, invalid characters
    - attributes: Attribute syntax, quoting, and duplication
    - document_structure: Document-level issues (multiple roots, etc.)
    - syntax: General syntax errors from lxml

    Usage:
        analyzer = EnhancedXMLAnalyzer()
        result = analyzer.analyze_file('path/to/file.xml')
        errors = analyzer.convert_to_error_objects(result)
    """

    def __init__(self):
        self.config = XMLErrorConfig()
        self.file_reader = XMLFileReader()
        self.error_formatter = XMLErrorFormatter()
        self.tag_analyzer = TagStructureAnalyzer()
        self.character_analyzer = CharacterAnalyzer()
        self.attribute_analyzer = AttributeAnalyzer()
        self.document_analyzer = DocumentStructureAnalyzer()
        self.lxml_handler = LXMLErrorHandler()

        # Initialize error categories with fresh empty dicts
        self.error_categories = {
            'tag_structure': {},
            'character_encoding': {},
            'attributes': {},
            'document_structure': {},
            'syntax': {}
        }

    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze XML file and return detailed error information.

        Args:
            file_path: Path to XML file to analyze

        Returns:
            Dictionary containing analysis results with error categorization
        """
        file_result = self.file_reader.read_file(file_path)

        if not file_result['success']:
            return {
                'well_formed': False,
                'file_readable': False,
                'read_error': file_result['message'],
                'errors': {},
                'summary': {'total_errors': 1, 'file_read_error': 1}
            }

        return self.analyze_content(file_result['content'])

    def analyze_content(self, xml_content: str) -> Dict[str, Any]:
        """
        Analyze XML content string using a two-stage approach.

        Stage 1: lxml parser (fast, industry-standard validation)
        Stage 2: Supplementary analyzers (comprehensive error detection)

        Rationale for two-stage approach:
        - lxml stops at the FIRST error it encounters (fail-fast behavior)
        - Supplementary analyzers scan the ENTIRE document
        - This provides comprehensive error reporting for evaluation purposes
        - Users get a complete list of ALL errors, not just the first one

        Performance consideration:
        - Well-formed files: Only lxml runs (fast, typically <1ms)
        - Malformed files: Full analysis runs (slower, but necessary for detailed reporting)

        Args:
            xml_content: XML document as string

        Returns:
            Dictionary containing:
                - well_formed (bool): Whether the XML is well-formed
                - file_readable (bool): Whether the file could be read
                - parser_error (str): lxml error message if any
                - errors (dict): Categorized errors from all analyzers
                - summary (dict): Error counts by category
        """
        # Reset error categories - create fresh empty dicts for each category
        # (deep copy to avoid sharing nested dicts between file analyses)
        self.error_categories = {
            'tag_structure': {},
            'character_encoding': {},
            'attributes': {},
            'document_structure': {},
            'syntax': {}
        }

        # Try parsing first to get basic well-formedness
        parsing_successful = True
        parser_error = None

        try:
            # Use lxml's strict parser (no error recovery, preserves CDATA)
            parser = etree.XMLParser(recover=False, strip_cdata=False)
            etree.fromstring(xml_content.encode('utf-8'), parser)
        except etree.XMLSyntaxError as e:
            parsing_successful = False
            parser_error = e

            # Stage 1: Categorize lxml's error message
            self._analyze_lxml_error(e)

            # Stage 2: Comprehensive error detection
            # Unlike lxml (which stops at first error), these analyzers scan
            # the entire document to find ALL errors for complete reporting
            self._analyze_tag_structure(xml_content)
            self._analyze_characters(xml_content)
            self._analyze_attributes(xml_content)
            self._analyze_document_structure(xml_content)
        except Exception as e:
                # Handle other potential errors (encoding, etc.)
                parsing_successful = False
                parser_error = e
                self.error_categories['syntax']['other'] = [f"Parsing error: {str(e)}"]

        return {
            'well_formed': parsing_successful,
            'file_readable': True,
            'parser_error': str(parser_error) if parser_error else None,
            'errors': dict(self.error_categories),
            'summary': self._generate_summary()
        }

    def convert_to_error_objects(self, analysis_result: Dict[str, Any]) -> List[Error]:
        """
        Convert analysis results to Error objects for evaluator.

        Args:
            analysis_result: Dictionary containing analysis results

        Returns:
            List of Error objects
        """
        return self.error_formatter.convert_analysis_to_errors(analysis_result)



    def _analyze_lxml_error(self, error: etree.XMLSyntaxError):
        """
        Categorize lxml XMLSyntaxError messages.

        Args:
            error: lxml XMLSyntaxError object
        """
        categorized_errors = self.lxml_handler.categorize_lxml_error(error)

        # Merge categorized errors into our error categories
        for category, errors in categorized_errors.items():
            if category in self.error_categories:
                if isinstance(self.error_categories[category], dict):
                    for subcategory, error_list in errors.items():
                        if subcategory not in self.error_categories[category]:
                            self.error_categories[category][subcategory] = []
                        self.error_categories[category][subcategory].extend(error_list)
                else:
                    # Handle integer counts (like multiple_roots)
                    self.error_categories[category] = len(errors)

    def _analyze_tag_structure(self, xml_string: str):
        """
        Analyze tag nesting and closure issues - supplementary to lxml parsing.

        Args:
            xml_string: XML content as string
        """
        tag_errors = self.tag_analyzer.analyze_tag_structure(xml_string)
        self.error_categories['tag_structure'].update(tag_errors)

    def _analyze_characters(self, xml_content: str):
        """
        Analyze character encoding and escaping issues - supplementary to lxml.

        Args:
            xml_content: XML content as string
        """
        char_errors = self.character_analyzer.analyze_characters(xml_content)
        self.error_categories['character_encoding'].update(char_errors)

    def _analyze_attributes(self, xml_string: str):
        """
        Analyze attribute syntax issues - supplementary to lxml.

        Args:
            xml_string: XML content as string
        """
        attr_errors = self.attribute_analyzer.analyze_attributes(xml_string)
        self.error_categories['attributes'].update(attr_errors)

    def _analyze_document_structure(self, xml_string: str):
        """
        Analyze overall document structure - supplementary to lxml.

        Args:
            xml_string: XML content as string
        """
        doc_errors = self.document_analyzer.analyze_document_structure(xml_string)
        self.error_categories['document_structure'].update(doc_errors)

    def _generate_summary(self) -> Dict[str, int]:
        """
        Generate a summary of error counts.

        Returns:
            Dictionary mapping error categories to counts
        """
        summary = {}
        total_errors = 0

        for category, subcategories in self.error_categories.items():
            category_count = 0
            for subcat, errors in subcategories.items():
                if isinstance(errors, list):
                    count = len(errors)
                else:
                    count = errors if errors else 0
                category_count += count

            summary[category] = category_count
            total_errors += category_count

        summary['total_errors'] = total_errors
        return summary