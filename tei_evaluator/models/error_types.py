# error_types.py

from enum import Enum
from dataclasses import dataclass
from typing import Optional

class ErrorType(Enum):
    """Classification of TEI encoding errors"""
    # Dimension 1: Syntactic errors
    XML_MALFORMED = "xml_malformed"
    CHARACTER_ENCODING = "character_encoding"
    TAG_STRUCTURE = "tag_structure"
    ATTRIBUTE_SYNTAX = "attribute_syntax"
    CONTENT_PRESERVATION = "content_preservation"
    FILE_NOT_FOUND = "file_not_found"

    # Dimension 2: Schema compliance errors
    TEI_SCHEMA_VIOLATION = "tei_schema_violation"          # TEI schema violations from Jing
    PROJECT_SCHEMA_VIOLATION = "project_schema_violation"  # Project schema violations from Jing

    # Dimension 2b: Structural errors
    TEI_STRUCTURE = "tei_structure"

    # Dimension 3: Semantic errors
    ENTITY_RECOGNITION = "entity_recognition"
    TEMPORAL_PROCESSING = "temporal_processing"
    EDITORIAL_CONVERSION = "editorial_conversion"
    LINGUISTIC_PROCESSING = "linguistic_processing"
    METADATA_EXTRACTION = "metadata_extraction"

    # Dimension 4: Interpretive errors
    INTERTEXTUAL_REFERENCE = "intertextual_reference"
    AUTHORITY_LINKING = "authority_linking"
    SCHOLARLY_INTERPRETATION = "scholarly_interpretation"

    # Dimension 5: Quality management errors
    CONSISTENCY_VIOLATION = "consistency_violation"
    INTEROPERABILITY_ISSUE = "interoperability_issue"
    QUALITY_STANDARD = "quality_standard"

@dataclass
class Error:
    """Individual error instance"""
    type: ErrorType
    severity: int  # 1-10 scale
    location: str
    message: str
    raw_error: str = ""
    suggested_fix: Optional[str] = None

    def __str__(self):
        return f"{self.type.value}: {self.message} (Severity: {self.severity})"