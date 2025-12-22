# utils/jing_validator.py

import re
import subprocess
from typing import Tuple, List, Optional
from pathlib import Path
from ..models import Error, ErrorType

class JingValidator:
    """Utility class for XML validation using Jing RelaxNG validator"""

    def __init__(self, jing_jar_path: Optional[str] = None):
        """
        Initialize Jing validator

        Args:
            jing_jar_path: Path to Jing jar file. If None, uses bundled version.
        """
        if jing_jar_path is None:
            # Use bundled Jing jar
            package_dir = Path(__file__).parent.parent
            self.jing_jar_path = package_dir / "resources" / "tools" / "jing-RELEASE220.jar"
        else:
            self.jing_jar_path = Path(jing_jar_path)

    def validate_quick(self, xml_file: str, schema_file: str) -> Tuple[bool, str]:
        """
        Quick validation - just returns pass/fail and basic error message

        Args:
            xml_file: Path to XML file to validate
            schema_file: Path to RelaxNG schema file

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            result = subprocess.run(
                ["java", "-jar", str(self.jing_jar_path), str(schema_file), str(xml_file)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                return True, ""
            else:
                # Return just the first error for quick feedback
                error_lines = result.stdout.strip().split('\n')
                first_error = next((line for line in error_lines if 'error:' in line), "Validation failed")
                return False, self._extract_clean_error_message(first_error)

        except subprocess.TimeoutExpired:
            return False, "Validation timed out"
        except FileNotFoundError:
            return False, "Java not found - please ensure Java is installed and in PATH"
        except (OSError, subprocess.SubprocessError, ValueError) as e:
            return False, f"Validation error: {str(e)}"

    def validate_detailed(self, xml_file: str, schema_file: str, schema_type: str) -> Tuple[bool, List[Error]]:
        """
        Detailed validation - returns all errors as Error objects

        Args:
            xml_file: Path to XML file to validate
            schema_file: Path to RelaxNG schema file
            schema_type: Type of schema for error categorization ("TEI" or "Project")

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []        # Check if Jing jar exists
        if not self.jing_jar_path.exists():
            error = Error(
                type=ErrorType.TEI_SCHEMA_VIOLATION,  # Use new error type
                severity=5,
                location="Validator",
                message=f"Jing jar not found at: {self.jing_jar_path}",
                raw_error=""
            )
            return False, [error]

        # Check if schema exists
        schema_path = Path(schema_file)
        if not schema_path.exists():
            error_type = ErrorType.TEI_SCHEMA_VIOLATION if schema_type == "TEI" else ErrorType.PROJECT_SCHEMA_VIOLATION
            error = Error(
                type=error_type,
                severity=5,
                location="Schema",
                message=f"{schema_type} schema not found at: {schema_file}",
                raw_error=""
            )
            return False, [error]

        try:
            result = subprocess.run(
                ["java", "-jar", str(self.jing_jar_path), str(schema_file), str(xml_file)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                return True, []

            # Parse all validation errors
            error_lines = result.stdout.strip().split('\n')
            for line in error_lines:
                if 'error:' in line:
                    error = self._parse_jing_error(line, schema_type)
                    if error:
                        errors.append(error)

            return False, errors

        except subprocess.TimeoutExpired:
            error_type = ErrorType.TEI_SCHEMA_VIOLATION if schema_type == "TEI" else ErrorType.PROJECT_SCHEMA_VIOLATION
            error = Error(
                type=error_type,
                severity=8,
                location="Validation",
                message=f"{schema_type} schema validation timed out",
                raw_error="Timeout"
            )
            return False, [error]

        except FileNotFoundError:
            error_type = ErrorType.TEI_SCHEMA_VIOLATION if schema_type == "TEI" else ErrorType.PROJECT_SCHEMA_VIOLATION
            error = Error(
                type=error_type,
                severity=8,
                location="Validation",
                message="Java not found. Please ensure Java is installed and in PATH.",
                raw_error="Java not found"
            )
            return False, [error]

        except (OSError, subprocess.SubprocessError, ValueError, RuntimeError) as e:
            error_type = ErrorType.TEI_SCHEMA_VIOLATION if schema_type == "TEI" else ErrorType.PROJECT_SCHEMA_VIOLATION
            error = Error(
                type=error_type,
                severity=6,
                location="Validation",
                message=f"Error during {schema_type} schema validation: {str(e)}",
                raw_error=str(e)
            )
            return False, [error]

    def _extract_clean_error_message(self, error_line: str) -> str:
        """Extract clean error message from Jing output"""
        match = re.search(r'error: (.*)', error_line)
        return match.group(1).strip() if match else error_line.strip()

    def _parse_jing_error(self, error_line: str, schema_type: str) -> Optional[Error]:
        """
        Parse Jing error line into Error object with categorization
        """
        # Extract location information
        location_match = re.search(r'(\d+):(\d+):', error_line)
        if location_match:
            line_num = location_match.group(1)
            col_num = location_match.group(2)
            location = f"Line {line_num}, Column {col_num}"
        else:
            location = "Unknown"

        # Extract error message
        error_message = self._extract_clean_error_message(error_line)

        # Determine error type and severity
        if schema_type.upper() == "TEI":
            error_type = ErrorType.TEI_SCHEMA_VIOLATION
            severity = 7
        else:  # Project schema
            error_type = ErrorType.PROJECT_SCHEMA_VIOLATION
            severity = 5

        # Create Error object without jing_category for now
        return Error(
            type=error_type,
            severity=severity,
            location=location,
            message=f"{schema_type} schema violation: {error_message}",
            raw_error=error_line
        )
    def _categorize_jing_error(self, error_message: str, schema_type: str) -> Tuple[ErrorType, int, str]:
        """
        Categorize Jing error messages and assign severity and category

        Returns:
            Tuple of (ErrorType, severity_int, jing_category)
        """
        error_msg_lower = error_message.lower()

        # Determine base error type
        if schema_type.upper() == "TEI":
            base_error_type = ErrorType.TEI_SCHEMA_VIOLATION
            base_severity = 7
        else:  # Project schema
            base_error_type = ErrorType.PROJECT_SCHEMA_VIOLATION
            base_severity = 5

        # Categorize Jing error types
        jing_category = "other"
        severity_modifier = 0

        if any(phrase in error_msg_lower for phrase in [
            "element", "not allowed", "unexpected element", "unknown element"
        ]):
            jing_category = "element_not_allowed"
            severity_modifier = 1
        elif any(phrase in error_msg_lower for phrase in [
            "missing", "required", "expecting", "incomplete content"
        ]):
            jing_category = "missing_required_element"
            severity_modifier = 2
        elif any(phrase in error_msg_lower for phrase in [
            "attribute", "invalid attribute", "unknown attribute", "bad attribute"
        ]):
            jing_category = "invalid_attribute"
            severity_modifier = 0
        elif any(phrase in error_msg_lower for phrase in [
            "content model", "content", "mixed content", "empty content"
        ]):
            jing_category = "content_model_violation"
            severity_modifier = 1
        elif any(phrase in error_msg_lower for phrase in [
            "data", "value", "datatype", "type", "format", "pattern"
        ]):
            jing_category = "datatype_constraint_violation"
            severity_modifier = -1
        elif any(phrase in error_msg_lower for phrase in [
            "text", "character data", "cdata"
        ]):
            jing_category = "invalid_text_content"
            severity_modifier = -1

        # Calculate final severity
        severity = base_severity + severity_modifier
        severity = max(1, min(10, severity))

        return base_error_type, severity, jing_category

    def is_available(self) -> bool:
        """Check if Jing validator is available"""
        return self.jing_jar_path.exists()

    def get_validator_info(self) -> dict:
        """Get information about the validator setup"""
        return {
            "jing_jar_path": str(self.jing_jar_path),
            "jing_available": self.is_available(),
            "java_available": self._check_java_available()
        }

    def _check_java_available(self) -> bool:
        """Check if Java is available in PATH"""
        try:
            result = subprocess.run(
                ["java", "-version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False