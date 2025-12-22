"""
XML File Reader Utility.

This module provides safe XML file reading with comprehensive error handling.
"""

from pathlib import Path
from typing import Any, Dict


class XMLFileReader:
    """
    Handle XML file I/O operations.

    This class provides safe file reading with proper error handling for
    various file access issues (not found, permissions, encoding, etc.).
    """

    @staticmethod
    def read_file(file_path: str) -> Dict[str, Any]:
        """
        Read XML file and return content or error information.

        Args:
            file_path: Path to the XML file

        Returns:
            Dictionary with keys:
                - success: Boolean indicating if read was successful
                - content: File content (if successful)
                - error_type: Type of error (if unsuccessful)
                - message: Error message (if unsuccessful)
                - file_path: Original file path
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                xml_content = f.read()

            return {
                'success': True,
                'content': xml_content,
                'file_path': file_path
            }

        except FileNotFoundError:
            return {
                'success': False,
                'error_type': 'file_not_found',
                'message': f"File not found: {file_path}",
                'file_path': file_path
            }

        except PermissionError:
            return {
                'success': False,
                'error_type': 'permission_error',
                'message': f"Permission denied: {file_path}",
                'file_path': file_path
            }

        except UnicodeDecodeError as e:
            return {
                'success': False,
                'error_type': 'encoding_error',
                'message': f"Encoding error: {str(e)}",
                'file_path': file_path
            }

        except Exception as e:
            return {
                'success': False,
                'error_type': 'unknown_error',
                'message': f"Unexpected error reading file: {str(e)}",
                'file_path': file_path
            }

    @staticmethod
    def validate_file_path(file_path: str) -> bool:
        """
        Validate if the file path exists and is readable.

        Args:
            file_path: Path to validate

        Returns:
            True if file exists, is a file, and has non-zero size; False otherwise
        """
        path = Path(file_path)
        return path.exists() and path.is_file() and path.stat().st_size > 0