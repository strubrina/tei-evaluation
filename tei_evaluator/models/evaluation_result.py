# evaluation_result.py

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from .error_types import Error

@dataclass
class EvaluationResult:
    """Results from evaluating a TEI file at a specific dimension"""
    dimension: int
    passed: bool
    score: float
    errors: List[Error] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    file_path: Optional[str] = None
    timestamp: Optional[str] = None

    def add_error(self, error: Error):
        """Add an error to the result"""
        self.errors.append(error)

    def get_errors_by_type(self, error_type):
        """Get all errors of a specific type"""
        return [error for error in self.errors if error.type == error_type]

    def get_error_count_by_severity(self, min_severity: int = 1):
        """Count errors above a certain severity threshold"""
        return len([error for error in self.errors if error.severity >= min_severity])