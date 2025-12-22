from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from ..models import EvaluationResult

class BaseEvaluator(ABC):
    """Abstract base class for all TEI evaluation dimensions"""

    def __init__(self):
        self.dimension = None  # To be set by subclasses

    @abstractmethod
    def evaluate_file(self, file_path: str, config: Optional[Dict[str, Any]] = None) -> EvaluationResult:
        """Evaluate a single TEI file"""
        pass

    @abstractmethod
    def calculate_score(self, errors: list, base_score: float = 100.0) -> float:
        """Calculate score with graduated penalties"""
        pass

    def validate_input(self, file_path: str) -> bool:
        """Common input validation"""
        from pathlib import Path
        return Path(file_path).exists() and Path(file_path).suffix.lower() in ['.xml', '.tei']