from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

@dataclass
class Metrics:
    """Container for evaluation metrics"""
    level: int
    data: Dict[str, Any] = field(default_factory=dict)

    def add_metric(self, key: str, value: Any):
        """Add a metric"""
        self.data[key] = value

    def get_metric(self, key: str, default: Any = None):
        """Get a metric value"""
        return self.data.get(key, default)