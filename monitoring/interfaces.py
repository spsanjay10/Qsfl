"""
Monitoring and Logging Interfaces

Defines abstract base classes and interfaces for monitoring and logging operations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional


class EventSeverity(Enum):
    """Severity levels for security events."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EventType(Enum):
    """Types of security events."""
    AUTHENTICATION = "authentication"
    CRYPTOGRAPHIC = "cryptographic"
    ANOMALY_DETECTION = "anomaly_detection"
    AGGREGATION = "aggregation"
    SYSTEM = "system"


@dataclass
class SecurityEvent:
    """Security event data structure."""
    event_id: str
    event_type: EventType
    severity: EventSeverity
    client_id: Optional[str]
    description: str
    metadata: Dict[str, Any]
    timestamp: datetime


@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    active_clients: int
    training_rounds_completed: int
    anomalies_detected: int
    authentication_failures: int


class ISecurityEventLogger(ABC):
    """Interface for security event logging."""
    
    @abstractmethod
    def log_event(self, event: SecurityEvent) -> None:
        """Log a security event.
        
        Args:
            event: Security event to log
        """
        pass
    
    @abstractmethod
    def log_authentication_event(self, client_id: str, success: bool, details: str) -> None:
        """Log an authentication event."""
        pass
    
    @abstractmethod
    def log_cryptographic_event(self, operation: str, success: bool, details: str) -> None:
        """Log a cryptographic operation event."""
        pass
    
    @abstractmethod
    def log_anomaly_event(self, client_id: str, anomaly_score: float, action: str) -> None:
        """Log an anomaly detection event."""
        pass
    
    @abstractmethod
    def get_events(self, start_time: datetime, end_time: datetime, 
                   event_type: Optional[EventType] = None) -> List[SecurityEvent]:
        """Retrieve security events within time range."""
        pass


class IMetricsCollector(ABC):
    """Interface for collecting system metrics."""
    
    @abstractmethod
    def collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics.
        
        Returns:
            Current system metrics
        """
        pass
    
    @abstractmethod
    def record_metric(self, name: str, value: float, timestamp: Optional[datetime] = None) -> None:
        """Record a custom metric."""
        pass
    
    @abstractmethod
    def get_metric_history(self, name: str, start_time: datetime, end_time: datetime) -> List[tuple]:
        """Get historical values for a metric."""
        pass


class IAlertManager(ABC):
    """Interface for managing alerts and notifications."""
    
    @abstractmethod
    def create_alert(self, title: str, description: str, severity: EventSeverity) -> str:
        """Create a new alert.
        
        Returns:
            Alert ID
        """
        pass
    
    @abstractmethod
    def resolve_alert(self, alert_id: str) -> None:
        """Mark an alert as resolved."""
        pass
    
    @abstractmethod
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get list of active alerts."""
        pass
    
    @abstractmethod
    def configure_threshold(self, metric_name: str, threshold: float, severity: EventSeverity) -> None:
        """Configure alert threshold for a metric."""
        pass


class IDashboardService(ABC):
    """Interface for dashboard and visualization services."""
    
    @abstractmethod
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status for dashboard."""
        pass
    
    @abstractmethod
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security events summary."""
        pass
    
    @abstractmethod
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary."""
        pass