"""
Security Event Logger Implementation
"""

import logging
from datetime import datetime
from typing import List, Optional
from .interfaces import SecurityEvent, EventType, EventSeverity

logger = logging.getLogger(__name__)


class SecurityEventLogger:
    """Logger for security events in the QSFL-CAAD system."""
    
    def __init__(self):
        """Initialize the security event logger."""
        self.events = []
    
    def log_event(self, event: SecurityEvent) -> None:
        """Log a security event."""
        self.events.append(event)
        logger.info(f"Security event: {event.event_type.value} - {event.description}")
    
    def log_authentication_event(self, client_id: str, success: bool, details: str) -> None:
        """Log an authentication event."""
        severity = EventSeverity.LOW if success else EventSeverity.HIGH
        event = SecurityEvent(
            event_id=f"auth_{len(self.events)}",
            event_type=EventType.AUTHENTICATION,
            severity=severity,
            client_id=client_id,
            description=f"Authentication {'successful' if success else 'failed'}: {details}",
            metadata={'success': success},
            timestamp=datetime.now()
        )
        self.log_event(event)
    
    def log_cryptographic_event(self, operation: str, success: bool, details: str) -> None:
        """Log a cryptographic operation event."""
        severity = EventSeverity.LOW if success else EventSeverity.MEDIUM
        event = SecurityEvent(
            event_id=f"crypto_{len(self.events)}",
            event_type=EventType.CRYPTOGRAPHIC,
            severity=severity,
            client_id=None,
            description=f"Cryptographic operation {operation}: {details}",
            metadata={'operation': operation, 'success': success},
            timestamp=datetime.now()
        )
        self.log_event(event)
    
    def log_anomaly_event(self, client_id: str, anomaly_score: float, action: str) -> None:
        """Log an anomaly detection event."""
        severity = EventSeverity.HIGH if anomaly_score > 0.8 else EventSeverity.MEDIUM
        event = SecurityEvent(
            event_id=f"anomaly_{len(self.events)}",
            event_type=EventType.ANOMALY_DETECTION,
            severity=severity,
            client_id=client_id,
            description=f"Anomaly detected (score: {anomaly_score:.3f}), action: {action}",
            metadata={'anomaly_score': anomaly_score, 'action': action},
            timestamp=datetime.now()
        )
        self.log_event(event)
    
    def get_events(self, start_time: datetime, end_time: datetime, 
                   event_type: Optional[EventType] = None) -> List[SecurityEvent]:
        """Retrieve security events within time range."""
        filtered_events = []
        
        for event in self.events:
            if start_time <= event.timestamp <= end_time:
                if event_type is None or event.event_type == event_type:
                    filtered_events.append(event)
        
        return filtered_events