"""
Alert Manager Implementation
"""

from datetime import datetime
from typing import Dict, List, Any
from .interfaces import EventSeverity


class AlertManager:
    """Manager for system alerts and notifications."""
    
    def __init__(self):
        """Initialize the alert manager."""
        self.alerts = {}
        self.thresholds = {}
        self.alert_counter = 0
    
    def create_alert(self, title: str, description: str, severity: EventSeverity) -> str:
        """Create a new alert."""
        self.alert_counter += 1
        alert_id = f"alert_{self.alert_counter}"
        
        self.alerts[alert_id] = {
            'id': alert_id,
            'title': title,
            'description': description,
            'severity': severity.value,
            'created_at': datetime.now().isoformat(),
            'resolved': False
        }
        
        return alert_id
    
    def resolve_alert(self, alert_id: str) -> None:
        """Mark an alert as resolved."""
        if alert_id in self.alerts:
            self.alerts[alert_id]['resolved'] = True
            self.alerts[alert_id]['resolved_at'] = datetime.now().isoformat()
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get list of active alerts."""
        return [alert for alert in self.alerts.values() if not alert['resolved']]
    
    def configure_threshold(self, metric_name: str, threshold: float, severity: EventSeverity) -> None:
        """Configure alert threshold for a metric."""
        self.thresholds[metric_name] = {
            'threshold': threshold,
            'severity': severity
        }