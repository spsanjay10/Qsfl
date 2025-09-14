"""
Integration tests for AlertManager with other monitoring components.

Tests alert generation and response workflows in the context of the
complete monitoring system.
"""

import tempfile
import time
import unittest
from datetime import datetime

from monitoring.alert_manager import AlertManager
from monitoring.security_logger import SecurityEventLogger
from monitoring.metrics_collector import MetricsCollector
from monitoring.interfaces import EventSeverity, EventType, SecurityEvent


class TestAlertIntegration(unittest.TestCase):
    """Integration tests for AlertManager with monitoring system."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Initialize components
        self.alert_manager = AlertManager(db_path=f"{self.temp_dir}/alerts.db")
        self.security_logger = SecurityEventLogger(db_path=f"{self.temp_dir}/security.db")
        self.metrics_collector = MetricsCollector()
    
    def tearDown(self):
        """Clean up test environment."""
        self.alert_manager.shutdown()
        
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_security_event_triggers_alert(self):
        """Test that security events can trigger alerts."""
        # Configure alert threshold for authentication failures
        self.alert_manager.configure_threshold("authentication_failures", 3, EventSeverity.HIGH)
        
        # Simulate multiple authentication failures
        for i in range(4):
            self.security_logger.log_authentication_event(f"client_{i}", False, "Invalid signature")
            self.alert_manager.record_metric_value("authentication_failures", i + 1)
        
        # Wait for threshold monitoring
        time.sleep(2)
        
        # Check if alert was created
        active_alerts = self.alert_manager.get_active_alerts()
        auth_alerts = [alert for alert in active_alerts if "authentication_failures" in alert['title']]
        
        self.assertTrue(len(auth_alerts) > 0, "Expected authentication failure alert")
    
    def test_anomaly_detection_alert_workflow(self):
        """Test complete anomaly detection alert workflow."""
        # Configure threshold for anomaly scores
        self.alert_manager.configure_threshold("anomaly_score", 0.8, EventSeverity.CRITICAL)
        
        # Simulate high anomaly score
        high_anomaly_score = 0.95
        self.alert_manager.record_metric_value("anomaly_score", high_anomaly_score)
        
        # Log corresponding security event
        anomaly_event = SecurityEvent(
            event_id="anomaly_001",
            event_type=EventType.ANOMALY_DETECTION,
            severity=EventSeverity.CRITICAL,
            client_id="malicious_client_1",
            description=f"High anomaly score detected: {high_anomaly_score}",
            metadata={"anomaly_score": high_anomaly_score},
            timestamp=datetime.now()
        )
        self.security_logger.log_event(anomaly_event)
        
        # Wait for processing
        time.sleep(2)
        
        # Verify alert was created
        active_alerts = self.alert_manager.get_active_alerts()
        anomaly_alerts = [alert for alert in active_alerts if "anomaly_score" in alert['title']]
        
        self.assertTrue(len(anomaly_alerts) > 0, "Expected anomaly score alert")
        
        # Verify alert has correct severity
        if anomaly_alerts:
            self.assertEqual(anomaly_alerts[0]['severity'], EventSeverity.CRITICAL.value)
    
    def test_metrics_collection_and_alerting(self):
        """Test metrics collection triggering alerts."""
        # Configure system resource thresholds
        self.alert_manager.configure_threshold("cpu_usage", 90.0, EventSeverity.MEDIUM)
        self.alert_manager.configure_threshold("memory_usage", 85.0, EventSeverity.MEDIUM)
        
        # Collect system metrics
        metrics = self.metrics_collector.collect_metrics()
        
        # Simulate high resource usage
        self.alert_manager.record_metric_value("cpu_usage", 95.0)
        self.alert_manager.record_metric_value("memory_usage", 90.0)
        
        # Wait for threshold monitoring
        time.sleep(2)
        
        # Check for resource alerts
        active_alerts = self.alert_manager.get_active_alerts()
        resource_alerts = [
            alert for alert in active_alerts 
            if any(metric in alert['title'] for metric in ['cpu_usage', 'memory_usage'])
        ]
        
        self.assertTrue(len(resource_alerts) > 0, "Expected resource usage alerts")
    
    def test_alert_escalation_workflow(self):
        """Test alert escalation workflow."""
        # Create a medium severity alert
        alert_id = self.alert_manager.create_alert(
            "Test Escalation Alert",
            "This alert will be escalated",
            EventSeverity.MEDIUM
        )
        
        # Escalate the alert
        self.alert_manager.escalate_alert(alert_id)
        
        # Verify escalation
        alert = self.alert_manager._alerts[alert_id]
        self.assertTrue(alert.escalated)
        self.assertEqual(alert.severity, EventSeverity.CRITICAL)
    
    def test_multiple_notification_channels(self):
        """Test multiple notification channels."""
        # Add webhook notification channel
        webhook_config = {
            "url": "https://example.com/webhook",
            "headers": {"Authorization": "Bearer test-token"}
        }
        self.alert_manager.add_notification_channel("test_webhook", "webhook", webhook_config)
        
        # Verify channel was added
        self.assertIn("test_webhook", self.alert_manager._notification_channels)
        
        # Create alert (should trigger notifications to all channels)
        alert_id = self.alert_manager.create_alert(
            "Multi-Channel Alert",
            "Testing multiple notification channels",
            EventSeverity.HIGH
        )
        
        # Verify alert was created
        self.assertIsNotNone(alert_id)
        active_alerts = self.alert_manager.get_active_alerts()
        self.assertEqual(len(active_alerts), 1)


if __name__ == '__main__':
    unittest.main()