"""
Unit tests for AlertManager

Tests alert generation, threshold monitoring, notification mechanisms,
and escalation procedures.
"""

import json
import os
import shutil
import sqlite3
import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from monitoring.alert_manager import AlertManager, Alert, AlertThreshold, NotificationChannel
from monitoring.interfaces import EventSeverity


class TestAlertManager(unittest.TestCase):
    """Test cases for AlertManager functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_alerts.db")
        self.config_path = os.path.join(self.temp_dir, "test_config.json")
        
        # Create test configuration
        test_config = {
            "thresholds": [
                {
                    "metric_name": "test_metric",
                    "threshold": 10.0,
                    "severity": "high",
                    "comparison": "greater_than",
                    "window_minutes": 1,
                    "enabled": True
                }
            ],
            "notification_channels": [
                {
                    "channel_id": "test_email",
                    "channel_type": "email",
                    "config": {
                        "smtp_server": "smtp.test.com",
                        "username": "test@test.com",
                        "password": "password",
                        "recipients": ["admin@test.com"]
                    },
                    "enabled": True
                }
            ]
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(test_config, f)
        
        self.alert_manager = AlertManager(db_path=self.db_path, config_path=self.config_path)
    
    def tearDown(self):
        """Clean up test environment."""
        self.alert_manager.shutdown()
        
        # Give some time for database connections to close
        time.sleep(0.1)
        
        # Clean up temporary directory
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass  # Ignore cleanup errors in tests
    
    def test_create_alert(self):
        """Test alert creation."""
        title = "Test Alert"
        description = "This is a test alert"
        severity = EventSeverity.HIGH
        
        alert_id = self.alert_manager.create_alert(title, description, severity)
        
        # Verify alert was created
        self.assertIsInstance(alert_id, str)
        self.assertTrue(len(alert_id) > 0)
        
        # Verify alert is in active alerts
        active_alerts = self.alert_manager.get_active_alerts()
        self.assertEqual(len(active_alerts), 1)
        
        alert = active_alerts[0]
        self.assertEqual(alert['title'], title)
        self.assertEqual(alert['description'], description)
        self.assertEqual(alert['severity'], severity.value)
        self.assertIsNone(alert['resolved_at'])
    
    def test_resolve_alert(self):
        """Test alert resolution."""
        # Create an alert
        alert_id = self.alert_manager.create_alert("Test", "Description", EventSeverity.MEDIUM)
        
        # Verify it's active
        active_alerts = self.alert_manager.get_active_alerts()
        self.assertEqual(len(active_alerts), 1)
        
        # Resolve the alert
        self.alert_manager.resolve_alert(alert_id)
        
        # Verify it's no longer active
        active_alerts = self.alert_manager.get_active_alerts()
        self.assertEqual(len(active_alerts), 0)
    
    def test_resolve_nonexistent_alert(self):
        """Test resolving non-existent alert raises error."""
        with self.assertRaises(ValueError):
            self.alert_manager.resolve_alert("nonexistent_id")
    
    def test_configure_threshold(self):
        """Test threshold configuration."""
        metric_name = "cpu_usage"
        threshold = 85.0
        severity = EventSeverity.HIGH
        
        self.alert_manager.configure_threshold(metric_name, threshold, severity)
        
        # Verify threshold was configured
        self.assertIn(metric_name, self.alert_manager._thresholds)
        configured_threshold = self.alert_manager._thresholds[metric_name]
        self.assertEqual(configured_threshold.threshold, threshold)
        self.assertEqual(configured_threshold.severity, severity)
    
    def test_add_notification_channel(self):
        """Test adding notification channel."""
        channel_id = "test_webhook"
        channel_type = "webhook"
        config = {"url": "https://example.com/webhook"}
        
        self.alert_manager.add_notification_channel(channel_id, channel_type, config)
        
        # Verify channel was added
        self.assertIn(channel_id, self.alert_manager._notification_channels)
        channel = self.alert_manager._notification_channels[channel_id]
        self.assertEqual(channel.channel_type, channel_type)
        self.assertEqual(channel.config, config)
    
    def test_record_metric_value(self):
        """Test recording metric values."""
        metric_name = "test_metric"
        value = 15.0
        
        self.alert_manager.record_metric_value(metric_name, value)
        
        # Verify metric was recorded
        self.assertIn(metric_name, self.alert_manager._metric_history)
        history = self.alert_manager._metric_history[metric_name]
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0][1], value)
    
    def test_threshold_violation_detection(self):
        """Test automatic threshold violation detection."""
        # Configure a low threshold for testing
        metric_name = "test_metric"
        threshold = 5.0
        self.alert_manager.configure_threshold(metric_name, threshold, EventSeverity.HIGH)
        
        # Record a value that exceeds the threshold
        violating_value = 10.0
        self.alert_manager.record_metric_value(metric_name, violating_value)
        
        # Wait for monitoring thread to process
        time.sleep(2)
        
        # Check if alert was created
        active_alerts = self.alert_manager.get_active_alerts()
        threshold_alerts = [
            alert for alert in active_alerts
            if "Threshold Violation" in alert['title'] and metric_name in alert['title']
        ]
        
        self.assertTrue(len(threshold_alerts) > 0)
    
    def test_escalate_alert(self):
        """Test alert escalation."""
        # Create an alert
        alert_id = self.alert_manager.create_alert("Test Alert", "Description", EventSeverity.MEDIUM)
        
        # Escalate the alert
        self.alert_manager.escalate_alert(alert_id)
        
        # Verify escalation
        alert = self.alert_manager._alerts[alert_id]
        self.assertTrue(alert.escalated)
        self.assertEqual(alert.severity, EventSeverity.CRITICAL)
    
    @patch('builtins.print')
    def test_console_notification(self, mock_print):
        """Test console notification."""
        # Create an alert (should trigger console notification)
        self.alert_manager.create_alert("Test Alert", "Test Description", EventSeverity.HIGH)
        
        # Verify console output was called
        mock_print.assert_called()
    
    @patch('smtplib.SMTP')
    def test_email_notification(self, mock_smtp):
        """Test email notification."""
        # Mock SMTP server
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        
        # Add email channel
        email_config = {
            "smtp_server": "smtp.test.com",
            "smtp_port": 587,
            "username": "test@test.com",
            "password": "password",
            "recipients": ["admin@test.com"]
        }
        self.alert_manager.add_notification_channel("test_email", "email", email_config)
        
        # Create an alert
        self.alert_manager.create_alert("Email Test", "Test Description", EventSeverity.CRITICAL)
        
        # Verify email was sent
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once()
        mock_server.send_message.assert_called_once()
    
    @patch('requests.post')
    def test_webhook_notification(self, mock_post):
        """Test webhook notification."""
        # Mock successful webhook response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        # Add webhook channel
        webhook_config = {
            "url": "https://example.com/webhook",
            "headers": {"Authorization": "Bearer token"}
        }
        self.alert_manager.add_notification_channel("test_webhook", "webhook", webhook_config)
        
        # Create an alert
        self.alert_manager.create_alert("Webhook Test", "Test Description", EventSeverity.HIGH)
        
        # Verify webhook was called
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        self.assertEqual(call_args[1]['json']['title'], "Webhook Test")
        self.assertEqual(call_args[1]['json']['severity'], "high")
    
    def test_database_persistence(self):
        """Test alert persistence in database."""
        # Create an alert
        alert_id = self.alert_manager.create_alert("Persistent Alert", "Description", EventSeverity.MEDIUM)
        
        # Verify alert is in database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT * FROM alerts WHERE alert_id = ?", (alert_id,))
            row = cursor.fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row[1], "Persistent Alert")  # title column
    
    def test_threshold_persistence(self):
        """Test threshold configuration persistence."""
        metric_name = "persistent_metric"
        threshold = 42.0
        severity = EventSeverity.CRITICAL
        
        self.alert_manager.configure_threshold(metric_name, threshold, severity)
        
        # Verify threshold is in database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT * FROM alert_thresholds WHERE metric_name = ?", (metric_name,))
            row = cursor.fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row[1], threshold)  # threshold column
            self.assertEqual(row[2], severity.value)  # severity column
    
    def test_notification_channel_persistence(self):
        """Test notification channel persistence."""
        channel_id = "persistent_channel"
        channel_type = "webhook"
        config = {"url": "https://persistent.example.com"}
        
        self.alert_manager.add_notification_channel(channel_id, channel_type, config)
        
        # Verify channel is in database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT * FROM notification_channels WHERE channel_id = ?", (channel_id,))
            row = cursor.fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row[1], channel_type)  # channel_type column
            self.assertEqual(json.loads(row[2]), config)  # config column
    
    def test_metric_history_cleanup(self):
        """Test metric history cleanup (keeps only recent values)."""
        metric_name = "cleanup_test"
        
        # Record old value (should be cleaned up)
        old_timestamp = datetime.now() - timedelta(hours=25)
        self.alert_manager.record_metric_value(metric_name, 1.0, old_timestamp)
        
        # Record recent value (should be kept)
        recent_timestamp = datetime.now() - timedelta(minutes=30)
        self.alert_manager.record_metric_value(metric_name, 2.0, recent_timestamp)
        
        # Trigger cleanup by recording another value
        self.alert_manager.record_metric_value(metric_name, 3.0)
        
        # Verify only recent values are kept
        history = self.alert_manager._metric_history[metric_name]
        values = [value for _, value in history]
        self.assertNotIn(1.0, values)  # Old value should be cleaned up
        self.assertIn(2.0, values)     # Recent value should be kept
        self.assertIn(3.0, values)     # New value should be kept
    
    def test_duplicate_threshold_alerts_prevention(self):
        """Test that duplicate threshold alerts are not created."""
        metric_name = "duplicate_test"
        threshold = 5.0
        
        # Configure threshold
        self.alert_manager.configure_threshold(metric_name, threshold, EventSeverity.MEDIUM)
        
        # Record multiple violating values
        for i in range(3):
            self.alert_manager.record_metric_value(metric_name, 10.0)
            time.sleep(0.1)
        
        # Wait for monitoring
        time.sleep(2)
        
        # Should only have one threshold alert
        active_alerts = self.alert_manager.get_active_alerts()
        threshold_alerts = [
            alert for alert in active_alerts
            if "Threshold Violation" in alert['title'] and metric_name in alert['title']
        ]
        
        self.assertLessEqual(len(threshold_alerts), 1)
    
    def test_different_threshold_comparisons(self):
        """Test different threshold comparison operators."""
        # Test less_than comparison
        threshold_lt = AlertThreshold(
            metric_name="test_lt",
            threshold=5.0,
            severity=EventSeverity.MEDIUM,
            comparison="less_than"
        )
        
        # Test equals comparison
        threshold_eq = AlertThreshold(
            metric_name="test_eq", 
            threshold=10.0,
            severity=EventSeverity.LOW,
            comparison="equals"
        )
        
        # Test violations
        self.assertTrue(self.alert_manager._check_threshold_violation([3.0], threshold_lt))
        self.assertFalse(self.alert_manager._check_threshold_violation([7.0], threshold_lt))
        
        self.assertTrue(self.alert_manager._check_threshold_violation([10.0], threshold_eq))
        self.assertFalse(self.alert_manager._check_threshold_violation([9.5], threshold_eq))
    
    def test_configuration_loading_failure(self):
        """Test graceful handling of configuration loading failure."""
        # Create AlertManager with invalid config path
        invalid_config_path = "/nonexistent/config.json"
        
        # Should not raise exception and use defaults
        alert_manager = AlertManager(config_path=invalid_config_path)
        
        # Should have default thresholds
        self.assertIn("anomaly_score", alert_manager._thresholds)
        self.assertIn("console", alert_manager._notification_channels)
        
        alert_manager.shutdown()
    
    def test_concurrent_alert_creation(self):
        """Test thread safety of alert creation."""
        def create_alerts():
            for i in range(10):
                self.alert_manager.create_alert(f"Concurrent Alert {i}", "Description", EventSeverity.LOW)
        
        # Create alerts from multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=create_alerts)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all alerts were created
        active_alerts = self.alert_manager.get_active_alerts()
        self.assertEqual(len(active_alerts), 30)


class TestAlertDataStructures(unittest.TestCase):
    """Test cases for alert data structures."""
    
    def test_alert_creation(self):
        """Test Alert dataclass creation."""
        alert = Alert(
            alert_id="test_id",
            title="Test Alert",
            description="Test Description",
            severity=EventSeverity.HIGH,
            created_at=datetime.now()
        )
        
        self.assertEqual(alert.alert_id, "test_id")
        self.assertEqual(alert.title, "Test Alert")
        self.assertEqual(alert.severity, EventSeverity.HIGH)
        self.assertIsNone(alert.resolved_at)
        self.assertIsInstance(alert.metadata, dict)
    
    def test_alert_threshold_creation(self):
        """Test AlertThreshold dataclass creation."""
        threshold = AlertThreshold(
            metric_name="cpu_usage",
            threshold=80.0,
            severity=EventSeverity.MEDIUM
        )
        
        self.assertEqual(threshold.metric_name, "cpu_usage")
        self.assertEqual(threshold.threshold, 80.0)
        self.assertEqual(threshold.comparison, "greater_than")  # default
        self.assertEqual(threshold.window_minutes, 5)  # default
        self.assertTrue(threshold.enabled)  # default
    
    def test_notification_channel_creation(self):
        """Test NotificationChannel dataclass creation."""
        channel = NotificationChannel(
            channel_id="email_channel",
            channel_type="email",
            config={"smtp_server": "smtp.example.com"}
        )
        
        self.assertEqual(channel.channel_id, "email_channel")
        self.assertEqual(channel.channel_type, "email")
        self.assertTrue(channel.enabled)  # default


if __name__ == '__main__':
    unittest.main()