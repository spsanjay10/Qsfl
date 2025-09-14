"""
Unit tests for SecurityEventLogger

Tests comprehensive audit trail logging functionality including
structured logging, database persistence, and log rotation.
"""

import json
import os
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

from monitoring.interfaces import SecurityEvent, EventType, EventSeverity
from monitoring.security_logger import SecurityEventLogger


class TestSecurityEventLogger(unittest.TestCase):
    """Test cases for SecurityEventLogger class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test logs and database
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = os.path.join(self.temp_dir, "logs")
        self.db_path = os.path.join(self.log_dir, "test_security_events.db")
        
        # Initialize logger with test configuration
        self.logger = SecurityEventLogger(
            log_dir=self.log_dir,
            db_path=self.db_path,
            max_log_size=1024,  # Small size for testing rotation
            backup_count=2
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temporary files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test logger initialization."""
        # Check that log directory was created
        self.assertTrue(Path(self.log_dir).exists())
        
        # Check that database was created and initialized
        self.assertTrue(Path(self.db_path).exists())
        
        # Verify database schema
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='security_events'"
            )
            self.assertIsNotNone(cursor.fetchone())
    
    def test_log_event(self):
        """Test basic event logging functionality."""
        # Create test event
        event = SecurityEvent(
            event_id="test-event-1",
            event_type=EventType.AUTHENTICATION,
            severity=EventSeverity.MEDIUM,
            client_id="client-123",
            description="Test authentication event",
            metadata={"test": "data", "success": True},
            timestamp=datetime.utcnow()
        )
        
        # Log the event
        self.logger.log_event(event)
        
        # Verify event was stored in database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM security_events WHERE event_id = ?",
                (event.event_id,)
            )
            row = cursor.fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row[1], event.event_type.value)  # event_type
            self.assertEqual(row[2], event.severity.value)    # severity
            self.assertEqual(row[3], event.client_id)         # client_id
        
        # Verify event counter was incremented
        counts = self.logger.get_event_counts()
        self.assertEqual(counts[EventType.AUTHENTICATION], 1)
    
    def test_log_authentication_event(self):
        """Test authentication event logging."""
        # Test successful authentication
        self.logger.log_authentication_event("client-123", True, "Valid credentials")
        
        # Test failed authentication
        self.logger.log_authentication_event("client-456", False, "Invalid signature")
        
        # Verify events were logged
        events = self.logger.get_recent_events(10)
        self.assertEqual(len(events), 2)
        
        # Check successful authentication event
        success_event = next(e for e in events if e.metadata.get("success"))
        self.assertEqual(success_event.event_type, EventType.AUTHENTICATION)
        self.assertEqual(success_event.severity, EventSeverity.LOW)
        self.assertEqual(success_event.client_id, "client-123")
        
        # Check failed authentication event
        failure_event = next(e for e in events if not e.metadata.get("success"))
        self.assertEqual(failure_event.event_type, EventType.AUTHENTICATION)
        self.assertEqual(failure_event.severity, EventSeverity.MEDIUM)
        self.assertEqual(failure_event.client_id, "client-456")
    
    def test_log_cryptographic_event(self):
        """Test cryptographic event logging."""
        # Test successful operation
        self.logger.log_cryptographic_event("key_generation", True, "Kyber keypair generated")
        
        # Test failed operation
        self.logger.log_cryptographic_event("signature_verify", False, "Invalid signature format")
        
        # Verify events were logged
        events = self.logger.get_recent_events(10)
        self.assertEqual(len(events), 2)
        
        # Check events have correct type and metadata
        for event in events:
            self.assertEqual(event.event_type, EventType.CRYPTOGRAPHIC)
            self.assertIn("operation", event.metadata)
            self.assertIn("success", event.metadata)
    
    def test_log_anomaly_event(self):
        """Test anomaly detection event logging."""
        test_cases = [
            (0.9, EventSeverity.CRITICAL),
            (0.7, EventSeverity.HIGH),
            (0.5, EventSeverity.MEDIUM),
            (0.2, EventSeverity.LOW)
        ]
        
        for score, expected_severity in test_cases:
            with self.subTest(score=score):
                self.logger.log_anomaly_event("client-test", score, "quarantine")
                
                # Get the most recent event
                events = self.logger.get_recent_events(1)
                event = events[0]
                
                self.assertEqual(event.event_type, EventType.ANOMALY_DETECTION)
                self.assertEqual(event.severity, expected_severity)
                self.assertEqual(event.client_id, "client-test")
                self.assertEqual(event.metadata["anomaly_score"], score)
                self.assertEqual(event.metadata["action_taken"], "quarantine")
    
    def test_get_events_time_range(self):
        """Test retrieving events within time range."""
        # Create events with different timestamps
        base_time = datetime.utcnow()
        
        events = [
            SecurityEvent(
                event_id=f"event-{i}",
                event_type=EventType.SYSTEM,
                severity=EventSeverity.LOW,
                client_id=None,
                description=f"Test event {i}",
                metadata={},
                timestamp=base_time + timedelta(hours=i)
            )
            for i in range(5)
        ]
        
        # Log all events
        for event in events:
            self.logger.log_event(event)
        
        # Query events within specific time range
        start_time = base_time + timedelta(hours=1)
        end_time = base_time + timedelta(hours=3)
        
        retrieved_events = self.logger.get_events(start_time, end_time)
        
        # Should get events 1, 2, and 3
        self.assertEqual(len(retrieved_events), 3)
        
        # Verify events are in correct time range
        for event in retrieved_events:
            self.assertGreaterEqual(event.timestamp, start_time)
            self.assertLessEqual(event.timestamp, end_time)
    
    def test_get_events_by_type(self):
        """Test filtering events by type."""
        # Create events of different types
        event_types = [EventType.AUTHENTICATION, EventType.CRYPTOGRAPHIC, EventType.ANOMALY_DETECTION]
        
        for event_type in event_types:
            event = SecurityEvent(
                event_id=f"event-{event_type.value}",
                event_type=event_type,
                severity=EventSeverity.LOW,
                client_id=None,
                description=f"Test {event_type.value} event",
                metadata={},
                timestamp=datetime.utcnow()
            )
            self.logger.log_event(event)
        
        # Query events by specific type
        start_time = datetime.utcnow() - timedelta(minutes=1)
        end_time = datetime.utcnow() + timedelta(minutes=1)
        
        auth_events = self.logger.get_events(start_time, end_time, EventType.AUTHENTICATION)
        self.assertEqual(len(auth_events), 1)
        self.assertEqual(auth_events[0].event_type, EventType.AUTHENTICATION)
    
    def test_get_recent_events(self):
        """Test retrieving recent events."""
        # Create multiple events
        for i in range(10):
            event = SecurityEvent(
                event_id=f"recent-event-{i}",
                event_type=EventType.SYSTEM,
                severity=EventSeverity.LOW,
                client_id=None,
                description=f"Recent event {i}",
                metadata={},
                timestamp=datetime.utcnow() + timedelta(seconds=i)
            )
            self.logger.log_event(event)
        
        # Get recent events with limit
        recent_events = self.logger.get_recent_events(5)
        self.assertEqual(len(recent_events), 5)
        
        # Verify events are ordered by timestamp (most recent first)
        for i in range(len(recent_events) - 1):
            self.assertGreaterEqual(
                recent_events[i].timestamp,
                recent_events[i + 1].timestamp
            )
    
    def test_cleanup_old_events(self):
        """Test cleanup of old events."""
        # Create events with different ages
        base_time = datetime.utcnow()
        
        # Old events (should be deleted)
        for i in range(3):
            old_event = SecurityEvent(
                event_id=f"old-event-{i}",
                event_type=EventType.SYSTEM,
                severity=EventSeverity.LOW,
                client_id=None,
                description=f"Old event {i}",
                metadata={},
                timestamp=base_time - timedelta(days=35)  # 35 days old
            )
            self.logger.log_event(old_event)
        
        # Recent events (should be kept)
        for i in range(2):
            recent_event = SecurityEvent(
                event_id=f"recent-event-{i}",
                event_type=EventType.SYSTEM,
                severity=EventSeverity.LOW,
                client_id=None,
                description=f"Recent event {i}",
                metadata={},
                timestamp=base_time - timedelta(days=10)  # 10 days old
            )
            self.logger.log_event(recent_event)
        
        # Cleanup events older than 30 days
        deleted_count = self.logger.cleanup_old_events(30)
        self.assertEqual(deleted_count, 3)
        
        # Verify only recent events remain
        all_events = self.logger.get_recent_events(100)
        self.assertEqual(len(all_events), 2)
    
    def test_database_error_handling(self):
        """Test handling of database errors."""
        # Create logger with invalid database path
        invalid_logger = SecurityEventLogger(
            log_dir=self.log_dir,
            db_path="/invalid/path/database.db"
        )
        
        # Logging should not raise exception even with database errors
        event = SecurityEvent(
            event_id="test-error",
            event_type=EventType.SYSTEM,
            severity=EventSeverity.LOW,
            client_id=None,
            description="Test error handling",
            metadata={},
            timestamp=datetime.utcnow()
        )
        
        # This should not raise an exception
        try:
            invalid_logger.log_event(event)
        except Exception as e:
            self.fail(f"log_event raised an exception: {e}")
    
    def test_concurrent_logging(self):
        """Test thread safety of logging operations."""
        import threading
        import time
        
        # Function to log events concurrently
        def log_events(thread_id):
            for i in range(10):
                event = SecurityEvent(
                    event_id=f"thread-{thread_id}-event-{i}",
                    event_type=EventType.SYSTEM,
                    severity=EventSeverity.LOW,
                    client_id=f"client-{thread_id}",
                    description=f"Concurrent event from thread {thread_id}",
                    metadata={"thread_id": thread_id, "event_num": i},
                    timestamp=datetime.utcnow()
                )
                self.logger.log_event(event)
                time.sleep(0.001)  # Small delay to simulate real usage
        
        # Create and start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=log_events, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all events were logged
        all_events = self.logger.get_recent_events(100)
        self.assertEqual(len(all_events), 50)  # 5 threads * 10 events each
    
    def test_structured_log_format(self):
        """Test that log entries are properly formatted as JSON."""
        # Create test event
        event = SecurityEvent(
            event_id="format-test",
            event_type=EventType.AUTHENTICATION,
            severity=EventSeverity.HIGH,
            client_id="client-format",
            description="Test log format",
            metadata={"key": "value", "number": 42},
            timestamp=datetime.utcnow()
        )
        
        # Log the event
        self.logger.log_event(event)
        
        # Read the log file and verify JSON format
        log_file = Path(self.log_dir) / "security_events.log"
        self.assertTrue(log_file.exists())
        
        with open(log_file, 'r') as f:
            log_lines = f.readlines()
        
        # Find our test event in the log
        test_log_line = None
        for line in log_lines:
            if "format-test" in line:
                test_log_line = line
                break
        
        self.assertIsNotNone(test_log_line)
        
        # Parse the JSON structure
        try:
            # Log format is now just the JSON event directly
            event_json = json.loads(test_log_line.strip())
            
            # Verify the JSON contains our event data
            self.assertEqual(event_json["event_id"], "format-test")
            self.assertEqual(event_json["event_type"], "authentication")
            self.assertEqual(event_json["severity"], "high")
            self.assertEqual(event_json["client_id"], "client-format")
            
        except json.JSONDecodeError as e:
            self.fail(f"Log entry is not valid JSON: {e}")


if __name__ == '__main__':
    unittest.main()