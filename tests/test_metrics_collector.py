"""
Unit tests for MetricsCollector

Tests comprehensive performance and security metrics collection
including real-time computation and historical analysis.
"""

import os
import sqlite3
import tempfile
import time
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from monitoring.interfaces import SystemMetrics
from monitoring.metrics_collector import MetricsCollector


class TestMetricsCollector(unittest.TestCase):
    """Test cases for MetricsCollector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test database
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_metrics.db")
        
        # Mock psutil to avoid system dependencies in tests
        self.psutil_patcher = patch('monitoring.metrics_collector.psutil')
        self.mock_psutil = self.psutil_patcher.start()
        
        # Configure mock psutil
        self.mock_psutil.cpu_percent.return_value = 25.5
        self.mock_psutil.virtual_memory.return_value = MagicMock(percent=60.0)
        
        # Initialize collector with test configuration
        self.collector = MetricsCollector(
            db_path=self.db_path,
            collection_interval=1,  # Short interval for testing
            history_window=50
        )
        
        # Stop automatic collection for controlled testing
        self.collector.stop_collection()
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Stop collection and cleanup
        self.collector.stop_collection()
        
        # Stop psutil mock
        self.psutil_patcher.stop()
        
        # Clean up temporary files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test collector initialization."""
        # Check that database was created and initialized
        self.assertTrue(os.path.exists(self.db_path))
        
        # Verify database schema
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = [row[0] for row in cursor.fetchall()]
            self.assertIn('system_metrics', tables)
            self.assertIn('custom_metrics', tables)
    
    def test_collect_metrics(self):
        """Test basic metrics collection."""
        # Set counter values
        self.collector.set_counter('active_clients', 5)
        self.collector.set_counter('training_rounds_completed', 10)
        self.collector.set_counter('anomalies_detected', 2)
        self.collector.set_counter('authentication_failures', 1)
        
        # Collect metrics
        metrics = self.collector.collect_metrics()
        
        # Verify metrics structure
        self.assertIsInstance(metrics, SystemMetrics)
        self.assertEqual(metrics.cpu_usage, 25.5)
        self.assertEqual(metrics.memory_usage, 60.0)
        self.assertEqual(metrics.active_clients, 5)
        self.assertEqual(metrics.training_rounds_completed, 10)
        self.assertEqual(metrics.anomalies_detected, 2)
        self.assertEqual(metrics.authentication_failures, 1)
        
        # Verify metrics were stored in database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM system_metrics")
            count = cursor.fetchone()[0]
            self.assertGreaterEqual(count, 1)  # At least 1 record should exist
    
    def test_record_custom_metric(self):
        """Test recording custom metrics."""
        # Record some custom metrics
        self.collector.record_metric('model_accuracy', 0.95)
        self.collector.record_metric('convergence_rate', 0.02)
        self.collector.record_metric('client_dropout_rate', 0.05)
        
        # Verify metrics were stored
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM custom_metrics")
            count = cursor.fetchone()[0]
            self.assertEqual(count, 3)
            
            # Check specific metric
            cursor = conn.execute(
                "SELECT value FROM custom_metrics WHERE name = 'model_accuracy'"
            )
            value = cursor.fetchone()[0]
            self.assertEqual(value, 0.95)
    
    def test_counter_operations(self):
        """Test counter increment and set operations."""
        # Test increment
        self.collector.increment_counter('test_counter', 5)
        self.assertEqual(self.collector.get_counter('test_counter'), 5)
        
        # Test additional increment
        self.collector.increment_counter('test_counter', 3)
        self.assertEqual(self.collector.get_counter('test_counter'), 8)
        
        # Test set
        self.collector.set_counter('test_counter', 20)
        self.assertEqual(self.collector.get_counter('test_counter'), 20)
        
        # Test non-existent counter
        self.assertEqual(self.collector.get_counter('non_existent'), 0)
    
    def test_get_metric_history(self):
        """Test retrieving metric history."""
        # Record metrics with different timestamps
        base_time = datetime.utcnow()
        
        for i in range(5):
            timestamp = base_time + timedelta(minutes=i)
            self.collector.record_metric('test_metric', float(i), timestamp)
        
        # Query history
        start_time = base_time
        end_time = base_time + timedelta(minutes=10)
        
        history = self.collector.get_metric_history('test_metric', start_time, end_time)
        
        # Verify results
        self.assertEqual(len(history), 5)
        
        # Check values are in correct order
        for i, (timestamp, value) in enumerate(history):
            self.assertEqual(value, float(i))
            expected_time = base_time + timedelta(minutes=i)
            # Allow small time differences due to precision
            self.assertLess(abs((timestamp - expected_time).total_seconds()), 1)
    
    def test_get_recent_metrics(self):
        """Test retrieving recent metrics from memory."""
        # Record several metrics
        for i in range(10):
            self.collector.record_metric('recent_test', float(i))
            time.sleep(0.01)  # Small delay to ensure different timestamps
        
        # Get recent metrics
        recent = self.collector.get_recent_metrics('recent_test', limit=5)
        
        # Should get the last 5 values
        self.assertEqual(len(recent), 5)
        
        # Values should be 5, 6, 7, 8, 9
        values = [value for _, value in recent]
        self.assertEqual(values, [5.0, 6.0, 7.0, 8.0, 9.0])
    
    def test_detection_rate_computation(self):
        """Test anomaly detection rate computation."""
        # Clear any existing system metrics first
        with sqlite3.connect(self.collector.db_path) as conn:
            conn.execute('DELETE FROM system_metrics')
            conn.commit()
        
        # Set up anomaly detection history by simulating system metrics collection
        base_time = datetime.utcnow() - timedelta(minutes=10)
        
        # Simulate increasing anomaly count over time by setting counter and collecting metrics
        for i in range(6):  # 6 data points over 5 minutes
            # Set the counter to simulate cumulative anomaly count
            self.collector.set_counter('anomalies_detected', i * 2)
            
            # Manually insert system metrics with specific timestamp
            timestamp = base_time + timedelta(minutes=i)
            with sqlite3.connect(self.collector.db_path) as conn:
                conn.execute('''
                    INSERT INTO system_metrics 
                    (timestamp, cpu_usage, memory_usage, active_clients, 
                     training_rounds_completed, anomalies_detected, authentication_failures)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    timestamp.isoformat(),
                    25.5, 60.0, 0, 0, i * 2, 0
                ))
                conn.commit()
        
        # Debug: Check what data is retrieved
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=15)
        history = self.collector.get_metric_history('anomalies_detected', start_time, end_time)
        
        # Compute detection rate
        detection_rate = self.collector.compute_detection_rate(time_window_minutes=15)
        
        # Should have 6 data points
        self.assertEqual(len(history), 6)
        
        # Rate should be (10 - 0) / 5 minutes = 2.0 anomalies per minute
        self.assertAlmostEqual(detection_rate, 2.0, places=1)
    
    def test_model_accuracy_trend(self):
        """Test model accuracy trend computation."""
        # Set up accuracy history with upward trend (use past timestamps)
        base_time = datetime.utcnow() - timedelta(minutes=30)
        accuracies = [0.80, 0.82, 0.84, 0.86, 0.88]  # Increasing trend
        
        for i, accuracy in enumerate(accuracies):
            timestamp = base_time + timedelta(minutes=i)
            self.collector.record_metric('model_accuracy', accuracy, timestamp)
        
        # Compute trend (use larger window to ensure all data is included)
        trend_stats = self.collector.compute_model_accuracy_trend(time_window_minutes=60)
        
        # Verify statistics
        self.assertEqual(trend_stats['sample_count'], 5)
        self.assertAlmostEqual(trend_stats['mean'], 0.84, places=2)
        self.assertGreater(trend_stats['trend'], 0)  # Positive trend
        self.assertGreater(trend_stats['variance'], 0)  # Some variance
    
    def test_system_health_score(self):
        """Test system health score calculation."""
        # Test with good system state
        self.collector._current_metrics = {
            'cpu_usage': 20.0,      # Low CPU usage (good)
            'memory_usage': 30.0,   # Low memory usage (good)
            'authentication_failures': 0  # No failures (good)
        }
        
        health_score = self.collector.get_system_health_score()
        self.assertGreater(health_score, 0.8)  # Should be high
        
        # Test with poor system state
        self.collector._current_metrics = {
            'cpu_usage': 90.0,      # High CPU usage (bad)
            'memory_usage': 85.0,   # High memory usage (bad)
            'authentication_failures': 10  # Many failures (bad)
        }
        
        health_score = self.collector.get_system_health_score()
        self.assertLess(health_score, 0.5)  # Should be low
    
    def test_performance_summary(self):
        """Test comprehensive performance summary."""
        # Set up some test data
        self.collector.set_counter('active_clients', 8)
        self.collector.set_counter('training_rounds_completed', 15)
        self.collector.record_metric('model_accuracy', 0.92)
        
        # Get performance summary
        summary = self.collector.get_performance_summary()
        
        # Verify summary structure
        required_keys = [
            'timestamp', 'system_health_score', 'cpu_usage', 'memory_usage',
            'active_clients', 'training_rounds_completed', 'anomalies_detected',
            'authentication_failures', 'detection_rate_per_minute', 'model_accuracy_trend'
        ]
        
        for key in required_keys:
            self.assertIn(key, summary)
        
        # Verify some values
        self.assertEqual(summary['active_clients'], 8)
        self.assertEqual(summary['training_rounds_completed'], 15)
        self.assertIsInstance(summary['system_health_score'], float)
        self.assertIsInstance(summary['model_accuracy_trend'], dict)
    
    def test_cleanup_old_metrics(self):
        """Test cleanup of old metrics."""
        # Create metrics with different ages
        base_time = datetime.utcnow()
        
        # Old system metrics (should be deleted)
        old_time = base_time - timedelta(days=35)
        self.collector.set_counter('active_clients', 1)
        
        # Manually insert old system metric
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO system_metrics 
                (timestamp, cpu_usage, memory_usage, active_clients, 
                 training_rounds_completed, anomalies_detected, authentication_failures)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (old_time.isoformat(), 50.0, 60.0, 1, 0, 0, 0))
            
            # Old custom metrics (should be deleted)
            conn.execute('''
                INSERT INTO custom_metrics (name, value, timestamp)
                VALUES (?, ?, ?)
            ''', ('old_metric', 1.0, old_time.isoformat()))
            
            conn.commit()
        
        # Recent metrics (should be kept)
        recent_time = base_time - timedelta(days=10)
        self.collector.record_metric('recent_metric', 2.0, recent_time)
        
        # Cleanup old metrics
        deleted_count = self.collector.cleanup_old_metrics(days_to_keep=30)
        self.assertEqual(deleted_count, 2)  # Should delete 2 old records
        
        # Verify recent metrics are still there
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM custom_metrics WHERE name = 'recent_metric'")
            count = cursor.fetchone()[0]
            self.assertEqual(count, 1)
    
    def test_automatic_collection(self):
        """Test automatic metrics collection."""
        # Create new collector with automatic collection
        auto_collector = MetricsCollector(
            db_path=os.path.join(self.temp_dir, "auto_test.db"),
            collection_interval=0.1,  # Very short interval for testing
            history_window=10
        )
        
        try:
            # Wait for a few collection cycles
            time.sleep(0.5)
            
            # Check that metrics were collected automatically
            with sqlite3.connect(auto_collector.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM system_metrics")
                count = cursor.fetchone()[0]
                self.assertGreater(count, 0)  # Should have collected some metrics
                
        finally:
            auto_collector.stop_collection()
    
    def test_thread_safety(self):
        """Test thread safety of metrics operations."""
        import threading
        
        # Function to record metrics concurrently
        def record_metrics(thread_id):
            for i in range(20):
                self.collector.record_metric(f'thread_{thread_id}_metric', float(i))
                self.collector.increment_counter(f'thread_{thread_id}_counter')
                time.sleep(0.001)
        
        # Create and start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=record_metrics, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all metrics were recorded
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM custom_metrics")
            count = cursor.fetchone()[0]
            self.assertEqual(count, 60)  # 3 threads * 20 metrics each
        
        # Verify counters
        for i in range(3):
            counter_value = self.collector.get_counter(f'thread_{i}_counter')
            self.assertEqual(counter_value, 20)
    
    def test_database_error_handling(self):
        """Test handling of database errors."""
        # Create collector with invalid database path
        invalid_collector = MetricsCollector(
            db_path="/invalid/path/metrics.db",
            collection_interval=60
        )
        
        # Operations should not raise exceptions even with database errors
        try:
            invalid_collector.record_metric('test', 1.0)
            invalid_collector.collect_metrics()
            invalid_collector.cleanup_old_metrics()
        except Exception as e:
            self.fail(f"Operations raised exception with invalid database: {e}")
        
        finally:
            invalid_collector.stop_collection()


if __name__ == '__main__':
    unittest.main()