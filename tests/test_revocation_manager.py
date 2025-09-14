"""
Unit tests for RevocationManager

Tests revocation scenarios, suspicious behavior tracking, and automated triggers.
"""

import pytest
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from auth.revocation_manager import (
    RevocationManager, 
    RevocationReason, 
    RevocationRecord, 
    SuspiciousBehaviorEvent
)


class TestRevocationManager:
    """Test suite for RevocationManager."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        yield path
        try:
            if os.path.exists(path):
                os.unlink(path)
        except PermissionError:
            pass
    
    @pytest.fixture
    def revocation_manager(self, temp_db):
        """Create revocation manager with temporary database."""
        return RevocationManager(
            db_path=temp_db,
            suspicious_behavior_threshold=100,
            auto_revoke_enabled=True
        )
    
    def test_init_creates_database(self, temp_db):
        """Test that initialization creates database tables."""
        manager = RevocationManager(db_path=temp_db)
        
        # Check that database file exists
        assert os.path.exists(temp_db)
        
        # Check that tables were created
        import sqlite3
        with sqlite3.connect(temp_db) as conn:
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name IN ('revocation_records', 'suspicious_behavior', 'behavior_scores')
            """)
            tables = [row[0] for row in cursor.fetchall()]
            assert 'revocation_records' in tables
            assert 'suspicious_behavior' in tables
            assert 'behavior_scores' in tables
    
    def test_revoke_credential_success(self, revocation_manager):
        """Test successful credential revocation."""
        client_id = "test_client_001"
        reason = "manual"
        details = "Test revocation"
        
        revocation_manager.revoke_credential(client_id, reason, "admin", details)
        
        # Verify revocation
        assert revocation_manager.is_revoked(client_id) is True
        
        # Check revocation record
        record = revocation_manager.get_revocation_record(client_id)
        assert record is not None
        assert record.client_id == client_id
        assert record.reason == RevocationReason.MANUAL
        assert record.details == details
        assert record.revoked_by == "admin"
    
    def test_revoke_credential_with_enum_reason(self, revocation_manager):
        """Test revocation with different reason types."""
        client_id = "test_client_002"
        
        revocation_manager.revoke_credential(
            client_id, 
            RevocationReason.SECURITY_BREACH.value, 
            "system"
        )
        
        record = revocation_manager.get_revocation_record(client_id)
        assert record.reason == RevocationReason.SECURITY_BREACH
    
    def test_is_revoked_nonexistent_client(self, revocation_manager):
        """Test checking revocation status for non-existent client."""
        assert revocation_manager.is_revoked("nonexistent_client") is False
    
    def test_get_revocation_list(self, revocation_manager):
        """Test getting list of revoked clients."""
        clients = ["client_001", "client_002", "client_003"]
        
        # Revoke some clients
        for client_id in clients[:2]:
            revocation_manager.revoke_credential(client_id, "manual", "admin")
        
        revoked_list = revocation_manager.get_revocation_list()
        
        assert len(revoked_list) == 2
        assert "client_001" in revoked_list
        assert "client_002" in revoked_list
        assert "client_003" not in revoked_list
    
    def test_get_revocation_record_nonexistent(self, revocation_manager):
        """Test getting revocation record for non-existent client."""
        record = revocation_manager.get_revocation_record("nonexistent_client")
        assert record is None
    
    def test_record_suspicious_behavior(self, revocation_manager):
        """Test recording suspicious behavior events."""
        client_id = "test_client_003"
        event_type = "failed_authentication"
        severity = 5
        details = {"attempt_count": 3, "source_ip": "192.168.1.1"}
        source = "auth_service"
        
        revocation_manager.record_suspicious_behavior(
            client_id, event_type, severity, details, source
        )
        
        # Check behavior score was updated
        score = revocation_manager.get_behavior_score(client_id)
        assert score > 0
        
        # Check event was recorded
        events = revocation_manager.get_suspicious_events(client_id)
        assert len(events) == 1
        assert events[0].client_id == client_id
        assert events[0].event_type == event_type
        assert events[0].severity == severity
        assert events[0].details == details
        assert events[0].source == source
    
    def test_behavior_score_calculation(self, revocation_manager):
        """Test behavior score calculation with different event types."""
        client_id = "test_client_004"
        
        # Record different types of suspicious behavior
        revocation_manager.record_suspicious_behavior(client_id, "failed_authentication", 5)
        initial_score = revocation_manager.get_behavior_score(client_id)
        
        revocation_manager.record_suspicious_behavior(client_id, "malicious_payload", 8)
        final_score = revocation_manager.get_behavior_score(client_id)
        
        # Score should increase
        assert final_score > initial_score
        
        # Malicious payload should have higher weight than failed auth
        assert final_score > initial_score * 2
    
    def test_severity_validation(self, revocation_manager):
        """Test that severity is properly validated."""
        client_id = "test_client_005"
        
        # Test severity clamping
        revocation_manager.record_suspicious_behavior(client_id, "test_event", 15)  # Should clamp to 10
        revocation_manager.record_suspicious_behavior(client_id, "test_event", -5)  # Should clamp to 1
        
        events = revocation_manager.get_suspicious_events(client_id)
        assert all(1 <= event.severity <= 10 for event in events)
    
    def test_automatic_revocation_threshold(self, revocation_manager):
        """Test automatic revocation when threshold is exceeded."""
        client_id = "test_client_006"
        
        # Record enough suspicious behavior to trigger automatic revocation
        # Using malicious_payload (weight 50) with severity 10 = 50 points per event
        # Need 100 points total, so 2 events should trigger revocation
        revocation_manager.record_suspicious_behavior(client_id, "malicious_payload", 10)
        assert not revocation_manager.is_revoked(client_id)
        
        revocation_manager.record_suspicious_behavior(client_id, "malicious_payload", 10)
        
        # Should now be automatically revoked
        assert revocation_manager.is_revoked(client_id)
        
        # Check that it was an automatic revocation
        record = revocation_manager.get_revocation_record(client_id)
        assert record.automatic is True
        assert record.reason == RevocationReason.AUTOMATED_TRIGGER
    
    def test_auto_revoke_disabled(self, temp_db):
        """Test that automatic revocation can be disabled."""
        manager = RevocationManager(
            db_path=temp_db,
            suspicious_behavior_threshold=50,
            auto_revoke_enabled=False
        )
        
        client_id = "test_client_007"
        
        # Record behavior that would normally trigger revocation
        manager.record_suspicious_behavior(client_id, "malicious_payload", 10)
        manager.record_suspicious_behavior(client_id, "malicious_payload", 10)
        
        # Should not be revoked since auto-revoke is disabled
        assert not manager.is_revoked(client_id)
        
        # But score should still be tracked
        assert manager.get_behavior_score(client_id) >= 50
    
    def test_custom_revocation_trigger(self, revocation_manager):
        """Test adding and using custom revocation triggers."""
        client_id = "test_client_008"
        
        # Create a custom trigger that revokes on any "critical_event"
        def critical_event_trigger(client_id: str, event: SuspiciousBehaviorEvent) -> bool:
            return event.event_type == "critical_event"
        
        revocation_manager.add_revocation_trigger(critical_event_trigger)
        
        # Record a critical event
        revocation_manager.record_suspicious_behavior(client_id, "critical_event", 1)
        
        # Should be revoked despite low severity and score
        assert revocation_manager.is_revoked(client_id)
    
    def test_remove_revocation_trigger(self, revocation_manager):
        """Test removing revocation triggers."""
        def test_trigger(client_id: str, event: SuspiciousBehaviorEvent) -> bool:
            return False
        
        # Add trigger
        revocation_manager.add_revocation_trigger(test_trigger)
        
        # Remove trigger
        result = revocation_manager.remove_revocation_trigger(test_trigger)
        assert result is True
        
        # Try to remove again (should fail)
        result = revocation_manager.remove_revocation_trigger(test_trigger)
        assert result is False
    
    def test_get_suspicious_events_filtering(self, revocation_manager):
        """Test filtering of suspicious events by time and limit."""
        client_id = "test_client_009"
        
        # Record events
        for i in range(5):
            revocation_manager.record_suspicious_behavior(
                client_id, f"event_{i}", 3, source=f"source_{i}"
            )
        
        # Test limit
        events = revocation_manager.get_suspicious_events(client_id, limit=3)
        assert len(events) == 3
        
        # Test time filtering (should get all events since they're recent)
        events = revocation_manager.get_suspicious_events(client_id, days_back=1)
        assert len(events) == 5
        
        # Test with very short time window (should get no events)
        events = revocation_manager.get_suspicious_events(client_id, days_back=0)
        assert len(events) == 0
    
    def test_cleanup_old_events(self, revocation_manager):
        """Test cleanup of old suspicious behavior events."""
        client_id = "test_client_010"
        
        # Record some events
        revocation_manager.record_suspicious_behavior(client_id, "test_event", 5)
        
        # Manually insert old event
        import sqlite3
        old_date = datetime.now() - timedelta(days=100)
        with sqlite3.connect(revocation_manager.db_path) as conn:
            conn.execute("""
                INSERT INTO suspicious_behavior 
                (client_id, event_type, severity, timestamp, details, source)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (client_id, "old_event", 5, old_date.isoformat(), "{}", "test"))
            conn.commit()
        
        # Verify we have 2 events
        events = revocation_manager.get_suspicious_events(client_id, days_back=365)
        assert len(events) == 2
        
        # Cleanup events older than 30 days
        removed_count = revocation_manager.cleanup_old_events(days_to_keep=30)
        
        assert removed_count == 1
        
        # Verify only recent event remains
        events = revocation_manager.get_suspicious_events(client_id, days_back=365)
        assert len(events) == 1
        assert events[0].event_type == "test_event"
    
    def test_revocation_statistics(self, revocation_manager):
        """Test getting revocation statistics."""
        # Create some test data
        clients = ["client_001", "client_002", "client_003"]
        
        # Manual revocations
        revocation_manager.revoke_credential(clients[0], "manual", "admin")
        revocation_manager.revoke_credential(clients[1], "security_breach", "admin")
        
        # Automatic revocation
        revocation_manager.record_suspicious_behavior(clients[2], "malicious_payload", 10)
        revocation_manager.record_suspicious_behavior(clients[2], "malicious_payload", 10)
        
        # Get statistics
        stats = revocation_manager.get_revocation_statistics()
        
        assert stats['total_revocations'] == 3
        assert stats['manual_revocations'] == 2
        assert stats['automatic_revocations'] == 1
        assert 'revocations_by_reason' in stats
        assert stats['revocation_threshold'] == 100
        assert stats['auto_revoke_enabled'] is True
    
    def test_reset_behavior_score(self, revocation_manager):
        """Test resetting behavior score for a client."""
        client_id = "test_client_011"
        
        # Build up behavior score
        revocation_manager.record_suspicious_behavior(client_id, "failed_authentication", 8)
        initial_score = revocation_manager.get_behavior_score(client_id)
        assert initial_score > 0
        
        # Reset score
        revocation_manager.reset_behavior_score(client_id)
        
        # Score should be 0
        final_score = revocation_manager.get_behavior_score(client_id)
        assert final_score == 0
    
    def test_unrevoke_client(self, revocation_manager):
        """Test unrevoking a client."""
        client_id = "test_client_012"
        
        # Revoke client
        revocation_manager.revoke_credential(client_id, "manual", "admin")
        assert revocation_manager.is_revoked(client_id) is True
        
        # Unrevoke client
        result = revocation_manager.unrevoke_client(client_id, "administrative_decision")
        assert result is True
        assert revocation_manager.is_revoked(client_id) is False
        
        # Try to unrevoke again (should fail)
        result = revocation_manager.unrevoke_client(client_id)
        assert result is False
    
    def test_trigger_exception_handling(self, revocation_manager):
        """Test that exceptions in custom triggers are handled gracefully."""
        client_id = "test_client_013"
        
        # Create a trigger that raises an exception
        def failing_trigger(client_id: str, event: SuspiciousBehaviorEvent) -> bool:
            raise Exception("Trigger failed")
        
        revocation_manager.add_revocation_trigger(failing_trigger)
        
        # Record behavior - should not crash despite trigger exception
        revocation_manager.record_suspicious_behavior(client_id, "test_event", 5)
        
        # Client should not be revoked due to trigger failure
        assert not revocation_manager.is_revoked(client_id)
    
    def test_multiple_revocations_same_client(self, revocation_manager):
        """Test handling multiple revocations for the same client."""
        client_id = "test_client_014"
        
        # Revoke client multiple times
        revocation_manager.revoke_credential(client_id, "manual", "admin1")
        revocation_manager.revoke_credential(client_id, "security_breach", "admin2")
        
        # Should still be revoked
        assert revocation_manager.is_revoked(client_id) is True
        
        # Should get the most recent revocation record
        record = revocation_manager.get_revocation_record(client_id)
        assert record.reason == RevocationReason.SECURITY_BREACH
        assert record.revoked_by == "admin2"
    
    def test_behavior_score_persistence(self, revocation_manager):
        """Test that behavior scores persist across manager instances."""
        client_id = "test_client_015"
        
        # Record behavior
        revocation_manager.record_suspicious_behavior(client_id, "failed_authentication", 7)
        original_score = revocation_manager.get_behavior_score(client_id)
        
        # Create new manager instance with same database
        new_manager = RevocationManager(db_path=revocation_manager.db_path)
        
        # Score should be preserved
        preserved_score = new_manager.get_behavior_score(client_id)
        assert preserved_score == original_score
    
    def test_concurrent_behavior_recording(self, revocation_manager):
        """Test concurrent recording of suspicious behavior."""
        client_id = "test_client_016"
        
        # Record multiple events rapidly
        for i in range(10):
            revocation_manager.record_suspicious_behavior(
                client_id, 
                "rapid_event", 
                3, 
                {"sequence": i}
            )
        
        # All events should be recorded
        events = revocation_manager.get_suspicious_events(client_id)
        assert len(events) == 10
        
        # Behavior score should reflect all events
        score = revocation_manager.get_behavior_score(client_id)
        assert score > 0


if __name__ == "__main__":
    pytest.main([__file__])