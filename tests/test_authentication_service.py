"""
Integration tests for AuthenticationService

Tests complete authentication workflows including registration, verification, and failure handling.
"""

import pytest
import tempfile
import os
import time
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from auth.authentication_service import AuthenticationService, AuthenticationAttempt
from auth.credential_manager import ClientCredentialManager
from auth.revocation_manager import RevocationManager
from auth.interfaces import CredentialStatus
from pq_security.dilithium import DilithiumSigner


class TestAuthenticationService:
    """Test suite for AuthenticationService."""
    
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
    def credential_manager(self, temp_db):
        """Create credential manager with temporary database."""
        return ClientCredentialManager(db_path=temp_db, credential_validity_days=30)
    
    @pytest.fixture
    def revocation_manager(self):
        """Create revocation manager with temporary database."""
        fd, path = tempfile.mkstemp(suffix='_revocation.db')
        os.close(fd)
        manager = RevocationManager(db_path=path, suspicious_behavior_threshold=50)
        yield manager
        try:
            if os.path.exists(path):
                os.unlink(path)
        except PermissionError:
            pass
    
    @pytest.fixture
    def auth_service(self, credential_manager, revocation_manager):
        """Create authentication service."""
        return AuthenticationService(
            credential_manager=credential_manager,
            revocation_manager=revocation_manager,
            max_failed_attempts=3,
            lockout_duration_minutes=5
        )
    
    @pytest.fixture
    def auth_service_no_revocation(self, credential_manager):
        """Create authentication service without revocation manager."""
        return AuthenticationService(
            credential_manager=credential_manager,
            max_failed_attempts=3,
            lockout_duration_minutes=5
        )
    
    @pytest.fixture
    def dilithium_signer(self):
        """Create Dilithium signer for testing."""
        return DilithiumSigner(security_level=3)
    
    def test_register_client_success(self, auth_service):
        """Test successful client registration."""
        client_id = "test_client_001"
        metadata = {"organization": "test_org"}
        
        credentials = auth_service.register_client(client_id, metadata)
        
        assert credentials.client_id == client_id
        assert credentials.status == CredentialStatus.ACTIVE
        assert isinstance(credentials.public_key, bytes)
        assert isinstance(credentials.private_key, bytes)
        
        # Verify client is valid
        assert auth_service.is_client_valid(client_id) is True
    
    def test_register_duplicate_client(self, auth_service):
        """Test registering duplicate client raises error."""
        client_id = "test_client_002"
        
        # Register first time
        auth_service.register_client(client_id)
        
        # Try to register again
        with pytest.raises(ValueError, match="already has active credentials"):
            auth_service.register_client(client_id)
    
    def test_authenticate_client_success(self, auth_service, dilithium_signer):
        """Test successful client authentication."""
        client_id = "test_client_003"
        
        # Register client
        credentials = auth_service.register_client(client_id)
        
        # Create test message and signature
        test_message = b"test_authentication_message"
        signature = dilithium_signer.sign(test_message, credentials.private_key)
        
        # Authenticate
        result = auth_service.authenticate_client(client_id, signature, test_message)
        
        assert result is True
    
    def test_authenticate_nonexistent_client(self, auth_service):
        """Test authentication with non-existent client."""
        result = auth_service.authenticate_client(
            "nonexistent_client", 
            b"fake_signature", 
            b"test_message"
        )
        
        assert result is False
    
    def test_authenticate_invalid_signature(self, auth_service, dilithium_signer):
        """Test authentication with invalid signature."""
        client_id = "test_client_004"
        
        # Register client
        credentials = auth_service.register_client(client_id)
        
        # Create message but use wrong signature
        test_message = b"test_authentication_message"
        wrong_signature = b"invalid_signature_data"
        
        # Authenticate with wrong signature
        result = auth_service.authenticate_client(client_id, wrong_signature, test_message)
        
        assert result is False
    
    def test_authenticate_expired_credentials(self, auth_service, dilithium_signer):
        """Test authentication with expired credentials."""
        client_id = "test_client_005"
        
        # Register client
        credentials = auth_service.register_client(client_id)
        
        # Manually expire credentials
        import sqlite3
        past_date = datetime.now() - timedelta(days=1)
        with sqlite3.connect(auth_service.credential_manager.db_path) as conn:
            conn.execute("""
                UPDATE client_credentials 
                SET expires_at = ? 
                WHERE client_id = ?
            """, (past_date.isoformat(), client_id))
            conn.commit()
        
        # Try to authenticate
        test_message = b"test_message"
        signature = dilithium_signer.sign(test_message, credentials.private_key)
        result = auth_service.authenticate_client(client_id, signature, test_message)
        
        assert result is False
    
    def test_authenticate_revoked_client(self, auth_service, dilithium_signer):
        """Test authentication with revoked client."""
        client_id = "test_client_006"
        
        # Register client
        credentials = auth_service.register_client(client_id)
        
        # Revoke client
        auth_service.revoke_client(client_id)
        
        # Try to authenticate
        test_message = b"test_message"
        signature = dilithium_signer.sign(test_message, credentials.private_key)
        result = auth_service.authenticate_client(client_id, signature, test_message)
        
        assert result is False
        assert auth_service.is_client_valid(client_id) is False
    
    def test_failed_attempts_lockout(self, auth_service):
        """Test client lockout after multiple failed attempts."""
        client_id = "test_client_007"
        
        # Register client
        auth_service.register_client(client_id)
        
        # Make multiple failed authentication attempts
        for i in range(3):  # max_failed_attempts = 3
            result = auth_service.authenticate_client(
                client_id, 
                b"invalid_signature", 
                b"test_message"
            )
            assert result is False
        
        # Client should now be locked
        locked_clients = auth_service.get_locked_clients()
        assert client_id in locked_clients
        
        # Further authentication attempts should fail due to lockout
        result = auth_service.authenticate_client(
            client_id, 
            b"any_signature", 
            b"test_message"
        )
        assert result is False
    
    def test_lockout_expiry(self, auth_service):
        """Test that lockout expires after the specified duration."""
        client_id = "test_client_008"
        
        # Register client
        auth_service.register_client(client_id)
        
        # Trigger lockout
        for i in range(3):
            auth_service.authenticate_client(client_id, b"invalid", b"message")
        
        # Verify client is locked
        assert client_id in auth_service.get_locked_clients()
        
        # Manually set lockout time to past (simulate time passage)
        past_time = datetime.now() - timedelta(minutes=10)  # lockout_duration = 5 minutes
        auth_service._locked_clients[client_id] = past_time
        
        # Check if client is still locked (should not be)
        assert auth_service._is_client_locked(client_id) is False
        assert client_id not in auth_service.get_locked_clients()
    
    def test_successful_auth_clears_failed_attempts(self, auth_service, dilithium_signer):
        """Test that successful authentication clears failed attempts."""
        client_id = "test_client_009"
        
        # Register client
        credentials = auth_service.register_client(client_id)
        
        # Make some failed attempts
        for i in range(2):
            auth_service.authenticate_client(client_id, b"invalid", b"message")
        
        # Verify failed attempts are recorded
        assert client_id in auth_service._failed_attempts
        assert len(auth_service._failed_attempts[client_id]) == 2
        
        # Make successful authentication
        test_message = b"test_message"
        signature = dilithium_signer.sign(test_message, credentials.private_key)
        result = auth_service.authenticate_client(client_id, signature, test_message)
        
        assert result is True
        # Failed attempts should be cleared
        assert client_id not in auth_service._failed_attempts
    
    def test_manual_unlock_client(self, auth_service):
        """Test manual client unlock functionality."""
        client_id = "test_client_010"
        
        # Register client and trigger lockout
        auth_service.register_client(client_id)
        for i in range(3):
            auth_service.authenticate_client(client_id, b"invalid", b"message")
        
        # Verify client is locked
        assert client_id in auth_service.get_locked_clients()
        
        # Manually unlock
        result = auth_service.unlock_client(client_id)
        
        assert result is True
        assert client_id not in auth_service.get_locked_clients()
        assert auth_service.is_client_valid(client_id) is True
    
    def test_unlock_non_locked_client(self, auth_service):
        """Test unlocking a client that isn't locked."""
        client_id = "test_client_011"
        
        # Register client (but don't lock)
        auth_service.register_client(client_id)
        
        # Try to unlock
        result = auth_service.unlock_client(client_id)
        
        assert result is False
    
    def test_get_client_credentials(self, auth_service):
        """Test retrieving client credentials."""
        client_id = "test_client_012"
        
        # Register client
        original_credentials = auth_service.register_client(client_id)
        
        # Retrieve credentials
        retrieved_credentials = auth_service.get_client_credentials(client_id)
        
        assert retrieved_credentials is not None
        assert retrieved_credentials.client_id == original_credentials.client_id
        assert retrieved_credentials.public_key == original_credentials.public_key
    
    def test_get_credentials_invalid_client(self, auth_service):
        """Test retrieving credentials for invalid client."""
        # Non-existent client
        result = auth_service.get_client_credentials("nonexistent")
        assert result is None
        
        # Locked client
        client_id = "locked_client"
        auth_service.register_client(client_id)
        for i in range(3):
            auth_service.authenticate_client(client_id, b"invalid", b"message")
        
        result = auth_service.get_client_credentials(client_id)
        assert result is None
    
    def test_authentication_stats(self, auth_service, dilithium_signer):
        """Test authentication statistics collection."""
        client_id = "test_client_013"
        
        # Register client
        credentials = auth_service.register_client(client_id)
        
        # Make some authentication attempts
        test_message = b"test_message"
        signature = dilithium_signer.sign(test_message, credentials.private_key)
        
        # Successful attempts
        auth_service.authenticate_client(client_id, signature, test_message)
        auth_service.authenticate_client(client_id, signature, test_message)
        
        # Failed attempts
        auth_service.authenticate_client(client_id, b"invalid", test_message)
        
        # Get stats
        stats = auth_service.get_authentication_stats()
        
        assert stats['total_attempts'] >= 4  # Including registration
        assert stats['successful_attempts'] >= 3  # Including registration
        assert stats['failed_attempts'] >= 1
        assert 0 <= stats['success_rate'] <= 1
    
    def test_client_authentication_history(self, auth_service, dilithium_signer):
        """Test retrieving client authentication history."""
        client_id = "test_client_014"
        
        # Register client
        credentials = auth_service.register_client(client_id)
        
        # Make some authentication attempts
        test_message = b"test_message"
        signature = dilithium_signer.sign(test_message, credentials.private_key)
        
        auth_service.authenticate_client(client_id, signature, test_message)
        auth_service.authenticate_client(client_id, b"invalid", test_message)
        
        # Get history
        history = auth_service.get_client_authentication_history(client_id, limit=5)
        
        assert len(history) >= 3  # Including registration
        assert all(attempt.client_id == client_id for attempt in history)
        
        # Should be sorted by timestamp (most recent first)
        timestamps = [attempt.timestamp for attempt in history]
        assert timestamps == sorted(timestamps, reverse=True)
    
    def test_cleanup_old_logs(self, auth_service):
        """Test cleanup of old authentication logs."""
        client_id = "test_client_015"
        
        # Register client to generate some log entries
        auth_service.register_client(client_id)
        
        # Manually add old log entries
        old_attempt = AuthenticationAttempt(
            client_id=client_id,
            timestamp=datetime.now() - timedelta(days=35),
            success=True
        )
        auth_service._authentication_log.append(old_attempt)
        
        original_count = len(auth_service._authentication_log)
        
        # Cleanup logs older than 30 days
        removed_count = auth_service.cleanup_old_logs(days_to_keep=30)
        
        assert removed_count >= 1
        assert len(auth_service._authentication_log) < original_count
    
    def test_error_handling_in_authentication(self, auth_service):
        """Test error handling during authentication."""
        client_id = "test_client_016"
        
        # Register client
        credentials = auth_service.register_client(client_id)
        
        # Test with invalid signature format
        result = auth_service.authenticate_client(client_id, None, b"message")
        assert result is False
        
        # Test with invalid message format
        result = auth_service.authenticate_client(client_id, b"signature", None)
        assert result is False
    
    @patch('auth.authentication_service.DilithiumSigner')
    def test_signature_verification_error_handling(self, mock_dilithium_class, auth_service):
        """Test handling of signature verification errors."""
        # Setup mock to raise exception
        mock_signer = MagicMock()
        mock_signer.verify.side_effect = Exception("Signature verification error")
        mock_dilithium_class.return_value = mock_signer
        
        # Create new service with mocked signer
        service = AuthenticationService(
            credential_manager=auth_service.credential_manager,
            max_failed_attempts=3,
            lockout_duration_minutes=5
        )
        
        client_id = "test_client_017"
        
        # Register client (this will use the original signer)
        auth_service.register_client(client_id)
        
        # Try authentication (this will use the mocked signer)
        result = service.authenticate_client(client_id, b"signature", b"message")
        
        assert result is False
        
        # Should record failed attempt
        assert client_id in service._failed_attempts
    
    def test_concurrent_authentication_attempts(self, auth_service, dilithium_signer):
        """Test handling of concurrent authentication attempts."""
        client_id = "test_client_018"
        
        # Register client
        credentials = auth_service.register_client(client_id)
        
        # Simulate concurrent authentication attempts
        test_message = b"test_message"
        signature = dilithium_signer.sign(test_message, credentials.private_key)
        
        results = []
        for i in range(5):
            result = auth_service.authenticate_client(client_id, signature, test_message)
            results.append(result)
        
        # All should succeed
        assert all(results)
    
    def test_authentication_with_different_messages(self, auth_service, dilithium_signer):
        """Test authentication with different messages using same credentials."""
        client_id = "test_client_019"
        
        # Register client
        credentials = auth_service.register_client(client_id)
        
        # Test with different messages
        messages = [b"message1", b"message2", b"message3"]
        
        for message in messages:
            signature = dilithium_signer.sign(message, credentials.private_key)
            result = auth_service.authenticate_client(client_id, signature, message)
            assert result is True
    
    def test_revoke_client_clears_lockout(self, auth_service):
        """Test that revoking a client clears any existing lockout."""
        client_id = "test_client_020"
        
        # Register client and trigger lockout
        auth_service.register_client(client_id)
        for i in range(3):
            auth_service.authenticate_client(client_id, b"invalid", b"message")
        
        # Verify client is locked
        assert client_id in auth_service.get_locked_clients()
        
        # Revoke client
        auth_service.revoke_client(client_id)
        
        # Lockout should be cleared
        assert client_id not in auth_service.get_locked_clients()
        assert client_id not in auth_service._failed_attempts
    
    def test_revoked_client_authentication_blocked(self, auth_service, dilithium_signer):
        """Test that revoked clients cannot authenticate."""
        client_id = "test_client_021"
        
        # Register client
        credentials = auth_service.register_client(client_id)
        
        # Verify normal authentication works
        message = b"test_message"
        signature = dilithium_signer.sign(message, credentials.private_key)
        assert auth_service.authenticate_client(client_id, signature, message) is True
        
        # Revoke client
        auth_service.revoke_client(client_id, "security_breach")
        
        # Authentication should now fail
        assert auth_service.authenticate_client(client_id, signature, message) is False
        
        # Client should not be valid
        assert auth_service.is_client_valid(client_id) is False
        
        # Should be in revocation list
        assert auth_service.revocation_manager.is_revoked(client_id) is True
    
    def test_revoked_client_attempt_records_suspicious_behavior(self, auth_service, dilithium_signer):
        """Test that attempts by revoked clients are recorded as suspicious behavior."""
        client_id = "test_client_022"
        
        # Register and revoke client
        credentials = auth_service.register_client(client_id)
        auth_service.revoke_client(client_id, "manual")
        
        # Attempt authentication
        message = b"test_message"
        signature = dilithium_signer.sign(message, credentials.private_key)
        auth_service.authenticate_client(client_id, signature, message)
        
        # Should record suspicious behavior
        events = auth_service.revocation_manager.get_suspicious_events(client_id)
        assert len(events) > 0
        assert any(event.event_type == "revoked_client_attempt" for event in events)
    
    def test_invalid_signature_records_suspicious_behavior(self, auth_service):
        """Test that invalid signatures are recorded as suspicious behavior."""
        client_id = "test_client_023"
        
        # Register client
        auth_service.register_client(client_id)
        
        # Attempt authentication with invalid signature
        auth_service.authenticate_client(client_id, b"invalid_signature", b"message")
        
        # Should record suspicious behavior
        events = auth_service.revocation_manager.get_suspicious_events(client_id)
        assert len(events) > 0
        assert any(event.event_type == "invalid_signature" for event in events)
    
    def test_repeated_failures_records_suspicious_behavior(self, auth_service):
        """Test that repeated failures trigger suspicious behavior recording."""
        client_id = "test_client_024"
        
        # Register client
        auth_service.register_client(client_id)
        
        # Trigger lockout with repeated failures
        for i in range(3):
            auth_service.authenticate_client(client_id, b"invalid", b"message")
        
        # Should record suspicious behavior for repeated failures
        events = auth_service.revocation_manager.get_suspicious_events(client_id)
        assert len(events) > 0
        assert any(event.event_type == "repeated_failures" for event in events)
    
    def test_authentication_without_revocation_manager(self, auth_service_no_revocation, dilithium_signer):
        """Test that authentication works without revocation manager."""
        client_id = "test_client_025"
        
        # Register client
        credentials = auth_service_no_revocation.register_client(client_id)
        
        # Authentication should work normally
        message = b"test_message"
        signature = dilithium_signer.sign(message, credentials.private_key)
        assert auth_service_no_revocation.authenticate_client(client_id, signature, message) is True
        
        # Revocation should still work (just won't use revocation manager)
        auth_service_no_revocation.revoke_client(client_id)
        assert auth_service_no_revocation.is_client_valid(client_id) is False
    
    def test_revocation_manager_integration_with_automatic_revocation(self, auth_service):
        """Test integration with automatic revocation based on behavior score."""
        client_id = "test_client_026"
        
        # Register client
        auth_service.register_client(client_id)
        
        # Trigger enough suspicious behavior to cause automatic revocation
        # Using malicious_payload events with high severity
        for i in range(3):
            auth_service.revocation_manager.record_suspicious_behavior(
                client_id=client_id,
                event_type="malicious_payload",
                severity=10,
                source="test"
            )
        
        # Client should now be automatically revoked
        assert auth_service.revocation_manager.is_revoked(client_id) is True
        assert auth_service.is_client_valid(client_id) is False


if __name__ == "__main__":
    pytest.main([__file__])