"""
Unit tests for ClientCredentialManager

Tests credential issuance, storage, renewal, expiration, and edge cases.
"""

import pytest
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from auth.credential_manager import ClientCredentialManager
from auth.interfaces import CredentialStatus, ClientCredentials


class TestClientCredentialManager:
    """Test suite for ClientCredentialManager."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        yield path
        # On Windows, we need to be more careful about file cleanup
        try:
            if os.path.exists(path):
                os.unlink(path)
        except PermissionError:
            # File is still in use, skip cleanup
            pass
    
    @pytest.fixture
    def credential_manager(self, temp_db):
        """Create credential manager with temporary database."""
        return ClientCredentialManager(
            db_path=temp_db,
            credential_validity_days=30,
            security_level=3
        )
    
    def test_init_creates_database(self, temp_db):
        """Test that initialization creates database tables."""
        manager = ClientCredentialManager(db_path=temp_db)
        
        # Check that database file exists
        assert os.path.exists(temp_db)
        
        # Check that table was created
        import sqlite3
        with sqlite3.connect(temp_db) as conn:
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='client_credentials'
            """)
            assert cursor.fetchone() is not None
    
    def test_issue_credentials_success(self, credential_manager):
        """Test successful credential issuance."""
        client_id = "test_client_001"
        
        credentials = credential_manager.issue_credentials(client_id)
        
        assert credentials.client_id == client_id
        assert credentials.status == CredentialStatus.ACTIVE
        assert isinstance(credentials.public_key, bytes)
        assert isinstance(credentials.private_key, bytes)
        assert len(credentials.public_key) > 0
        assert len(credentials.private_key) > 0
        assert credentials.expires_at > credentials.issued_at
        assert credentials.expires_at > datetime.now()
    
    def test_issue_credentials_with_metadata(self, credential_manager):
        """Test credential issuance with metadata."""
        client_id = "test_client_002"
        metadata = {"organization": "test_org", "role": "participant"}
        
        credentials = credential_manager.issue_credentials(client_id, metadata)
        
        assert credentials.client_id == client_id
        assert credentials.status == CredentialStatus.ACTIVE
    
    def test_issue_credentials_invalid_client_id(self, credential_manager):
        """Test credential issuance with invalid client ID."""
        with pytest.raises(ValueError, match="Client ID must be a non-empty string"):
            credential_manager.issue_credentials("")
        
        with pytest.raises(ValueError, match="Client ID must be a non-empty string"):
            credential_manager.issue_credentials(None)
    
    def test_issue_credentials_duplicate_active(self, credential_manager):
        """Test that issuing credentials for existing active client raises error."""
        client_id = "test_client_003"
        
        # Issue first credentials
        credential_manager.issue_credentials(client_id)
        
        # Try to issue again - should raise error
        with pytest.raises(ValueError, match="already has active credentials"):
            credential_manager.issue_credentials(client_id)
    
    def test_store_and_load_credentials(self, credential_manager):
        """Test storing and loading credentials."""
        client_id = "test_client_004"
        
        # Issue and store credentials
        original_credentials = credential_manager.issue_credentials(client_id)
        
        # Load credentials
        loaded_credentials = credential_manager.load_credentials(client_id)
        
        assert loaded_credentials is not None
        assert loaded_credentials.client_id == original_credentials.client_id
        assert loaded_credentials.public_key == original_credentials.public_key
        assert loaded_credentials.private_key == original_credentials.private_key
        assert loaded_credentials.status == original_credentials.status
        assert loaded_credentials.issued_at == original_credentials.issued_at
        assert loaded_credentials.expires_at == original_credentials.expires_at
    
    def test_load_nonexistent_credentials(self, credential_manager):
        """Test loading credentials for non-existent client."""
        result = credential_manager.load_credentials("nonexistent_client")
        assert result is None
    
    def test_renew_credentials_success(self, credential_manager):
        """Test successful credential renewal."""
        client_id = "test_client_005"
        
        # Issue initial credentials
        original_credentials = credential_manager.issue_credentials(client_id)
        
        # Renew credentials
        renewed_credentials = credential_manager.renew_credentials(client_id)
        
        assert renewed_credentials.client_id == client_id
        assert renewed_credentials.status == CredentialStatus.ACTIVE
        assert renewed_credentials.public_key != original_credentials.public_key
        assert renewed_credentials.private_key != original_credentials.private_key
        assert renewed_credentials.issued_at > original_credentials.issued_at
    
    def test_renew_nonexistent_credentials(self, credential_manager):
        """Test renewing credentials for non-existent client."""
        with pytest.raises(ValueError, match="No credentials found"):
            credential_manager.renew_credentials("nonexistent_client")
    
    def test_renew_revoked_credentials(self, credential_manager):
        """Test that revoked credentials cannot be renewed."""
        client_id = "test_client_006"
        
        # Issue credentials
        credential_manager.issue_credentials(client_id)
        
        # Revoke credentials
        credential_manager._update_credential_status(client_id, CredentialStatus.REVOKED)
        
        # Try to renew - should fail
        with pytest.raises(ValueError, match="Cannot renew revoked credentials"):
            credential_manager.renew_credentials(client_id)
    
    def test_expire_credentials(self, credential_manager):
        """Test expiring credentials."""
        client_id = "test_client_007"
        
        # Issue credentials
        credential_manager.issue_credentials(client_id)
        
        # Expire credentials
        credential_manager.expire_credentials(client_id)
        
        # Check status
        credentials = credential_manager.load_credentials(client_id)
        assert credentials.status == CredentialStatus.EXPIRED
    
    def test_suspend_and_reactivate_credentials(self, credential_manager):
        """Test suspending and reactivating credentials."""
        client_id = "test_client_008"
        
        # Issue credentials
        credential_manager.issue_credentials(client_id)
        
        # Suspend credentials
        credential_manager.suspend_credentials(client_id)
        credentials = credential_manager.load_credentials(client_id)
        assert credentials.status == CredentialStatus.SUSPENDED
        
        # Reactivate credentials
        credential_manager.reactivate_credentials(client_id)
        credentials = credential_manager.load_credentials(client_id)
        assert credentials.status == CredentialStatus.ACTIVE
    
    def test_reactivate_revoked_credentials(self, credential_manager):
        """Test that revoked credentials cannot be reactivated."""
        client_id = "test_client_009"
        
        # Issue and revoke credentials
        credential_manager.issue_credentials(client_id)
        credential_manager._update_credential_status(client_id, CredentialStatus.REVOKED)
        
        # Try to reactivate - should fail
        with pytest.raises(ValueError, match="Cannot reactivate revoked credentials"):
            credential_manager.reactivate_credentials(client_id)
    
    def test_reactivate_expired_credentials(self, credential_manager):
        """Test that expired credentials cannot be reactivated."""
        client_id = "test_client_010"
        
        # Issue credentials with past expiry date
        credentials = credential_manager.issue_credentials(client_id)
        
        # Manually set expiry to past date
        past_date = datetime.now() - timedelta(days=1)
        import sqlite3
        with sqlite3.connect(credential_manager.db_path) as conn:
            conn.execute("""
                UPDATE client_credentials 
                SET expires_at = ? 
                WHERE client_id = ?
            """, (past_date.isoformat(), client_id))
            conn.commit()
        
        # Try to reactivate - should fail
        with pytest.raises(ValueError, match="Cannot reactivate expired credentials"):
            credential_manager.reactivate_credentials(client_id)
    
    def test_cleanup_expired_credentials(self, credential_manager):
        """Test cleanup of expired credentials."""
        # Issue credentials for multiple clients
        client_ids = ["client_001", "client_002", "client_003"]
        for client_id in client_ids:
            credential_manager.issue_credentials(client_id)
        
        # Expire some credentials
        credential_manager.expire_credentials("client_001")
        credential_manager.expire_credentials("client_002")
        
        # Manually set expiry to past date for expired credentials
        past_date = datetime.now() - timedelta(days=1)
        import sqlite3
        with sqlite3.connect(credential_manager.db_path) as conn:
            conn.execute("""
                UPDATE client_credentials 
                SET expires_at = ? 
                WHERE client_id IN (?, ?)
            """, (past_date.isoformat(), "client_001", "client_002"))
            conn.commit()
        
        # Cleanup expired credentials
        deleted_count = credential_manager.cleanup_expired_credentials()
        
        assert deleted_count == 2
        
        # Verify only active client remains
        remaining_clients = credential_manager.get_all_active_clients()
        assert len(remaining_clients) == 1
        assert "client_003" in remaining_clients
    
    def test_get_all_active_clients(self, credential_manager):
        """Test getting all active clients."""
        # Issue credentials for multiple clients
        active_clients = ["client_001", "client_002", "client_003"]
        for client_id in active_clients:
            credential_manager.issue_credentials(client_id)
        
        # Suspend one client
        credential_manager.suspend_credentials("client_002")
        
        # Get active clients
        result = credential_manager.get_all_active_clients()
        
        assert len(result) == 2
        assert "client_001" in result
        assert "client_003" in result
        assert "client_002" not in result
    
    def test_get_credential_info(self, credential_manager):
        """Test getting credential information."""
        client_id = "test_client_011"
        
        # Issue credentials
        credential_manager.issue_credentials(client_id)
        
        # Get credential info
        info = credential_manager.get_credential_info(client_id)
        
        assert info is not None
        assert info['client_id'] == client_id
        assert info['status'] == CredentialStatus.ACTIVE.value
        assert 'issued_at' in info
        assert 'expires_at' in info
        assert 'days_until_expiry' in info
        assert 'is_expired' in info
        assert info['is_expired'] is False
        
        # Verify private key is not included
        assert 'private_key' not in info
    
    def test_get_credential_info_nonexistent(self, credential_manager):
        """Test getting credential info for non-existent client."""
        info = credential_manager.get_credential_info("nonexistent_client")
        assert info is None
    
    def test_validate_keypair_success(self, credential_manager):
        """Test successful keypair validation."""
        client_id = "test_client_012"
        
        # Issue credentials
        credential_manager.issue_credentials(client_id)
        
        # Validate keypair
        is_valid = credential_manager.validate_keypair(client_id)
        assert is_valid is True
    
    def test_validate_keypair_nonexistent(self, credential_manager):
        """Test keypair validation for non-existent client."""
        is_valid = credential_manager.validate_keypair("nonexistent_client")
        assert is_valid is False
    
    def test_validate_keypair_corrupted(self, credential_manager):
        """Test keypair validation with corrupted keys."""
        client_id = "test_client_013"
        
        # Issue credentials
        credential_manager.issue_credentials(client_id)
        
        # Corrupt the private key in database
        import sqlite3
        with sqlite3.connect(credential_manager.db_path) as conn:
            conn.execute("""
                UPDATE client_credentials 
                SET private_key = ? 
                WHERE client_id = ?
            """, (b"corrupted_key", client_id))
            conn.commit()
        
        # Validate keypair - should fail
        is_valid = credential_manager.validate_keypair(client_id)
        assert is_valid is False
    
    def test_get_statistics(self, credential_manager):
        """Test getting credential statistics."""
        # Issue credentials for multiple clients with different statuses
        credential_manager.issue_credentials("active_client_1")
        credential_manager.issue_credentials("active_client_2")
        credential_manager.issue_credentials("suspended_client")
        credential_manager.issue_credentials("expired_client")
        
        # Change statuses
        credential_manager.suspend_credentials("suspended_client")
        credential_manager.expire_credentials("expired_client")
        
        # Get statistics
        stats = credential_manager.get_statistics()
        
        assert stats[CredentialStatus.ACTIVE.value] == 2
        assert stats[CredentialStatus.SUSPENDED.value] == 1
        assert stats[CredentialStatus.EXPIRED.value] == 1
        assert stats[CredentialStatus.REVOKED.value] == 0
        assert stats['total'] == 4
    
    def test_credential_expiry_calculation(self, credential_manager):
        """Test that credential expiry is calculated correctly."""
        client_id = "test_client_014"
        
        before_issue = datetime.now()
        credentials = credential_manager.issue_credentials(client_id)
        after_issue = datetime.now()
        
        # Check that expiry is approximately 30 days from now
        expected_expiry = before_issue + timedelta(days=30)
        assert credentials.expires_at >= expected_expiry
        
        expected_expiry = after_issue + timedelta(days=30)
        assert credentials.expires_at <= expected_expiry
    
    @patch('auth.credential_manager.DilithiumSigner')
    def test_dilithium_integration(self, mock_dilithium_class, credential_manager):
        """Test integration with Dilithium signer."""
        mock_signer = MagicMock()
        mock_dilithium_class.return_value = mock_signer
        
        # Mock keypair generation
        mock_public_key = b"mock_public_key"
        mock_private_key = b"mock_private_key"
        mock_signer.generate_keypair.return_value = (mock_public_key, mock_private_key)
        
        # Create new manager to use mocked signer
        manager = ClientCredentialManager(
            db_path=credential_manager.db_path,
            security_level=3
        )
        
        # Issue credentials
        credentials = manager.issue_credentials("test_client")
        
        # Verify Dilithium signer was called
        mock_dilithium_class.assert_called_with(security_level=3)
        mock_signer.generate_keypair.assert_called_once()
        
        assert credentials.public_key == mock_public_key
        assert credentials.private_key == mock_private_key


if __name__ == "__main__":
    pytest.main([__file__])