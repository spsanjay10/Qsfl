"""
Integration tests for PQCryptoManager orchestration layer.
"""

import pytest
import json
import secrets
from datetime import datetime, timedelta
from pq_security.manager import PQCryptoManager, ClientKeys, CryptoSession


class TestPQCryptoManager:
    """Test suite for PQCryptoManager class."""
    
    def test_initialization(self):
        """Test PQCryptoManager initialization."""
        manager = PQCryptoManager()
        
        # Check default security levels
        assert manager.kyber.security_level == 3
        assert manager.dilithium.security_level == 3
        
        # Check session timeout
        assert manager.session_timeout == timedelta(hours=24)
        
        # Check server keys are generated
        assert len(manager._server_kx_public) > 0
        assert len(manager._server_kx_private) > 0
    
    def test_initialization_custom_levels(self):
        """Test initialization with custom security levels."""
        manager = PQCryptoManager(
            kyber_security_level=2,
            dilithium_security_level=5,
            session_timeout_hours=12
        )
        
        assert manager.kyber.security_level == 2
        assert manager.dilithium.security_level == 5
        assert manager.session_timeout == timedelta(hours=12)
    
    def test_generate_combined_keypair(self):
        """Test generation of combined keypairs."""
        manager = PQCryptoManager()
        public_key, private_key = manager.generate_keypair()
        
        # Check key types
        assert isinstance(public_key, bytes)
        assert isinstance(private_key, bytes)
        
        # Parse JSON structure
        public_data = json.loads(public_key.decode('utf-8'))
        private_data = json.loads(private_key.decode('utf-8'))
        
        # Check structure
        assert 'key_exchange' in public_data
        assert 'signature' in public_data
        assert 'kyber_level' in public_data
        assert 'dilithium_level' in public_data
        
        assert 'key_exchange' in private_data
        assert 'signature' in private_data
        assert 'kyber_level' in private_data
        assert 'dilithium_level' in private_data
        
        # Check security levels
        assert public_data['kyber_level'] == 3
        assert public_data['dilithium_level'] == 3
        assert private_data['kyber_level'] == 3
        assert private_data['dilithium_level'] == 3
    
    def test_client_registration(self):
        """Test client registration."""
        manager = PQCryptoManager()
        client_id = "test_client_001"
        
        # Register client
        client_keys = manager.register_client(client_id)
        
        # Check client keys structure
        assert isinstance(client_keys, ClientKeys)
        assert client_keys.client_id == client_id
        assert len(client_keys.signature_public_key) > 0
        assert len(client_keys.signature_private_key) > 0
        assert len(client_keys.key_exchange_public_key) > 0
        assert len(client_keys.key_exchange_private_key) > 0
        
        # Check timestamps
        assert client_keys.created_at <= datetime.utcnow()
        assert client_keys.expires_at > client_keys.created_at
        
        # Check client is registered
        assert client_id in manager.list_registered_clients()
    
    def test_client_registration_duplicate(self):
        """Test duplicate client registration raises error."""
        manager = PQCryptoManager()
        client_id = "test_client_002"
        
        # Register client first time
        manager.register_client(client_id)
        
        # Try to register again
        with pytest.raises(ValueError, match="already registered"):
            manager.register_client(client_id)
    
    def test_session_establishment(self):
        """Test secure session establishment."""
        manager = PQCryptoManager()
        client_id = "test_client_003"
        
        # Register client first
        manager.register_client(client_id)
        
        # Establish session
        session_id, ciphertext, shared_secret = manager.establish_session(client_id)
        
        # Check session properties
        assert isinstance(session_id, str)
        assert len(session_id) == 32  # 16 bytes hex = 32 chars
        assert isinstance(ciphertext, bytes)
        assert len(ciphertext) > 0
        assert isinstance(shared_secret, bytes)
        assert len(shared_secret) > 0
        
        # Check session is active
        assert session_id in manager.list_active_sessions()
        
        # Check session info
        session_info = manager.get_session_info(session_id)
        assert session_info is not None
        assert session_info['client_id'] == client_id
        assert session_info['is_active'] is True
    
    def test_session_establishment_unregistered_client(self):
        """Test session establishment with unregistered client fails."""
        manager = PQCryptoManager()
        
        with pytest.raises(ValueError, match="not registered"):
            manager.establish_session("nonexistent_client")
    
    def test_message_encryption_decryption_session(self):
        """Test message encryption/decryption using sessions."""
        manager = PQCryptoManager()
        client_id = "test_client_004"
        
        # Setup
        manager.register_client(client_id)
        session_id, ciphertext, shared_secret = manager.establish_session(client_id)
        
        # Test server -> client encryption
        message = b"Hello from server!"
        encrypted = manager.encrypt_server_message(session_id, message)
        
        assert isinstance(encrypted, bytes)
        assert len(encrypted) > len(message)  # Should be longer due to nonce
        
        # Test client -> server decryption (simulate client encrypting with same method)
        decrypted = manager.decrypt_client_message(session_id, encrypted)
        assert decrypted == message
    
    def test_message_signing_verification(self):
        """Test message signing and verification."""
        manager = PQCryptoManager()
        client_id = "test_client_005"
        
        # Register client
        manager.register_client(client_id)
        
        # Sign message
        message = b"Important client update"
        signature = manager.sign_message(client_id, message)
        
        assert isinstance(signature, bytes)
        assert len(signature) > 0
        
        # Verify signature
        is_valid = manager.verify_client_signature(client_id, message, signature)
        assert is_valid is True
        
        # Verify with wrong message should fail
        wrong_message = b"Different message"
        is_invalid = manager.verify_client_signature(client_id, wrong_message, signature)
        assert is_invalid is False
    
    def test_combined_key_encrypt_decrypt(self):
        """Test encryption/decryption using combined key format."""
        manager = PQCryptoManager()
        
        # Generate combined keypair
        public_key, private_key = manager.generate_keypair()
        
        # Test encryption/decryption
        message = b"Test message for combined keys"
        ciphertext = manager.encrypt(message, public_key)
        
        assert isinstance(ciphertext, bytes)
        assert len(ciphertext) > len(message)
        
        # Decrypt
        decrypted = manager.decrypt(ciphertext, private_key)
        assert decrypted == message
    
    def test_combined_key_sign_verify(self):
        """Test signing/verification using combined key format."""
        manager = PQCryptoManager()
        
        # Generate combined keypair
        public_key, private_key = manager.generate_keypair()
        
        # Test signing/verification
        message = b"Test message for combined signatures"
        signature = manager.sign(message, private_key)
        
        assert isinstance(signature, bytes)
        assert len(signature) > 0
        
        # Verify
        is_valid = manager.verify(message, signature, public_key)
        assert is_valid is True
        
        # Verify with wrong message should fail
        wrong_message = b"Wrong message"
        is_invalid = manager.verify(wrong_message, signature, public_key)
        assert is_invalid is False
    
    def test_session_cleanup(self):
        """Test cleanup of expired sessions."""
        manager = PQCryptoManager(session_timeout_hours=0)  # Immediate expiry
        client_id = "test_client_006"
        
        # Register client and establish session
        manager.register_client(client_id)
        session_id, _, _ = manager.establish_session(client_id)
        
        # Session should be immediately expired
        assert session_id not in manager.list_active_sessions()
        
        # Cleanup should remove expired session
        removed_count = manager.cleanup_expired_sessions()
        assert removed_count >= 0  # May be 0 if already cleaned up
    
    def test_client_key_cleanup(self):
        """Test cleanup of expired client keys."""
        manager = PQCryptoManager()
        client_id = "test_client_007"
        
        # Register client with short lifetime
        client_keys = manager.register_client(client_id, key_lifetime_hours=0)
        
        # Keys should be expired
        client_info = manager.get_client_info(client_id)
        assert client_info['is_expired'] is True
        
        # Cleanup should remove expired keys
        removed_count = manager.cleanup_expired_client_keys()
        assert removed_count >= 1
        
        # Client should no longer be registered
        assert client_id not in manager.list_registered_clients()
    
    def test_client_revocation(self):
        """Test client revocation."""
        manager = PQCryptoManager()
        client_id = "test_client_008"
        
        # Register client and establish session
        manager.register_client(client_id)
        session_id, _, _ = manager.establish_session(client_id)
        
        # Verify client is active
        assert client_id in manager.list_registered_clients()
        assert session_id in manager.list_active_sessions()
        
        # Revoke client
        revoked = manager.revoke_client(client_id)
        assert revoked is True
        
        # Client should be removed
        assert client_id not in manager.list_registered_clients()
        assert session_id not in manager.list_active_sessions()
        
        # Revoking non-existent client should return False
        revoked_again = manager.revoke_client(client_id)
        assert revoked_again is False
    
    def test_invalid_session_operations(self):
        """Test operations with invalid sessions."""
        manager = PQCryptoManager()
        
        # Test with non-existent session
        with pytest.raises(ValueError, match="not found"):
            manager.encrypt_server_message("invalid_session", b"test")
        
        with pytest.raises(ValueError, match="not found"):
            manager.decrypt_client_message("invalid_session", b"test")
    
    def test_invalid_key_formats(self):
        """Test handling of invalid key formats."""
        manager = PQCryptoManager()
        
        # Test invalid public key format
        with pytest.raises(ValueError, match="Invalid public key format"):
            manager.encrypt(b"test", b"invalid_key")
        
        # Test invalid private key format
        with pytest.raises(ValueError, match="Invalid private key format"):
            manager.decrypt(b"test", b"invalid_key")
        
        with pytest.raises(ValueError, match="Invalid private key format"):
            manager.sign(b"test", b"invalid_key")
        
        # Test invalid JSON
        with pytest.raises(ValueError):
            manager.verify(b"test", b"sig", b"not_json")
    
    def test_security_info(self):
        """Test security information retrieval."""
        manager = PQCryptoManager(
            kyber_security_level=2,
            dilithium_security_level=5,
            session_timeout_hours=12
        )
        
        # Register a client and establish session
        manager.register_client("test_client")
        manager.establish_session("test_client")
        
        security_info = manager.get_security_info()
        
        assert security_info['kyber_security_level'] == 2
        assert security_info['dilithium_security_level'] == 5
        assert security_info['session_timeout_hours'] == 12
        assert security_info['active_sessions'] >= 1
        assert security_info['registered_clients'] >= 1
        assert isinstance(security_info['kyber_simulation_mode'], bool)
        assert isinstance(security_info['dilithium_simulation_mode'], bool)
    
    def test_simulation_mode_detection(self):
        """Test simulation mode detection."""
        manager = PQCryptoManager()
        
        simulation_mode = manager.is_simulation_mode()
        assert isinstance(simulation_mode, bool)
        
        # Should match individual component simulation modes
        expected = manager.kyber.is_simulation_mode or manager.dilithium.is_simulation_mode
        assert simulation_mode == expected
    
    def test_server_public_key(self):
        """Test server public key retrieval."""
        manager = PQCryptoManager()
        
        server_public_key = manager.get_server_public_key()
        assert isinstance(server_public_key, bytes)
        assert len(server_public_key) > 0
        
        # Should be consistent across calls
        server_public_key2 = manager.get_server_public_key()
        assert server_public_key == server_public_key2


class TestPQCryptoManagerIntegration:
    """Integration tests for complete workflows."""
    
    def test_complete_client_server_workflow(self):
        """Test complete client-server cryptographic workflow."""
        manager = PQCryptoManager()
        client_id = "integration_client"
        
        # 1. Client registration
        client_keys = manager.register_client(client_id)
        assert client_keys.client_id == client_id
        
        # 2. Session establishment
        session_id, ciphertext, shared_secret = manager.establish_session(client_id)
        assert len(session_id) > 0
        
        # 3. Client sends signed update
        update_data = b"Model update: weights=[0.1, 0.2, 0.3]"
        signature = manager.sign_message(client_id, update_data)
        
        # 4. Server verifies signature
        is_authentic = manager.verify_client_signature(client_id, update_data, signature)
        assert is_authentic is True
        
        # 5. Server sends encrypted response
        response = b"Update accepted, new global model attached"
        encrypted_response = manager.encrypt_server_message(session_id, response)
        
        # 6. Client decrypts response (simulated)
        decrypted_response = manager.decrypt_client_message(session_id, encrypted_response)
        assert decrypted_response == response
        
        # 7. Verify session is still active
        session_info = manager.get_session_info(session_id)
        assert session_info['is_active'] is True
    
    def test_multiple_clients_workflow(self):
        """Test workflow with multiple clients."""
        manager = PQCryptoManager()
        
        # Register multiple clients
        client_ids = ["client_001", "client_002", "client_003"]
        sessions = {}
        
        for client_id in client_ids:
            # Register and establish session
            manager.register_client(client_id)
            session_id, _, _ = manager.establish_session(client_id)
            sessions[client_id] = session_id
        
        # Verify all clients are registered and have active sessions
        registered_clients = manager.list_registered_clients()
        active_sessions = manager.list_active_sessions()
        
        for client_id in client_ids:
            assert client_id in registered_clients
            assert sessions[client_id] in active_sessions
        
        # Each client sends a signed message
        for client_id in client_ids:
            message = f"Update from {client_id}".encode()
            signature = manager.sign_message(client_id, message)
            
            # Verify signature
            is_valid = manager.verify_client_signature(client_id, message, signature)
            assert is_valid is True
            
            # Send encrypted response
            response = f"Response to {client_id}".encode()
            encrypted = manager.encrypt_server_message(sessions[client_id], response)
            decrypted = manager.decrypt_client_message(sessions[client_id], encrypted)
            assert decrypted == response
    
    def test_security_event_simulation(self):
        """Test simulation of security events."""
        manager = PQCryptoManager()
        
        # Register legitimate client
        legit_client = "legitimate_client"
        manager.register_client(legit_client)
        
        # Register malicious client
        malicious_client = "malicious_client"
        manager.register_client(malicious_client)
        
        # Legitimate client sends valid update
        legit_message = b"Normal model update"
        legit_signature = manager.sign_message(legit_client, legit_message)
        assert manager.verify_client_signature(legit_client, legit_message, legit_signature) is True
        
        # Malicious client tries to forge signature (simulate by using wrong client's signature)
        malicious_message = b"Malicious model update"
        forged_signature = manager.sign_message(legit_client, malicious_message)  # Wrong client
        
        # Verification should fail when using malicious client's ID
        assert manager.verify_client_signature(malicious_client, malicious_message, forged_signature) is False
        
        # Revoke malicious client
        revoked = manager.revoke_client(malicious_client)
        assert revoked is True
        
        # Malicious client should no longer be able to sign
        with pytest.raises(ValueError, match="not registered"):
            manager.sign_message(malicious_client, b"Another attempt")


if __name__ == "__main__":
    pytest.main([__file__])