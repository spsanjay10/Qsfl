"""
Unit tests for CRYSTALS-Dilithium digital signature implementation.
"""

import pytest
import hashlib
import secrets
from pq_security.dilithium import DilithiumSigner


class TestDilithiumSigner:
    """Test suite for DilithiumSigner class."""
    
    def test_initialization_valid_security_levels(self):
        """Test initialization with valid security levels."""
        for level in [2, 3, 5]:
            signer = DilithiumSigner(security_level=level)
            assert signer.get_security_level() == level
            assert isinstance(signer.is_simulation_mode, bool)
    
    def test_initialization_invalid_security_level(self):
        """Test initialization with invalid security level raises error."""
        with pytest.raises(ValueError, match="Unsupported security level"):
            DilithiumSigner(security_level=1)
        
        with pytest.raises(ValueError, match="Unsupported security level"):
            DilithiumSigner(security_level=6)
    
    def test_keypair_generation(self):
        """Test keypair generation produces valid keys."""
        signer = DilithiumSigner(security_level=3)
        public_key, private_key = signer.generate_keypair()
        
        # Check key types
        assert isinstance(public_key, bytes)
        assert isinstance(private_key, bytes)
        
        # Check key lengths are reasonable
        assert len(public_key) > 0
        assert len(private_key) > 0
        
        # Keys should be different
        assert public_key != private_key
        
        # Multiple generations should produce different keys
        public_key2, private_key2 = signer.generate_keypair()
        assert public_key != public_key2
        assert private_key != private_key2
    
    def test_sign_and_verify_basic(self):
        """Test basic signing and verification."""
        signer = DilithiumSigner(security_level=3)
        public_key, private_key = signer.generate_keypair()
        
        message = b"Hello, quantum-safe world!"
        signature = signer.sign(message, private_key)
        
        # Check signature properties
        assert isinstance(signature, bytes)
        assert len(signature) > 0
        
        # Verify signature
        assert signer.verify(message, signature, public_key) is True
    
    def test_verify_wrong_message(self):
        """Test verification fails with wrong message."""
        signer = DilithiumSigner(security_level=3)
        public_key, private_key = signer.generate_keypair()
        
        original_message = b"Original message"
        wrong_message = b"Wrong message"
        
        signature = signer.sign(original_message, private_key)
        
        # Verification should fail with wrong message
        assert signer.verify(wrong_message, signature, public_key) is False
    
    def test_sign_type_validation(self):
        """Test sign method validates input types."""
        signer = DilithiumSigner(security_level=3)
        public_key, private_key = signer.generate_keypair()
        
        # Test invalid message type
        with pytest.raises(TypeError, match="Message must be bytes"):
            signer.sign("string message", private_key)
        
        # Test invalid private key type
        with pytest.raises(TypeError, match="Private key must be bytes"):
            signer.sign(b"message", "string private key")
    
    def test_verify_type_validation(self):
        """Test verify method validates input types."""
        signer = DilithiumSigner(security_level=3)
        public_key, private_key = signer.generate_keypair()
        signature = signer.sign(b"test", private_key)
        
        # Test invalid message type
        with pytest.raises(TypeError, match="Message must be bytes"):
            signer.verify("string message", signature, public_key)
        
        # Test invalid signature type
        with pytest.raises(TypeError, match="Signature must be bytes"):
            signer.verify(b"message", "string signature", public_key)
        
        # Test invalid public key type
        with pytest.raises(TypeError, match="Public key must be bytes"):
            signer.verify(b"message", signature, "string public key")
    
    def test_verify_malformed_signature(self):
        """Test verification handles malformed signatures gracefully."""
        signer = DilithiumSigner(security_level=3)
        public_key, private_key = signer.generate_keypair()
        
        message = b"Test message"
        
        # Test with various malformed signatures
        malformed_signatures = [
            b"",  # Empty signature
            b"too_short",  # Too short
            secrets.token_bytes(32),  # Random bytes, wrong length
        ]
        
        for bad_signature in malformed_signatures:
            assert signer.verify(message, bad_signature, public_key) is False
    
    def test_security_level_consistency(self):
        """Test that different security levels work consistently."""
        message = b"Consistency test message"
        
        for level in [2, 3, 5]:
            signer = DilithiumSigner(security_level=level)
            public_key, private_key = signer.generate_keypair()
            
            signature = signer.sign(message, private_key)
            assert signer.verify(message, signature, public_key) is True
            
            # Check expected signature size
            expected_size = signer.get_signature_size()
            assert len(signature) == expected_size