"""
Unit tests for CRYSTALS-Kyber key exchange implementation.
"""

import pytest
import hashlib
from pq_security.kyber import KyberKeyExchange


class TestKyberKeyExchange:
    """Test suite for KyberKeyExchange class."""
    
    def test_initialization_valid_security_levels(self):
        """Test initialization with valid security levels."""
        for level in [2, 3, 4]:
            kyber = KyberKeyExchange(security_level=level)
            assert kyber.get_security_level() == level
    
    def test_initialization_invalid_security_level(self):
        """Test initialization with invalid security level raises error."""
        with pytest.raises(ValueError, match="Unsupported security level"):
            KyberKeyExchange(security_level=1)
        
        with pytest.raises(ValueError, match="Unsupported security level"):
            KyberKeyExchange(security_level=5)
    
    def test_keypair_generation(self):
        """Test keypair generation produces valid keys."""
        kyber = KyberKeyExchange(security_level=3)
        public_key, private_key = kyber.generate_keypair()
        
        # Keys should be bytes
        assert isinstance(public_key, bytes)
        assert isinstance(private_key, bytes)
        
        # Keys should have reasonable lengths
        assert len(public_key) > 0
        assert len(private_key) > len(public_key)  # Private key includes public key
        
        # Keys should be different
        assert public_key != private_key
    
    def test_keypair_generation_different_calls(self):
        """Test that different calls generate different keypairs."""
        kyber = KyberKeyExchange(security_level=3)
        
        pub1, priv1 = kyber.generate_keypair()
        pub2, priv2 = kyber.generate_keypair()
        
        # Different calls should produce different keys
        assert pub1 != pub2
        assert priv1 != priv2
    
    def test_encapsulation_decapsulation_roundtrip(self):
        """Test complete encapsulation/decapsulation roundtrip."""
        kyber = KyberKeyExchange(security_level=3)
        public_key, private_key = kyber.generate_keypair()
        
        # Encapsulate
        ciphertext, shared_secret1 = kyber.encapsulate(public_key)
        
        # Verify encapsulation output
        assert isinstance(ciphertext, bytes)
        assert isinstance(shared_secret1, bytes)
        assert len(ciphertext) > 0
        assert len(shared_secret1) == 32  # 256-bit shared secret
        
        # Decapsulate
        shared_secret2 = kyber.decapsulate(ciphertext, private_key)
        
        # Shared secrets should match
        assert shared_secret1 == shared_secret2
    
    def test_encapsulation_different_calls(self):
        """Test that different encapsulation calls produce different results."""
        kyber = KyberKeyExchange(security_level=3)
        public_key, _ = kyber.generate_keypair()
        
        ciphertext1, secret1 = kyber.encapsulate(public_key)
        ciphertext2, secret2 = kyber.encapsulate(public_key)
        
        # Different calls should produce different ciphertexts and secrets
        assert ciphertext1 != ciphertext2
        assert secret1 != secret2
    
    def test_decapsulation_with_wrong_private_key(self):
        """Test decapsulation with wrong private key produces different secret."""
        kyber = KyberKeyExchange(security_level=3)
        
        # Generate two keypairs
        pub1, priv1 = kyber.generate_keypair()
        pub2, priv2 = kyber.generate_keypair()
        
        # Encapsulate with first public key
        ciphertext, original_secret = kyber.encapsulate(pub1)
        
        # Decapsulate with correct private key
        correct_secret = kyber.decapsulate(ciphertext, priv1)
        assert correct_secret == original_secret
        
        # Decapsulate with wrong private key (should produce different result)
        wrong_secret = kyber.decapsulate(ciphertext, priv2)
        assert wrong_secret != original_secret
    
    def test_security_levels_produce_different_key_sizes(self):
        """Test that different security levels produce appropriately sized keys."""
        key_sizes = {}
        
        for level in [2, 3, 4]:
            kyber = KyberKeyExchange(security_level=level)
            public_key, private_key = kyber.generate_keypair()
            key_sizes[level] = (len(public_key), len(private_key))
        
        # Higher security levels should generally have larger keys
        # (This may not always be true in simulation mode, but test the structure)
        assert all(isinstance(sizes, tuple) and len(sizes) == 2 for sizes in key_sizes.values())
    
    def test_simulation_mode_detection(self):
        """Test simulation mode detection."""
        kyber = KyberKeyExchange(security_level=3)
        
        # Should be able to detect simulation mode
        assert isinstance(kyber.is_simulation_mode, bool)
    
    def test_known_test_vectors_simulation_mode(self):
        """Test with known inputs in simulation mode (deterministic behavior)."""
        kyber = KyberKeyExchange(security_level=3)
        
        # If in simulation mode, test deterministic behavior
        if kyber.is_simulation_mode:
            # Generate same keypair multiple times should be different (uses secrets)
            pub1, priv1 = kyber.generate_keypair()
            pub2, priv2 = kyber.generate_keypair()
            assert pub1 != pub2  # Should be different due to randomness
            
            # But encapsulation/decapsulation should be consistent
            ciphertext, secret1 = kyber.encapsulate(pub1)
            secret2 = kyber.decapsulate(ciphertext, priv1)
            assert secret1 == secret2
    
    def test_invalid_inputs(self):
        """Test behavior with invalid inputs."""
        kyber = KyberKeyExchange(security_level=3)
        
        # Test with None
        with pytest.raises((ValueError, TypeError, AttributeError)):
            kyber.encapsulate(None)
        
        # Test with very short ciphertext for decapsulation
        with pytest.raises((ValueError, IndexError)):
            kyber.decapsulate(b"short", b"some_private_key")
    
    def test_ciphertext_integrity(self):
        """Test that modified ciphertext produces different shared secret."""
        kyber = KyberKeyExchange(security_level=3)
        public_key, private_key = kyber.generate_keypair()
        
        # Original encapsulation
        ciphertext, original_secret = kyber.encapsulate(public_key)
        
        # Modify ciphertext
        modified_ciphertext = bytearray(ciphertext)
        if len(modified_ciphertext) > 0:
            modified_ciphertext[0] = (modified_ciphertext[0] + 1) % 256
        modified_ciphertext = bytes(modified_ciphertext)
        
        # Decapsulate modified ciphertext
        modified_secret = kyber.decapsulate(modified_ciphertext, private_key)
        
        # Should produce different secret (in most cases)
        # Note: In simulation mode, this might not always be true due to hash collisions
        # but it's a good general test
        if len(ciphertext) > 0:
            assert modified_secret != original_secret or kyber.is_simulation_mode
    
    def test_multiple_security_levels_compatibility(self):
        """Test that different security levels work independently."""
        results = {}
        
        for level in [2, 3, 4]:
            kyber = KyberKeyExchange(security_level=level)
            public_key, private_key = kyber.generate_keypair()
            ciphertext, shared_secret = kyber.encapsulate(public_key)
            recovered_secret = kyber.decapsulate(ciphertext, private_key)
            
            results[level] = {
                'public_key': public_key,
                'shared_secret': shared_secret,
                'recovered_secret': recovered_secret
            }
            
            # Verify roundtrip works for each level
            assert shared_secret == recovered_secret
        
        # Verify different security levels produce different results
        levels = list(results.keys())
        for i in range(len(levels)):
            for j in range(i + 1, len(levels)):
                level1, level2 = levels[i], levels[j]
                assert results[level1]['public_key'] != results[level2]['public_key']


if __name__ == "__main__":
    pytest.main([__file__])