"""
Post-Quantum Security Interfaces

Defines abstract base classes and interfaces for quantum-resistant cryptographic operations.
"""

from abc import ABC, abstractmethod
from typing import Tuple


class IPQCrypto(ABC):
    """Interface for post-quantum cryptographic operations."""
    
    @abstractmethod
    def generate_keypair(self) -> Tuple[bytes, bytes]:
        """Generate a public/private key pair.
        
        Returns:
            Tuple of (public_key, private_key) as bytes
        """
        pass
    
    @abstractmethod
    def encrypt(self, plaintext: bytes, public_key: bytes) -> bytes:
        """Encrypt plaintext using public key.
        
        Args:
            plaintext: Data to encrypt
            public_key: Public key for encryption
            
        Returns:
            Encrypted ciphertext
        """
        pass
    
    @abstractmethod
    def decrypt(self, ciphertext: bytes, private_key: bytes) -> bytes:
        """Decrypt ciphertext using private key.
        
        Args:
            ciphertext: Encrypted data
            private_key: Private key for decryption
            
        Returns:
            Decrypted plaintext
        """
        pass
    
    @abstractmethod
    def sign(self, message: bytes, private_key: bytes) -> bytes:
        """Sign a message using private key.
        
        Args:
            message: Message to sign
            private_key: Private key for signing
            
        Returns:
            Digital signature
        """
        pass
    
    @abstractmethod
    def verify(self, message: bytes, signature: bytes, public_key: bytes) -> bool:
        """Verify a signature using public key.
        
        Args:
            message: Original message
            signature: Digital signature to verify
            public_key: Public key for verification
            
        Returns:
            True if signature is valid, False otherwise
        """
        pass


class IKeyExchange(ABC):
    """Interface for key exchange operations."""
    
    @abstractmethod
    def generate_keypair(self) -> Tuple[bytes, bytes]:
        """Generate key exchange keypair."""
        pass
    
    @abstractmethod
    def encapsulate(self, public_key: bytes) -> Tuple[bytes, bytes]:
        """Encapsulate shared secret with public key.
        
        Returns:
            Tuple of (ciphertext, shared_secret)
        """
        pass
    
    @abstractmethod
    def decapsulate(self, ciphertext: bytes, private_key: bytes) -> bytes:
        """Decapsulate shared secret from ciphertext."""
        pass


class IDigitalSignature(ABC):
    """Interface for digital signature operations."""
    
    @abstractmethod
    def generate_keypair(self) -> Tuple[bytes, bytes]:
        """Generate signature keypair."""
        pass
    
    @abstractmethod
    def sign(self, message: bytes, private_key: bytes) -> bytes:
        """Sign message with private key."""
        pass
    
    @abstractmethod
    def verify(self, message: bytes, signature: bytes, public_key: bytes) -> bool:
        """Verify signature with public key."""
        pass