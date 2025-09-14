"""
Post-Quantum Cryptography Manager

Unified orchestration layer for CRYSTALS-Kyber key exchange and CRYSTALS-Dilithium signatures.
Provides high-level interface for all post-quantum cryptographic operations.
"""

import json
import hashlib
import secrets
from typing import Dict, Tuple, Optional, Any, List
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from .kyber import KyberKeyExchange
from .dilithium import DilithiumSigner
from .interfaces import IPQCrypto


@dataclass
class CryptoSession:
    """Represents an active cryptographic session between client and server."""
    session_id: str
    client_id: str
    shared_secret: bytes
    client_public_key: bytes
    server_private_key: bytes
    created_at: datetime
    expires_at: datetime
    is_active: bool = True


@dataclass
class ClientKeys:
    """Container for client cryptographic keys."""
    client_id: str
    signature_public_key: bytes
    signature_private_key: bytes
    key_exchange_public_key: bytes
    key_exchange_private_key: bytes
    created_at: datetime
    expires_at: datetime


class PQCryptoManager(IPQCrypto):
    """Unified post-quantum cryptography manager coordinating Kyber and Dilithium operations."""
    
    def __init__(self, 
                 kyber_security_level: int = 3,
                 dilithium_security_level: int = 3,
                 session_timeout_hours: int = 24):
        """Initialize PQCryptoManager.
        
        Args:
            kyber_security_level: Security level for Kyber key exchange (2, 3, or 4)
            dilithium_security_level: Security level for Dilithium signatures (2, 3, or 5)
            session_timeout_hours: Session timeout in hours
        """
        self.kyber = KyberKeyExchange(security_level=kyber_security_level)
        self.dilithium = DilithiumSigner(security_level=dilithium_security_level)
        self.session_timeout = timedelta(hours=session_timeout_hours)
        
        # Active sessions and client keys storage
        self._sessions: Dict[str, CryptoSession] = {}
        self._client_keys: Dict[str, ClientKeys] = {}
        
        # Server's own key exchange keys (generated once)
        self._server_kx_public, self._server_kx_private = self.kyber.generate_keypair()
    
    def generate_keypair(self) -> Tuple[bytes, bytes]:
        """Generate a combined keypair for both key exchange and signatures.
        
        Returns:
            Tuple of (combined_public_key, combined_private_key) as JSON bytes
        """
        # Generate separate keypairs for key exchange and signatures
        kx_public, kx_private = self.kyber.generate_keypair()
        sig_public, sig_private = self.dilithium.generate_keypair()
        
        # Combine into JSON structure
        combined_public = {
            'key_exchange': kx_public.hex(),
            'signature': sig_public.hex(),
            'kyber_level': self.kyber.security_level,
            'dilithium_level': self.dilithium.security_level
        }
        
        combined_private = {
            'key_exchange': kx_private.hex(),
            'signature': sig_private.hex(),
            'kyber_level': self.kyber.security_level,
            'dilithium_level': self.dilithium.security_level
        }
        
        return (
            json.dumps(combined_public).encode('utf-8'),
            json.dumps(combined_private).encode('utf-8')
        )
    
    def register_client(self, client_id: str, key_lifetime_hours: int = 168) -> ClientKeys:
        """Register a new client with cryptographic keys.
        
        Args:
            client_id: Unique client identifier
            key_lifetime_hours: Key lifetime in hours (default: 1 week)
            
        Returns:
            ClientKeys object containing all client keys
        """
        if client_id in self._client_keys:
            raise ValueError(f"Client {client_id} already registered")
        
        # Generate keys for the client
        kx_public, kx_private = self.kyber.generate_keypair()
        sig_public, sig_private = self.dilithium.generate_keypair()
        
        now = datetime.utcnow()
        expires_at = now + timedelta(hours=key_lifetime_hours)
        
        client_keys = ClientKeys(
            client_id=client_id,
            signature_public_key=sig_public,
            signature_private_key=sig_private,
            key_exchange_public_key=kx_public,
            key_exchange_private_key=kx_private,
            created_at=now,
            expires_at=expires_at
        )
        
        self._client_keys[client_id] = client_keys
        return client_keys
    
    def establish_session(self, client_id: str) -> Tuple[str, bytes, bytes]:
        """Establish a secure session with a client using key exchange.
        
        Args:
            client_id: Client identifier
            
        Returns:
            Tuple of (session_id, ciphertext_for_client, shared_secret)
        """
        if client_id not in self._client_keys:
            raise ValueError(f"Client {client_id} not registered")
        
        client_keys = self._client_keys[client_id]
        
        # Check if client keys are still valid
        if datetime.utcnow() > client_keys.expires_at:
            raise ValueError(f"Client {client_id} keys have expired")
        
        # Perform key encapsulation with client's key exchange public key
        ciphertext, shared_secret = self.kyber.encapsulate(client_keys.key_exchange_public_key)
        
        # Create session
        session_id = secrets.token_hex(16)
        now = datetime.utcnow()
        
        session = CryptoSession(
            session_id=session_id,
            client_id=client_id,
            shared_secret=shared_secret,
            client_public_key=client_keys.key_exchange_public_key,
            server_private_key=self._server_kx_private,
            created_at=now,
            expires_at=now + self.session_timeout
        )
        
        self._sessions[session_id] = session
        return session_id, ciphertext, shared_secret
    
    def decrypt_client_message(self, session_id: str, ciphertext: bytes) -> bytes:
        """Decrypt a message from client using session shared secret.
        
        Args:
            session_id: Session identifier
            ciphertext: Encrypted message from client
            
        Returns:
            Decrypted plaintext
        """
        session = self._get_active_session(session_id)
        
        # Use shared secret as encryption key (simplified - in production use proper AEAD)
        key = hashlib.sha256(session.shared_secret).digest()
        
        # Simple XOR encryption for demonstration (use proper AEAD in production)
        if len(ciphertext) < 32:
            raise ValueError("Ciphertext too short")
        
        # Extract nonce and encrypted data
        nonce = ciphertext[:32]
        encrypted_data = ciphertext[32:]
        
        # Derive decryption key
        decryption_key = hashlib.sha256(key + nonce).digest()
        
        # Decrypt (XOR for simplicity)
        plaintext = bytes(a ^ b for a, b in zip(encrypted_data, 
                                               (decryption_key * ((len(encrypted_data) // 32) + 1))[:len(encrypted_data)]))
        
        return plaintext
    
    def encrypt_server_message(self, session_id: str, plaintext: bytes) -> bytes:
        """Encrypt a message to client using session shared secret.
        
        Args:
            session_id: Session identifier
            plaintext: Message to encrypt
            
        Returns:
            Encrypted ciphertext
        """
        session = self._get_active_session(session_id)
        
        # Use shared secret as encryption key
        key = hashlib.sha256(session.shared_secret).digest()
        
        # Generate random nonce
        nonce = secrets.token_bytes(32)
        
        # Derive encryption key
        encryption_key = hashlib.sha256(key + nonce).digest()
        
        # Encrypt (XOR for simplicity)
        ciphertext = bytes(a ^ b for a, b in zip(plaintext,
                                                (encryption_key * ((len(plaintext) // 32) + 1))[:len(plaintext)]))
        
        return nonce + ciphertext
    
    def sign_message(self, client_id: str, message: bytes) -> bytes:
        """Sign a message using client's private signature key.
        
        Args:
            client_id: Client identifier
            message: Message to sign
            
        Returns:
            Digital signature
        """
        if client_id not in self._client_keys:
            raise ValueError(f"Client {client_id} not registered")
        
        client_keys = self._client_keys[client_id]
        
        # Check if client keys are still valid
        if datetime.utcnow() > client_keys.expires_at:
            raise ValueError(f"Client {client_id} keys have expired")
        
        return self.dilithium.sign(message, client_keys.signature_private_key)
    
    def verify_client_signature(self, client_id: str, message: bytes, signature: bytes) -> bool:
        """Verify a client's signature.
        
        Args:
            client_id: Client identifier
            message: Original message
            signature: Signature to verify
            
        Returns:
            True if signature is valid, False otherwise
        """
        if client_id not in self._client_keys:
            return False
        
        client_keys = self._client_keys[client_id]
        
        # Check if client keys are still valid
        if datetime.utcnow() > client_keys.expires_at:
            return False
        
        return self.dilithium.verify(message, signature, client_keys.signature_public_key)
    
    def encrypt(self, plaintext: bytes, public_key: bytes) -> bytes:
        """Encrypt plaintext using public key (implements IPQCrypto interface).
        
        This method uses the combined public key format.
        """
        try:
            # Parse combined public key
            public_key_data = json.loads(public_key.decode('utf-8'))
            kx_public_key = bytes.fromhex(public_key_data['key_exchange'])
            
            # Perform key encapsulation
            ciphertext, shared_secret = self.kyber.encapsulate(kx_public_key)
            
            # Encrypt plaintext with shared secret
            key = hashlib.sha256(shared_secret).digest()
            nonce = secrets.token_bytes(32)
            encryption_key = hashlib.sha256(key + nonce).digest()
            
            encrypted_data = bytes(a ^ b for a, b in zip(plaintext,
                                                        (encryption_key * ((len(plaintext) // 32) + 1))[:len(plaintext)]))
            
            # Return ciphertext + nonce + encrypted_data
            return ciphertext + nonce + encrypted_data
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise ValueError(f"Invalid public key format: {e}")
    
    def decrypt(self, ciphertext: bytes, private_key: bytes) -> bytes:
        """Decrypt ciphertext using private key (implements IPQCrypto interface).
        
        This method uses the combined private key format.
        """
        try:
            # Parse combined private key
            private_key_data = json.loads(private_key.decode('utf-8'))
            kx_private_key = bytes.fromhex(private_key_data['key_exchange'])
            
            # Extract components (ciphertext length depends on Kyber security level)
            kyber_ciphertext_size = self.kyber.get_ciphertext_size()
            
            if len(ciphertext) < kyber_ciphertext_size + 32:
                raise ValueError("Ciphertext too short")
            
            kyber_ciphertext = ciphertext[:kyber_ciphertext_size]
            nonce = ciphertext[kyber_ciphertext_size:kyber_ciphertext_size + 32]
            encrypted_data = ciphertext[kyber_ciphertext_size + 32:]
            
            # Decapsulate shared secret
            shared_secret = self.kyber.decapsulate(kyber_ciphertext, kx_private_key)
            
            # Decrypt data
            key = hashlib.sha256(shared_secret).digest()
            decryption_key = hashlib.sha256(key + nonce).digest()
            
            plaintext = bytes(a ^ b for a, b in zip(encrypted_data,
                                                   (decryption_key * ((len(encrypted_data) // 32) + 1))[:len(encrypted_data)]))
            
            return plaintext
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise ValueError(f"Invalid private key format or ciphertext: {e}")
    
    def sign(self, message: bytes, private_key: bytes) -> bytes:
        """Sign message using private key (implements IPQCrypto interface).
        
        This method uses the combined private key format.
        """
        try:
            # Parse combined private key
            private_key_data = json.loads(private_key.decode('utf-8'))
            sig_private_key = bytes.fromhex(private_key_data['signature'])
            
            return self.dilithium.sign(message, sig_private_key)
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise ValueError(f"Invalid private key format: {e}")
    
    def verify(self, message: bytes, signature: bytes, public_key: bytes) -> bool:
        """Verify signature using public key (implements IPQCrypto interface).
        
        This method uses the combined public key format.
        """
        try:
            # Parse combined public key
            public_key_data = json.loads(public_key.decode('utf-8'))
            sig_public_key = bytes.fromhex(public_key_data['signature'])
            
            return self.dilithium.verify(message, signature, sig_public_key)
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            return False
    
    def cleanup_expired_sessions(self) -> int:
        """Remove expired sessions and return count of removed sessions."""
        now = datetime.utcnow()
        expired_sessions = [sid for sid, session in self._sessions.items() 
                           if now > session.expires_at]
        
        for session_id in expired_sessions:
            del self._sessions[session_id]
        
        return len(expired_sessions)
    
    def cleanup_expired_client_keys(self) -> int:
        """Remove expired client keys and return count of removed clients."""
        now = datetime.utcnow()
        expired_clients = [cid for cid, keys in self._client_keys.items()
                          if now > keys.expires_at]
        
        for client_id in expired_clients:
            del self._client_keys[client_id]
            # Also remove any sessions for this client
            client_sessions = [sid for sid, session in self._sessions.items()
                             if session.client_id == client_id]
            for session_id in client_sessions:
                del self._sessions[session_id]
        
        return len(expired_clients)
    
    def revoke_client(self, client_id: str) -> bool:
        """Revoke a client's access by removing their keys and sessions.
        
        Args:
            client_id: Client to revoke
            
        Returns:
            True if client was revoked, False if client not found
        """
        if client_id not in self._client_keys:
            return False
        
        # Remove client keys
        del self._client_keys[client_id]
        
        # Remove all sessions for this client
        client_sessions = [sid for sid, session in self._sessions.items()
                          if session.client_id == client_id]
        for session_id in client_sessions:
            del self._sessions[session_id]
        
        return True
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session information dictionary or None if not found
        """
        if session_id not in self._sessions:
            return None
        
        session = self._sessions[session_id]
        return {
            'session_id': session.session_id,
            'client_id': session.client_id,
            'created_at': session.created_at.isoformat(),
            'expires_at': session.expires_at.isoformat(),
            'is_active': session.is_active and datetime.utcnow() <= session.expires_at
        }
    
    def get_client_info(self, client_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a registered client.
        
        Args:
            client_id: Client identifier
            
        Returns:
            Client information dictionary or None if not found
        """
        if client_id not in self._client_keys:
            return None
        
        keys = self._client_keys[client_id]
        return {
            'client_id': keys.client_id,
            'created_at': keys.created_at.isoformat(),
            'expires_at': keys.expires_at.isoformat(),
            'is_expired': datetime.utcnow() > keys.expires_at
        }
    
    def list_active_sessions(self) -> List[str]:
        """Get list of active session IDs."""
        now = datetime.utcnow()
        return [sid for sid, session in self._sessions.items()
                if session.is_active and now <= session.expires_at]
    
    def list_registered_clients(self) -> List[str]:
        """Get list of registered client IDs."""
        return list(self._client_keys.keys())
    
    def get_server_public_key(self) -> bytes:
        """Get server's public key for key exchange."""
        return self._server_kx_public
    
    def is_simulation_mode(self) -> bool:
        """Check if running in simulation mode."""
        return self.kyber.is_simulation_mode or self.dilithium.is_simulation_mode
    
    def get_security_info(self) -> Dict[str, Any]:
        """Get information about current security configuration."""
        return {
            'kyber_security_level': self.kyber.security_level,
            'dilithium_security_level': self.dilithium.security_level,
            'kyber_simulation_mode': self.kyber.is_simulation_mode,
            'dilithium_simulation_mode': self.dilithium.is_simulation_mode,
            'session_timeout_hours': self.session_timeout.total_seconds() / 3600,
            'active_sessions': len(self.list_active_sessions()),
            'registered_clients': len(self.list_registered_clients())
        }
    
    def _get_active_session(self, session_id: str) -> CryptoSession:
        """Get an active session, raising error if not found or expired."""
        if session_id not in self._sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self._sessions[session_id]
        
        if not session.is_active:
            raise ValueError(f"Session {session_id} is not active")
        
        if datetime.utcnow() > session.expires_at:
            session.is_active = False
            raise ValueError(f"Session {session_id} has expired")
        
        return session