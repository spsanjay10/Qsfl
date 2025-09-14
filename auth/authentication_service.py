"""
Authentication Service Implementation
"""

import logging
from datetime import datetime, timedelta
from typing import Optional
from .interfaces import IAuthenticationService, ClientCredentials, CredentialStatus

logger = logging.getLogger(__name__)


class AuthenticationService(IAuthenticationService):
    """Service for client authentication operations."""
    
    def __init__(self):
        """Initialize the authentication service."""
        self.clients = {}
    
    def register_client(self, client_id: str) -> ClientCredentials:
        """Register a new client and issue credentials."""
        from pq_security.dilithium import DilithiumSigner
        
        # Generate post-quantum keypair
        signer = DilithiumSigner()
        public_key, private_key = signer.generate_keypair()
        
        # Create credentials
        credentials = ClientCredentials(
            client_id=client_id,
            public_key=public_key,
            private_key=private_key,
            issued_at=datetime.now(),
            expires_at=datetime.now() + timedelta(days=365),
            status=CredentialStatus.ACTIVE
        )
        
        self.clients[client_id] = credentials
        logger.info(f"Client {client_id} registered successfully")
        
        return credentials
    
    def authenticate_client(self, client_id: str, signature: bytes, message: bytes) -> bool:
        """Authenticate a client using signature verification."""
        try:
            if client_id not in self.clients:
                return False
            
            credentials = self.clients[client_id]
            
            # Check if credentials are valid
            if credentials.status != CredentialStatus.ACTIVE:
                return False
            
            if credentials.expires_at < datetime.now():
                return False
            
            # Verify signature (simplified for demo)
            from pq_security.dilithium import DilithiumSigner
            signer = DilithiumSigner()
            
            return signer.verify(message, signature, credentials.public_key)
            
        except Exception as e:
            logger.error(f"Authentication error for {client_id}: {e}")
            return False
    
    def revoke_client(self, client_id: str) -> None:
        """Revoke client credentials."""
        if client_id in self.clients:
            self.clients[client_id].status = CredentialStatus.REVOKED
            logger.info(f"Client {client_id} credentials revoked")
    
    def is_client_valid(self, client_id: str) -> bool:
        """Check if client credentials are valid."""
        if client_id not in self.clients:
            return False
        
        credentials = self.clients[client_id]
        return (credentials.status == CredentialStatus.ACTIVE and 
                credentials.expires_at > datetime.now())
    
    def get_client_credentials(self, client_id: str) -> Optional[ClientCredentials]:
        """Retrieve client credentials."""
        return self.clients.get(client_id)