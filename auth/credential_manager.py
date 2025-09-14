"""
Credential Manager Implementation
"""

import logging
from datetime import datetime, timedelta
from typing import Optional
from .interfaces import ICredentialManager, ClientCredentials, CredentialStatus

logger = logging.getLogger(__name__)


class CredentialManager(ICredentialManager):
    """Manager for credential lifecycle operations."""
    
    def __init__(self):
        """Initialize the credential manager."""
        self.credentials_store = {}
    
    def issue_credentials(self, client_id: str) -> ClientCredentials:
        """Issue new credentials for a client."""
        from pq_security.dilithium import DilithiumSigner
        
        signer = DilithiumSigner()
        public_key, private_key = signer.generate_keypair()
        
        credentials = ClientCredentials(
            client_id=client_id,
            public_key=public_key,
            private_key=private_key,
            issued_at=datetime.now(),
            expires_at=datetime.now() + timedelta(days=365),
            status=CredentialStatus.ACTIVE
        )
        
        self.store_credentials(credentials)
        return credentials
    
    def renew_credentials(self, client_id: str) -> ClientCredentials:
        """Renew existing credentials for a client."""
        # Revoke old credentials
        old_credentials = self.load_credentials(client_id)
        if old_credentials:
            old_credentials.status = CredentialStatus.EXPIRED
        
        # Issue new credentials
        return self.issue_credentials(client_id)
    
    def store_credentials(self, credentials: ClientCredentials) -> None:
        """Store client credentials securely."""
        self.credentials_store[credentials.client_id] = credentials
        logger.debug(f"Credentials stored for {credentials.client_id}")
    
    def load_credentials(self, client_id: str) -> Optional[ClientCredentials]:
        """Load client credentials from storage."""
        return self.credentials_store.get(client_id)