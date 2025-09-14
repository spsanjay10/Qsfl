"""
Authentication Interfaces

Defines abstract base classes and interfaces for client authentication operations.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from typing import Optional


class CredentialStatus(Enum):
    """Status of client credentials."""
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    SUSPENDED = "suspended"


@dataclass
class ClientCredentials:
    """Client credential data structure."""
    client_id: str
    public_key: bytes
    private_key: bytes
    issued_at: datetime
    expires_at: datetime
    status: CredentialStatus


class IAuthenticationService(ABC):
    """Interface for client authentication operations."""
    
    @abstractmethod
    def register_client(self, client_id: str) -> ClientCredentials:
        """Register a new client and issue credentials.
        
        Args:
            client_id: Unique identifier for the client
            
        Returns:
            Client credentials including keypair
        """
        pass
    
    @abstractmethod
    def authenticate_client(self, client_id: str, signature: bytes, message: bytes) -> bool:
        """Authenticate a client using signature verification.
        
        Args:
            client_id: Client identifier
            signature: Digital signature to verify
            message: Original message that was signed
            
        Returns:
            True if authentication successful, False otherwise
        """
        pass
    
    @abstractmethod
    def revoke_client(self, client_id: str) -> None:
        """Revoke client credentials.
        
        Args:
            client_id: Client identifier to revoke
        """
        pass
    
    @abstractmethod
    def is_client_valid(self, client_id: str) -> bool:
        """Check if client credentials are valid.
        
        Args:
            client_id: Client identifier to check
            
        Returns:
            True if client is valid and active, False otherwise
        """
        pass
    
    @abstractmethod
    def get_client_credentials(self, client_id: str) -> Optional[ClientCredentials]:
        """Retrieve client credentials.
        
        Args:
            client_id: Client identifier
            
        Returns:
            Client credentials if found, None otherwise
        """
        pass


class ICredentialManager(ABC):
    """Interface for credential lifecycle management."""
    
    @abstractmethod
    def issue_credentials(self, client_id: str) -> ClientCredentials:
        """Issue new credentials for a client."""
        pass
    
    @abstractmethod
    def renew_credentials(self, client_id: str) -> ClientCredentials:
        """Renew existing credentials for a client."""
        pass
    
    @abstractmethod
    def store_credentials(self, credentials: ClientCredentials) -> None:
        """Store client credentials securely."""
        pass
    
    @abstractmethod
    def load_credentials(self, client_id: str) -> Optional[ClientCredentials]:
        """Load client credentials from storage."""
        pass


class IRevocationManager(ABC):
    """Interface for credential revocation management."""
    
    @abstractmethod
    def revoke_credential(self, client_id: str, reason: str) -> None:
        """Revoke a client's credentials."""
        pass
    
    @abstractmethod
    def is_revoked(self, client_id: str) -> bool:
        """Check if a client's credentials are revoked."""
        pass
    
    @abstractmethod
    def get_revocation_list(self) -> list[str]:
        """Get list of revoked client IDs."""
        pass