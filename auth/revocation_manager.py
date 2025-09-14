"""
Revocation Manager Implementation
"""

import logging
from datetime import datetime
from typing import List
from .interfaces import IRevocationManager

logger = logging.getLogger(__name__)


class RevocationManager(IRevocationManager):
    """Manager for credential revocation operations."""
    
    def __init__(self):
        """Initialize the revocation manager."""
        self.revoked_clients = {}
    
    def revoke_credential(self, client_id: str, reason: str) -> None:
        """Revoke a client's credentials."""
        self.revoked_clients[client_id] = {
            'revoked_at': datetime.now(),
            'reason': reason
        }
        logger.info(f"Credentials revoked for {client_id}: {reason}")
    
    def is_revoked(self, client_id: str) -> bool:
        """Check if a client's credentials are revoked."""
        return client_id in self.revoked_clients
    
    def get_revocation_list(self) -> List[str]:
        """Get list of revoked client IDs."""
        return list(self.revoked_clients.keys())