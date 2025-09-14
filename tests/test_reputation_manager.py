"""
Unit tests for dynamic response and reputation system.

Tests reputation-based response scenarios and client behavior tracking.
"""

import pytest
import numpy as np
import tempfile
import os
from datetime import datetime, timedelta
from typing import List

from anomaly_detection.reputation_manager import (
    ClientReputationManager, 
    ResponseOrchestrator,
    ReputationStatus,
    ClientReputation
)
from anomaly_detection.interfaces import AnomalyReport, ResponseAction


class TestClientReputationManager:
    """Test cases for ClientReputationManager."""
    
    @pytest.fixture
    def temp_db_path(self) -> str:
        """Create temporary database path."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp_file:
            return tmp_file.name
    
    @pytest.fixture
    def reputation_manager(self, temp_db_path) -> ClientReputationManager:
        """Create reputation manager with temporary database."""
        return ClientReputationManager(
            db_path=temp_db_path,
            reputation_decay=0.95,
            anomaly_penalty=0.1,
            recovery_bonus=0.05,
            quarantine_threshold=0.3,
            ban_threshold=0.1,
            consecutive_anomaly_limit=3,
            quarantine_duration_hours=1,  # Short for testing
            max_quarantine_count=2
        )
    
    def teardown_method(self, method):
        """Clean up temporary files."""
        # Clean up any temporary database files
        for file in os.listdir('.'):
            if file.endswith('.db') and 'tmp' in file:
                try:
                    os.unlink(file)
                except:
                    pass
    
    def test_initialization(self, reputation_manager):
        """Test reputation manager initialization."""
        assert reputation_manager.reputation_decay == 0.95
        assert reputation_manager.anomaly_penalty == 0.1
        assert reputation_manager.quarantine_threshold == 0.3
        assert len(reputation_manager.clients) == 0
    
    def test_new_client_default_reputation(self, reputation_manager):
        """Test default reputation for new clients."""
        client_id = "new_client"
        
        # New client should have default reputation
        reputation = reputation_manager.get_reputation(client_id)
        assert reputation == 1.0
        
        influence = reputation_manager.get_influence_weight(client_id)
        assert influence == 1.0
        
        assert not reputation_manager.is_quarantined(client_id)
    
    def test_normal_update_reputation(self, reputation_manager):
        """Test reputation update for normal behavior."""
        client_id = "normal_client"
        
        # Send normal updates (positive anomaly scores)
        for i in range(5):
            reputation_manager.update_reputation(client_id, 0.5)  # Normal score
        
        reputation = reputation_manager.get_reputation(client_id)
        assert reputation >= 0.9  # Should maintain high reputation
        
        status = reputation_manager.get_client_status(client_id)
        assert status in [ReputationStatus.GOOD, ReputationStatus.EXCELLENT]
    
    def test_anomalous_update_reputation(self, reputation_manager):
        """Test reputation update for anomalous behavior."""
        client_id = "anomalous_client"
        
        # Send anomalous updates (negative anomaly scores)
        for i in range(3):
            reputation_manager.update_reputation(client_id, -0.5)  # Anomalous score
        
        reputation = reputation_manager.get_reputation(client_id)
        assert reputation < 1.0  # Should decrease reputation
        
        client_info = reputation_manager.get_client_info(client_id)
        assert client_info['anomalous_updates'] == 3
        assert client_info['consecutive_anomalies'] == 3
    
    def test_consecutive_anomaly_quarantine(self, reputation_manager):
        """Test quarantine due to consecutive anomalies."""
        client_id = "consecutive_anomaly_client"
        
        # Send consecutive anomalous updates to trigger quarantine
        for i in range(4):  # Exceeds consecutive_anomaly_limit of 3
            reputation_manager.update_reputation(client_id, -0.3)
        
        assert reputation_manager.is_quarantined(client_id)
        
        status = reputation_manager.get_client_status(client_id)
        assert status == ReputationStatus.QUARANTINED
        
        influence = reputation_manager.get_influence_weight(client_id)
        assert influence == 0.0
    
    def test_low_reputation_quarantine(self, reputation_manager):
        """Test quarantine due to low reputation score."""
        client_id = "low_reputation_client"
        
        # Send many anomalous updates to drive reputation below threshold
        # Use smaller penalties to avoid consecutive anomaly quarantine
        for i in range(15):
            reputation_manager.update_reputation(client_id, -0.05)
            # Add normal updates to reset consecutive counter
            if i % 4 == 3:
                reputation_manager.update_reputation(client_id, 0.1)
        
        reputation = reputation_manager.get_reputation(client_id)
        
        # Should be quarantined either due to low reputation or consecutive anomalies
        assert reputation_manager.is_quarantined(client_id)
        
        status = reputation_manager.get_client_status(client_id)
        assert status == ReputationStatus.QUARANTINED
    
    def test_reputation_recovery(self, reputation_manager):
        """Test reputation recovery after anomalies."""
        client_id = "recovery_client"
        
        # First, cause some anomalies
        for i in range(2):
            reputation_manager.update_reputation(client_id, -0.3)
        
        initial_reputation = reputation_manager.get_reputation(client_id)
        
        # Then send normal updates
        for i in range(3):
            reputation_manager.update_reputation(client_id, 0.5)
        
        final_reputation = reputation_manager.get_reputation(client_id)
        assert final_reputation > initial_reputation  # Should recover
        
        client_info = reputation_manager.get_client_info(client_id)
        assert client_info['consecutive_anomalies'] == 0  # Should reset
    
    def test_quarantine_expiration(self, reputation_manager):
        """Test quarantine expiration after duration."""
        client_id = "quarantine_expiry_client"
        
        # Trigger quarantine
        for i in range(4):
            reputation_manager.update_reputation(client_id, -0.3)
        
        assert reputation_manager.is_quarantined(client_id)
        
        # Manually set quarantine start time to past
        client = reputation_manager.clients[client_id]
        client.quarantine_start = datetime.now() - timedelta(hours=2)  # Expired
        
        # Check reputation (should update status)
        reputation_manager.get_reputation(client_id)
        
        # Should no longer be quarantined
        assert not reputation_manager.is_quarantined(client_id)
        
        status = reputation_manager.get_client_status(client_id)
        assert status == ReputationStatus.SUSPICIOUS  # Released to suspicious
    
    def test_permanent_ban(self, reputation_manager):
        """Test permanent ban conditions."""
        client_id = "ban_client"
        
        # Trigger multiple quarantines to reach ban threshold
        for quarantine_round in range(3):  # Exceeds max_quarantine_count of 2
            # Trigger quarantine
            for i in range(4):
                reputation_manager.update_reputation(client_id, -0.5)
            
            # Manually expire quarantine
            if client_id in reputation_manager.clients:
                client = reputation_manager.clients[client_id]
                if client.quarantine_start:
                    client.quarantine_start = datetime.now() - timedelta(hours=2)
        
        # Should be banned
        status = reputation_manager.get_client_status(client_id)
        assert status == ReputationStatus.BANNED
        
        influence = reputation_manager.get_influence_weight(client_id)
        assert influence == 0.0
    
    def test_influence_weight_calculation(self, reputation_manager):
        """Test influence weight calculation based on reputation."""
        client_id = "influence_client"
        
        # Test different reputation levels
        test_cases = [
            (0.3, "high"),      # High reputation (normal updates)
            (-0.1, "medium"),   # Medium reputation (small anomalies)
            (-0.2, "low"),      # Low reputation (larger anomalies)
        ]
        
        for anomaly_score, case_name in test_cases:
            # Reset client
            reputation_manager.reset_client_reputation(client_id)
            
            # Apply updates to reach desired reputation level
            # Use fewer updates to avoid quarantine
            num_updates = 2 if anomaly_score < 0 else 1
            for i in range(num_updates):
                reputation_manager.update_reputation(client_id, anomaly_score)
                # Add normal update to prevent consecutive anomaly quarantine
                if anomaly_score < 0:
                    reputation_manager.update_reputation(client_id, 0.1)
            
            influence = reputation_manager.get_influence_weight(client_id)
            reputation = reputation_manager.get_reputation(client_id)
            
            # Influence should be between 0 and 1
            assert 0.0 <= influence <= 1.0
            
            # If not quarantined, higher reputation should mean higher influence
            if not reputation_manager.is_quarantined(client_id):
                if reputation > 0.5:
                    assert influence > 0.1
                # Influence should generally correlate with reputation
                assert influence >= 0.0
    
    def test_client_info_retrieval(self, reputation_manager):
        """Test comprehensive client information retrieval."""
        client_id = "info_client"
        
        # Send some updates
        reputation_manager.update_reputation(client_id, 0.3)  # Normal
        reputation_manager.update_reputation(client_id, -0.4)  # Anomaly
        reputation_manager.update_reputation(client_id, 0.2)  # Normal
        
        client_info = reputation_manager.get_client_info(client_id)
        
        assert client_info is not None
        assert client_info['client_id'] == client_id
        assert client_info['total_updates'] == 3
        assert client_info['anomalous_updates'] == 1
        assert client_info['anomaly_rate'] == 1/3
        assert 'reputation_score' in client_info
        assert 'status' in client_info
        assert 'influence_weight' in client_info
        assert 'last_update_time' in client_info
    
    def test_suspicious_clients_list(self, reputation_manager):
        """Test retrieval of suspicious clients."""
        # Create clients with different behaviors
        normal_client = "normal_client"
        suspicious_client = "suspicious_client"
        quarantined_client = "quarantined_client"
        
        # Normal client
        reputation_manager.update_reputation(normal_client, 0.5)
        
        # Suspicious client
        for i in range(5):
            reputation_manager.update_reputation(suspicious_client, -0.2)
        
        # Quarantined client
        for i in range(4):
            reputation_manager.update_reputation(quarantined_client, -0.5)
        
        suspicious_clients = reputation_manager.get_suspicious_clients()
        
        assert normal_client not in suspicious_clients
        assert suspicious_client in suspicious_clients or quarantined_client in suspicious_clients
    
    def test_reputation_statistics(self, reputation_manager):
        """Test overall reputation system statistics."""
        # Create clients with various behaviors
        clients = ["client_1", "client_2", "client_3"]
        
        for i, client_id in enumerate(clients):
            # Different behavior patterns
            if i == 0:  # Normal client
                reputation_manager.update_reputation(client_id, 0.5)
            elif i == 1:  # Suspicious client
                for j in range(3):
                    reputation_manager.update_reputation(client_id, -0.3)
            else:  # Mixed behavior
                reputation_manager.update_reputation(client_id, 0.2)
                reputation_manager.update_reputation(client_id, -0.1)
        
        stats = reputation_manager.get_reputation_statistics()
        
        assert stats['total_clients'] == len(clients)
        assert 'status_distribution' in stats
        assert 'average_reputation' in stats
        assert 'total_updates' in stats
        assert 'total_anomalies' in stats
        assert 'overall_anomaly_rate' in stats
        
        assert stats['total_updates'] > 0
        assert 0.0 <= stats['average_reputation'] <= 1.0
        assert 0.0 <= stats['overall_anomaly_rate'] <= 1.0
    
    def test_manual_ban_unban(self, reputation_manager):
        """Test manual ban and unban functionality."""
        client_id = "manual_ban_client"
        
        # Initially normal
        reputation_manager.update_reputation(client_id, 0.5)
        assert not reputation_manager.is_quarantined(client_id)
        
        # Manual ban
        reputation_manager.ban_client(client_id)
        
        status = reputation_manager.get_client_status(client_id)
        assert status == ReputationStatus.BANNED
        assert reputation_manager.get_influence_weight(client_id) == 0.0
        
        # Manual unban
        reputation_manager.unban_client(client_id)
        
        status = reputation_manager.get_client_status(client_id)
        assert status == ReputationStatus.SUSPICIOUS  # Should be suspicious after unban
        assert reputation_manager.get_influence_weight(client_id) > 0.0
    
    def test_database_persistence(self, temp_db_path):
        """Test database persistence across manager instances."""
        client_id = "persistence_client"
        
        # Create first manager instance
        manager1 = ClientReputationManager(
            db_path=temp_db_path,
            quarantine_duration_hours=1
        )
        
        # Add client data
        manager1.update_reputation(client_id, -0.3)
        manager1.update_reputation(client_id, 0.2)
        
        original_reputation = manager1.get_reputation(client_id)
        original_info = manager1.get_client_info(client_id)
        
        # Create second manager instance (should load from database)
        manager2 = ClientReputationManager(
            db_path=temp_db_path,
            quarantine_duration_hours=1
        )
        
        # Should have same data
        loaded_reputation = manager2.get_reputation(client_id)
        loaded_info = manager2.get_client_info(client_id)
        
        assert abs(original_reputation - loaded_reputation) < 0.001
        assert original_info['total_updates'] == loaded_info['total_updates']
        assert original_info['anomalous_updates'] == loaded_info['anomalous_updates']
        
        # Clean up
        try:
            os.unlink(temp_db_path)
        except:
            pass


class TestResponseOrchestrator:
    """Test cases for ResponseOrchestrator."""
    
    @pytest.fixture
    def temp_db_path(self) -> str:
        """Create temporary database path."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp_file:
            return tmp_file.name
    
    @pytest.fixture
    def orchestrator(self, temp_db_path) -> ResponseOrchestrator:
        """Create response orchestrator with reputation manager."""
        reputation_manager = ClientReputationManager(
            db_path=temp_db_path,
            quarantine_threshold=0.3,
            consecutive_anomaly_limit=3,
            quarantine_duration_hours=1
        )
        return ResponseOrchestrator(reputation_manager)
    
    def teardown_method(self, method):
        """Clean up temporary files."""
        for file in os.listdir('.'):
            if file.endswith('.db') and 'tmp' in file:
                try:
                    os.unlink(file)
                except:
                    pass
    
    def test_normal_update_response(self, orchestrator):
        """Test response to normal updates."""
        report = AnomalyReport(
            client_id="normal_client",
            anomaly_score=0.5,  # Normal score
            shap_values={},
            explanation="Normal update",
            recommended_action=ResponseAction.ALLOW,
            timestamp=datetime.now()
        )
        
        action = orchestrator.process_anomaly_report(report)
        assert action == ResponseAction.ALLOW
    
    def test_moderate_anomaly_response(self, orchestrator):
        """Test response to moderate anomalies."""
        report = AnomalyReport(
            client_id="moderate_client",
            anomaly_score=-0.4,  # Moderate anomaly
            shap_values={},
            explanation="Moderate anomaly",
            recommended_action=ResponseAction.REDUCE_WEIGHT,
            timestamp=datetime.now()
        )
        
        action = orchestrator.process_anomaly_report(report)
        assert action in [ResponseAction.REDUCE_WEIGHT, ResponseAction.ALLOW]
    
    def test_severe_anomaly_response(self, orchestrator):
        """Test response to severe anomalies."""
        client_id = "severe_client"
        
        # First make client suspicious
        for i in range(3):
            report = AnomalyReport(
                client_id=client_id,
                anomaly_score=-0.3,
                shap_values={},
                explanation="Building suspicion",
                recommended_action=ResponseAction.REDUCE_WEIGHT,
                timestamp=datetime.now()
            )
            orchestrator.process_anomaly_report(report)
        
        # Then send severe anomaly
        severe_report = AnomalyReport(
            client_id=client_id,
            anomaly_score=-0.8,  # Severe anomaly
            shap_values={},
            explanation="Severe anomaly",
            recommended_action=ResponseAction.QUARANTINE,
            timestamp=datetime.now()
        )
        
        action = orchestrator.process_anomaly_report(severe_report)
        assert action in [ResponseAction.QUARANTINE, ResponseAction.REJECT]
    
    def test_quarantined_client_response(self, orchestrator):
        """Test response for quarantined clients."""
        client_id = "quarantined_client"
        
        # Trigger quarantine
        for i in range(4):
            report = AnomalyReport(
                client_id=client_id,
                anomaly_score=-0.5,
                shap_values={},
                explanation="Triggering quarantine",
                recommended_action=ResponseAction.QUARANTINE,
                timestamp=datetime.now()
            )
            orchestrator.process_anomaly_report(report)
        
        # Any subsequent update should be rejected
        new_report = AnomalyReport(
            client_id=client_id,
            anomaly_score=0.5,  # Even normal updates
            shap_values={},
            explanation="Update from quarantined client",
            recommended_action=ResponseAction.ALLOW,
            timestamp=datetime.now()
        )
        
        action = orchestrator.process_anomaly_report(new_report)
        assert action == ResponseAction.REJECT
    
    def test_aggregation_weights(self, orchestrator):
        """Test aggregation weight calculation."""
        clients = ["good_client", "suspicious_client", "quarantined_client"]
        
        # Set up different client reputations
        # Good client
        orchestrator.reputation_manager.update_reputation("good_client", 0.5)
        
        # Suspicious client (fewer anomalies to avoid quarantine)
        for i in range(2):
            orchestrator.reputation_manager.update_reputation("suspicious_client", -0.15)
        
        # Quarantined client
        for i in range(4):
            orchestrator.reputation_manager.update_reputation("quarantined_client", -0.5)
        
        weights = orchestrator.get_aggregation_weights(clients)
        
        assert len(weights) == len(clients)
        
        # Good client should have high weight
        assert weights["good_client"] > 0.5
        
        # Suspicious client should have reduced weight (if not quarantined)
        if not orchestrator.reputation_manager.is_quarantined("suspicious_client"):
            assert 0.0 < weights["suspicious_client"] < weights["good_client"]
        else:
            # If quarantined, should have zero weight
            assert weights["suspicious_client"] == 0.0
        
        # Quarantined client should have zero weight
        assert weights["quarantined_client"] == 0.0
    
    def test_client_filtering(self, orchestrator):
        """Test filtering of quarantined clients."""
        clients = ["normal_1", "normal_2", "quarantined_1", "quarantined_2"]
        
        # Set up normal clients
        orchestrator.reputation_manager.update_reputation("normal_1", 0.5)
        orchestrator.reputation_manager.update_reputation("normal_2", 0.3)
        
        # Set up quarantined clients
        for client in ["quarantined_1", "quarantined_2"]:
            for i in range(4):
                orchestrator.reputation_manager.update_reputation(client, -0.5)
        
        filtered_clients = orchestrator.filter_clients(clients)
        
        # Should only include normal clients
        assert "normal_1" in filtered_clients
        assert "normal_2" in filtered_clients
        assert "quarantined_1" not in filtered_clients
        assert "quarantined_2" not in filtered_clients


class TestReputationIntegration:
    """Integration tests for reputation system components."""
    
    def test_end_to_end_reputation_workflow(self):
        """Test complete reputation management workflow."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp_file:
            db_path = tmp_file.name
        
        try:
            # Initialize system
            reputation_manager = ClientReputationManager(
                db_path=db_path,
                quarantine_threshold=0.4,
                consecutive_anomaly_limit=3,
                quarantine_duration_hours=1
            )
            
            orchestrator = ResponseOrchestrator(reputation_manager)
            
            # Simulate federated learning rounds
            clients = ["honest_1", "honest_2", "malicious_1", "intermittent_1"]
            
            for round_num in range(10):
                for client_id in clients:
                    # Generate different behavior patterns
                    if "honest" in client_id:
                        anomaly_score = np.random.normal(0.3, 0.1)  # Mostly normal
                    elif "malicious" in client_id:
                        anomaly_score = np.random.normal(-0.5, 0.2)  # Mostly anomalous
                    else:  # intermittent
                        anomaly_score = np.random.choice([0.4, -0.3], p=[0.7, 0.3])  # Mixed
                    
                    # Create anomaly report
                    report = AnomalyReport(
                        client_id=client_id,
                        anomaly_score=anomaly_score,
                        shap_values={},
                        explanation=f"Round {round_num} update",
                        recommended_action=ResponseAction.ALLOW,
                        timestamp=datetime.now()
                    )
                    
                    # Process report
                    action = orchestrator.process_anomaly_report(report)
                    
                    # Log action for verification
                    print(f"Round {round_num}, {client_id}: score={anomaly_score:.3f}, action={action}")
            
            # Verify final states
            stats = reputation_manager.get_reputation_statistics()
            
            # Should have processed updates for all clients
            assert stats['total_clients'] == len(clients)
            assert stats['total_updates'] > 0
            
            # Honest clients should have good reputation
            for client_id in ["honest_1", "honest_2"]:
                reputation = reputation_manager.get_reputation(client_id)
                assert reputation > 0.5
                assert not reputation_manager.is_quarantined(client_id)
            
            # Malicious client should have poor reputation or be quarantined
            malicious_reputation = reputation_manager.get_reputation("malicious_1")
            malicious_quarantined = reputation_manager.is_quarantined("malicious_1")
            
            assert malicious_reputation < 0.6 or malicious_quarantined
            
            # Test aggregation weights
            weights = orchestrator.get_aggregation_weights(clients)
            
            # Honest clients should have higher weights than malicious
            honest_weights = [weights[c] for c in clients if "honest" in c]
            malicious_weight = weights["malicious_1"]
            
            assert all(hw >= malicious_weight for hw in honest_weights)
            
            # Test client filtering
            allowed_clients = orchestrator.filter_clients(clients)
            
            # Should filter out quarantined clients
            if malicious_quarantined:
                assert "malicious_1" not in allowed_clients
            
            print(f"Final statistics: {stats}")
            print(f"Final weights: {weights}")
            print(f"Allowed clients: {allowed_clients}")
            
        finally:
            # Clean up
            try:
                os.unlink(db_path)
            except:
                pass
    
    def test_reputation_persistence_and_recovery(self):
        """Test reputation persistence and recovery scenarios."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp_file:
            db_path = tmp_file.name
        
        try:
            # Phase 1: Build reputation
            manager1 = ClientReputationManager(db_path=db_path)
            
            client_id = "recovery_test_client"
            
            # Start with good behavior
            for i in range(5):
                manager1.update_reputation(client_id, 0.4)
            
            good_reputation = manager1.get_reputation(client_id)
            
            # Then bad behavior
            for i in range(3):
                manager1.update_reputation(client_id, -0.6)
            
            bad_reputation = manager1.get_reputation(client_id)
            assert bad_reputation < good_reputation
            
            # Phase 2: Restart system (test persistence)
            manager2 = ClientReputationManager(db_path=db_path)
            
            # Should load previous state
            loaded_reputation = manager2.get_reputation(client_id)
            assert abs(loaded_reputation - bad_reputation) < 0.001
            
            # Phase 3: Recovery
            for i in range(5):
                manager2.update_reputation(client_id, 0.5)
            
            recovered_reputation = manager2.get_reputation(client_id)
            assert recovered_reputation > bad_reputation
            
            # Verify client info consistency
            client_info = manager2.get_client_info(client_id)
            assert client_info['total_updates'] == 13  # 5 + 3 + 5
            assert client_info['anomalous_updates'] == 3
            
        finally:
            try:
                os.unlink(db_path)
            except:
                pass


if __name__ == "__main__":
    pytest.main([__file__])