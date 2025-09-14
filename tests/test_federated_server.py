"""
Tests for Secure Federated Learning Server

Integration tests for the complete federated learning workflow with security.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock

from federated_learning.server import (
    SecureFederatedServer,
    ClientManager,
    TrainingOrchestrator,
    ServerState,
    TrainingRoundState
)
from federated_learning.model_aggregator import FederatedAveragingAggregator
from anomaly_detection.interfaces import ModelUpdate, IAnomalyDetector, IReputationManager
from auth.interfaces import IAuthenticationService
from pq_security.interfaces import IPQCrypto


class TestSecureFederatedServer:
    """Test cases for SecureFederatedServer."""
    
    @pytest.fixture
    def mock_auth_service(self):
        """Mock authentication service."""
        mock = Mock(spec=IAuthenticationService)
        mock.authenticate_client.return_value = True
        mock.is_client_valid.return_value = True
        return mock
    
    @pytest.fixture
    def mock_pq_crypto(self):
        """Mock post-quantum crypto manager."""
        return Mock(spec=IPQCrypto)
    
    @pytest.fixture
    def mock_anomaly_detector(self):
        """Mock anomaly detector."""
        mock = Mock(spec=IAnomalyDetector)
        mock.predict_anomaly_score.return_value = 0.3
        return mock
    
    @pytest.fixture
    def mock_reputation_manager(self):
        """Mock reputation manager."""
        mock = Mock(spec=IReputationManager)
        mock.is_quarantined.return_value = False
        mock.get_influence_weight.return_value = 1.0
        return mock
    
    @pytest.fixture
    def aggregator(self):
        """Create aggregator for testing."""
        return FederatedAveragingAggregator()
    
    @pytest.fixture
    def server(self, mock_auth_service, mock_pq_crypto, aggregator, 
               mock_anomaly_detector, mock_reputation_manager):
        """Create SecureFederatedServer instance."""
        return SecureFederatedServer(
            auth_service=mock_auth_service,
            pq_crypto=mock_pq_crypto,
            aggregator=aggregator,
            anomaly_detector=mock_anomaly_detector,
            reputation_manager=mock_reputation_manager
        )
    
    @pytest.fixture
    def sample_update(self):
        """Sample model update for testing."""
        return ModelUpdate(
            client_id="client_001",
            round_id="test_round",
            weights={'layer1': np.array([1.0, 2.0], dtype=np.float32)},
            signature=b"test_signature",
            timestamp=datetime.now(),
            metadata={}
        )
    
    def test_initialization(self, server, mock_auth_service, mock_pq_crypto, aggregator):
        """Test server initialization."""
        assert server.auth_service == mock_auth_service
        assert server.pq_crypto == mock_pq_crypto
        assert server.aggregator == aggregator
        assert server.state == ServerState.IDLE
        assert server.current_round is None
        assert server.global_model is None
    
    def test_start_training_round(self, server):
        """Test starting a training round."""
        round_id = server.start_training_round()
        
        assert isinstance(round_id, str)
        assert server.state == ServerState.TRAINING_ROUND_ACTIVE
        assert server.current_round is not None
        assert server.current_round.round_id == round_id
        assert server.current_round.participants == []
    
    def test_start_training_round_when_not_idle(self, server):
        """Test error when starting round in non-idle state."""
        server.state = ServerState.AGGREGATING
        
        with pytest.raises(RuntimeError, match="Cannot start training round"):
            server.start_training_round()
    
    def test_receive_client_update_success(self, server, sample_update):
        """Test successful client update reception."""
        # Start training round
        round_id = server.start_training_round()
        sample_update.round_id = round_id
        
        result = server.receive_client_update("client_001", sample_update)
        
        assert result is True
        assert "client_001" in server.current_round.participants
        assert len(server.current_round.security_events) > 0
    
    def test_receive_client_update_wrong_state(self, server, sample_update):
        """Test client update rejection when server not in training state."""
        result = server.receive_client_update("client_001", sample_update)
        
        assert result is False
    
    def test_receive_client_update_client_id_mismatch(self, server, sample_update):
        """Test client update rejection with ID mismatch."""
        round_id = server.start_training_round()
        sample_update.round_id = round_id
        
        result = server.receive_client_update("different_client", sample_update)
        
        assert result is False
    
    def test_receive_client_update_round_id_mismatch(self, server, sample_update):
        """Test client update rejection with round ID mismatch."""
        server.start_training_round()
        sample_update.round_id = "wrong_round"
        
        result = server.receive_client_update("client_001", sample_update)
        
        assert result is False
    
    def test_aggregate_updates_success(self, server):
        """Test successful update aggregation."""
        # Start training round
        round_id = server.start_training_round()
        
        # Create and submit multiple updates
        updates = []
        for i in range(3):
            update = ModelUpdate(
                client_id=f"client_{i:03d}",
                round_id=round_id,
                weights={'layer1': np.array([float(i), float(i+1)], dtype=np.float32)},
                signature=b"signature",
                timestamp=datetime.now(),
                metadata={}
            )
            updates.append(update)
            server.receive_client_update(update.client_id, update)
        
        # Aggregate updates
        global_model = server.aggregate_updates(round_id)
        
        assert global_model is not None
        assert global_model.round_id == round_id
        assert 'layer1' in global_model.weights
        assert server.state == ServerState.IDLE
        assert server.current_round.completed_at is not None
    
    def test_aggregate_updates_insufficient_clients(self, server):
        """Test aggregation failure with insufficient clients."""
        round_id = server.start_training_round()
        
        # Submit only one update (less than min_clients_per_round = 2)
        update = ModelUpdate(
            client_id="client_001",
            round_id=round_id,
            weights={'layer1': np.array([1.0, 2.0], dtype=np.float32)},
            signature=b"signature",
            timestamp=datetime.now(),
            metadata={}
        )
        server.receive_client_update(update.client_id, update)
        
        with pytest.raises(RuntimeError, match="Insufficient updates"):
            server.aggregate_updates(round_id)
    
    def test_aggregate_updates_wrong_round(self, server):
        """Test aggregation failure with wrong round ID."""
        server.start_training_round()
        
        with pytest.raises(RuntimeError, match="No active round"):
            server.aggregate_updates("wrong_round_id")
    
    def test_distribute_global_model(self, server):
        """Test global model distribution."""
        # Create a global model
        from federated_learning.interfaces import GlobalModel
        
        model = GlobalModel(
            model_id="test_model",
            round_id="test_round",
            weights={'layer1': np.array([1.0, 2.0])},
            metadata={},
            created_at=datetime.now()
        )
        
        # Should not raise an exception
        server.distribute_global_model(model)
        assert server.state == ServerState.IDLE
    
    def test_get_current_model(self, server):
        """Test getting current model."""
        assert server.get_current_model() is None
        
        # Initialize a model
        weights = {'layer1': np.array([1.0, 2.0])}
        model = server.initialize_global_model(weights)
        
        current = server.get_current_model()
        assert current == model
        assert current.model_id == model.model_id
    
    def test_get_server_status(self, server):
        """Test getting server status."""
        status = server.get_server_status()
        
        assert 'state' in status
        assert 'current_round_id' in status
        assert 'total_rounds' in status
        assert 'current_model_id' in status
        assert 'active_participants' in status
        assert 'server_uptime' in status
        
        assert status['state'] == 'idle'
        assert status['current_round_id'] is None
        assert status['total_rounds'] == 0
    
    def test_set_configuration(self, server):
        """Test setting server configuration."""
        config = {
            'round_timeout_minutes': 60,
            'min_clients_per_round': 5,
            'max_clients_per_round': 50
        }
        
        server.set_configuration(config)
        
        assert server.round_timeout == timedelta(minutes=60)
        assert server.min_clients_per_round == 5
        assert server.max_clients_per_round == 50
    
    def test_initialize_global_model(self, server):
        """Test global model initialization."""
        weights = {
            'layer1': np.array([[1.0, 2.0], [3.0, 4.0]]),
            'layer2': np.array([0.5, 1.5])
        }
        
        model = server.initialize_global_model(weights)
        
        assert model is not None
        assert model.round_id == "initial"
        assert len(model.weights) == 2
        assert server.global_model == model
        assert len(server.model_history) == 1
    
    def test_training_history(self, server):
        """Test training history tracking."""
        assert len(server.get_training_history()) == 0
        
        # Complete a training round
        round_id = server.start_training_round()
        
        # Add multiple updates
        for i in range(3):
            update = ModelUpdate(
                client_id=f"client_{i:03d}",
                round_id=round_id,
                weights={'layer1': np.array([float(i), float(i+1)])},
                signature=b"signature",
                timestamp=datetime.now(),
                metadata={}
            )
            server.receive_client_update(update.client_id, update)
        
        server.aggregate_updates(round_id)
        
        history = server.get_training_history()
        assert len(history) == 1
        assert history[0].round_id == round_id
    
    def test_model_history(self, server):
        """Test model history tracking."""
        assert len(server.get_model_history()) == 0
        
        # Initialize model
        weights = {'layer1': np.array([1.0, 2.0])}
        server.initialize_global_model(weights)
        
        assert len(server.get_model_history()) == 1
    
    def test_complete_training_workflow(self, server):
        """Test complete training workflow."""
        # Initialize global model
        initial_weights = {'layer1': np.array([0.0, 0.0], dtype=np.float32)}
        server.initialize_global_model(initial_weights)
        
        # Start training round
        round_id = server.start_training_round()
        
        # Submit client updates
        client_updates = [
            ModelUpdate(
                client_id="client_001",
                round_id=round_id,
                weights={'layer1': np.array([1.0, 2.0], dtype=np.float32)},
                signature=b"sig1",
                timestamp=datetime.now(),
                metadata={}
            ),
            ModelUpdate(
                client_id="client_002",
                round_id=round_id,
                weights={'layer1': np.array([3.0, 4.0], dtype=np.float32)},
                signature=b"sig2",
                timestamp=datetime.now(),
                metadata={}
            )
        ]
        
        for update in client_updates:
            result = server.receive_client_update(update.client_id, update)
            assert result is True
        
        # Aggregate updates
        global_model = server.aggregate_updates(round_id)
        
        # Verify aggregated model
        expected_weights = np.array([2.0, 3.0], dtype=np.float32)  # Average of [1,2] and [3,4]
        np.testing.assert_array_equal(global_model.weights['layer1'], expected_weights)
        
        # Distribute model
        server.distribute_global_model(global_model)
        
        # Verify final state
        assert server.state == ServerState.IDLE
        assert server.current_round is None
        assert len(server.get_training_history()) == 1
        assert len(server.get_model_history()) == 2  # Initial + aggregated


class TestClientManager:
    """Test cases for ClientManager."""
    
    @pytest.fixture
    def mock_auth_service(self):
        """Mock authentication service."""
        mock = Mock(spec=IAuthenticationService)
        mock.is_client_valid.return_value = True
        return mock
    
    @pytest.fixture
    def client_manager(self, mock_auth_service):
        """Create ClientManager instance."""
        return ClientManager(mock_auth_service)
    
    def test_register_client(self, client_manager):
        """Test client registration."""
        client_manager.register_client("client_001")
        
        assert "client_001" in client_manager.active_clients
        assert client_manager.is_client_active("client_001")
        assert "client_001" in client_manager.get_active_clients()
    
    def test_register_invalid_client(self, client_manager, mock_auth_service):
        """Test registration of invalid client."""
        mock_auth_service.is_client_valid.return_value = False
        
        with pytest.raises(ValueError, match="not authenticated"):
            client_manager.register_client("invalid_client")
    
    def test_register_duplicate_client(self, client_manager):
        """Test duplicate client registration."""
        client_manager.register_client("client_001")
        
        # Should not raise an error, just log a warning
        client_manager.register_client("client_001")
        
        assert len(client_manager.get_active_clients()) == 1
    
    def test_update_client_status(self, client_manager):
        """Test updating client status."""
        client_manager.register_client("client_001")
        
        client_manager.update_client_status("client_001", "inactive")
        
        assert not client_manager.is_client_active("client_001")
        assert "client_001" not in client_manager.get_active_clients()
    
    def test_update_unregistered_client_status(self, client_manager):
        """Test updating status of unregistered client."""
        # Should not raise an error, just log a warning
        client_manager.update_client_status("unknown_client", "active")
        
        assert not client_manager.is_client_active("unknown_client")


class TestTrainingOrchestrator:
    """Test cases for TrainingOrchestrator."""
    
    @pytest.fixture
    def mock_server(self):
        """Mock federated learning server."""
        mock = Mock(spec=SecureFederatedServer)
        mock.start_training_round.return_value = "test_round_id"
        
        # Mock training round
        from federated_learning.interfaces import TrainingRound
        mock_round = TrainingRound(
            round_id="test_round_id",
            participants=["client_001", "client_002"],
            global_model_hash="test_hash",
            aggregation_method="FederatedAveraging",
            security_events=[],
            metrics={'num_participants': 2},
            started_at=datetime.now(),
            completed_at=datetime.now()
        )
        mock.get_training_history.return_value = [mock_round]
        
        return mock
    
    @pytest.fixture
    def mock_client_manager(self):
        """Mock client manager."""
        return Mock(spec=ClientManager)
    
    @pytest.fixture
    def orchestrator(self, mock_server, mock_client_manager):
        """Create TrainingOrchestrator instance."""
        return TrainingOrchestrator(mock_server, mock_client_manager)
    
    def test_orchestrate_round(self, orchestrator, mock_server):
        """Test training round orchestration."""
        from federated_learning.interfaces import GlobalModel
        
        # Mock global model
        mock_model = GlobalModel(
            model_id="test_model",
            round_id="test_round_id",
            weights={'layer1': np.array([1.0, 2.0])},
            metadata={},
            created_at=datetime.now()
        )
        mock_server.aggregate_updates.return_value = mock_model
        
        result = orchestrator.orchestrate_round("test_round")
        
        assert result is not None
        assert result.round_id == "test_round_id"
        
        # Verify server methods were called
        mock_server.start_training_round.assert_called_once()
        mock_server.aggregate_updates.assert_called_once()
        mock_server.distribute_global_model.assert_called_once_with(mock_model)
    
    def test_get_round_status_active(self, orchestrator, mock_server):
        """Test getting status of active round."""
        from federated_learning.interfaces import TrainingRound
        
        mock_round = TrainingRound(
            round_id="active_round",
            participants=[],
            global_model_hash="",
            aggregation_method="",
            security_events=[],
            metrics={},
            started_at=datetime.now()
        )
        mock_server.current_round = mock_round
        
        status = orchestrator.get_round_status("active_round")
        assert status == "active"
    
    def test_get_round_status_completed(self, orchestrator, mock_server):
        """Test getting status of completed round."""
        mock_server.current_round = None  # No active round
        status = orchestrator.get_round_status("test_round_id")
        assert status == "completed"
    
    def test_get_round_status_not_found(self, orchestrator, mock_server):
        """Test getting status of non-existent round."""
        mock_server.current_round = None
        mock_server.get_training_history.return_value = []
        
        status = orchestrator.get_round_status("unknown_round")
        assert status == "not_found"
    
    def test_get_round_metrics(self, orchestrator, mock_server):
        """Test getting round metrics."""
        mock_server.current_round = None  # No active round
        metrics = orchestrator.get_round_metrics("test_round_id")
        
        assert isinstance(metrics, dict)
        assert 'num_participants' in metrics
        assert metrics['num_participants'] == 2


if __name__ == "__main__":
    pytest.main([__file__])