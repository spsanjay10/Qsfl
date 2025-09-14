"""
Example Integration of Federated Learning Core

Demonstrates how to use the secure federated learning components together.
This is a simple example showing the complete workflow.
"""

import numpy as np
from datetime import datetime
from unittest.mock import Mock

from federated_learning import (
    SecureFederatedServer,
    ClientManager,
    TrainingOrchestrator,
    FederatedAveragingAggregator
)
from anomaly_detection.interfaces import ModelUpdate
from auth.interfaces import IAuthenticationService
from pq_security.interfaces import IPQCrypto


def create_mock_services():
    """Create mock services for demonstration."""
    # Mock authentication service
    auth_service = Mock(spec=IAuthenticationService)
    auth_service.authenticate_client.return_value = True
    auth_service.is_client_valid.return_value = True
    
    # Mock post-quantum crypto
    pq_crypto = Mock(spec=IPQCrypto)
    
    return auth_service, pq_crypto


def create_sample_update(client_id: str, round_id: str, weights_offset: float = 0.0):
    """Create a sample model update."""
    weights = {
        'layer1': np.array([[1.0 + weights_offset, 2.0 + weights_offset], 
                           [3.0 + weights_offset, 4.0 + weights_offset]], dtype=np.float32),
        'layer2': np.array([0.5 + weights_offset, 1.5 + weights_offset], dtype=np.float32)
    }
    
    return ModelUpdate(
        client_id=client_id,
        round_id=round_id,
        weights=weights,
        signature=b"mock_signature",
        timestamp=datetime.now(),
        metadata={'client_version': '1.0'}
    )


def demonstrate_federated_learning():
    """Demonstrate complete federated learning workflow."""
    print("=== Secure Federated Learning Integration Demo ===\n")
    
    # 1. Setup services
    print("1. Setting up services...")
    auth_service, pq_crypto = create_mock_services()
    aggregator = FederatedAveragingAggregator()
    
    # 2. Create server
    print("2. Creating secure federated server...")
    server = SecureFederatedServer(
        auth_service=auth_service,
        pq_crypto=pq_crypto,
        aggregator=aggregator
    )
    
    # 3. Initialize global model
    print("3. Initializing global model...")
    initial_weights = {
        'layer1': np.zeros((2, 2), dtype=np.float32),
        'layer2': np.zeros(2, dtype=np.float32)
    }
    global_model = server.initialize_global_model(initial_weights)
    print(f"   Initial model ID: {global_model.model_id}")
    
    # 4. Create client manager
    print("4. Setting up client manager...")
    client_manager = ClientManager(auth_service)
    
    # Register clients
    clients = ["client_001", "client_002", "client_003"]
    for client_id in clients:
        client_manager.register_client(client_id)
    
    print(f"   Registered clients: {client_manager.get_active_clients()}")
    
    # 5. Start training round
    print("\n5. Starting training round...")
    round_id = server.start_training_round()
    print(f"   Round ID: {round_id}")
    print(f"   Server state: {server.get_server_status()['state']}")
    
    # 6. Simulate client updates
    print("\n6. Receiving client updates...")
    for i, client_id in enumerate(clients):
        update = create_sample_update(client_id, round_id, weights_offset=float(i))
        
        success = server.receive_client_update(client_id, update)
        print(f"   {client_id}: {'✓ Accepted' if success else '✗ Rejected'}")
    
    # 7. Aggregate updates
    print("\n7. Aggregating updates...")
    try:
        new_global_model = server.aggregate_updates(round_id)
        print(f"   New global model ID: {new_global_model.model_id}")
        print(f"   Participants: {new_global_model.metadata['num_participants']}")
        
        # Show aggregated weights
        print("   Aggregated weights:")
        for layer_name, weights in new_global_model.weights.items():
            print(f"     {layer_name}: {weights.flatten()}")
        
    except Exception as e:
        print(f"   ✗ Aggregation failed: {e}")
        return
    
    # 8. Distribute model
    print("\n8. Distributing global model...")
    try:
        server.distribute_global_model(new_global_model)
        print("   ✓ Model distributed successfully")
    except Exception as e:
        print(f"   ✗ Distribution failed: {e}")
    
    # 9. Show final statistics
    print("\n9. Final statistics:")
    status = server.get_server_status()
    print(f"   Server state: {status['state']}")
    print(f"   Total rounds completed: {status['total_rounds']}")
    print(f"   Current model: {status['current_model_id']}")
    
    # Show training history
    history = server.get_training_history()
    if history:
        last_round = history[-1]
        print(f"   Last round participants: {len(last_round.participants)}")
        print(f"   Security events: {len(last_round.security_events)}")
    
    print("\n=== Demo completed successfully! ===")


def demonstrate_orchestrator():
    """Demonstrate training orchestrator usage."""
    print("\n=== Training Orchestrator Demo ===\n")
    
    # Setup
    auth_service, pq_crypto = create_mock_services()
    aggregator = FederatedAveragingAggregator()
    
    server = SecureFederatedServer(
        auth_service=auth_service,
        pq_crypto=pq_crypto,
        aggregator=aggregator
    )
    
    client_manager = ClientManager(auth_service)
    orchestrator = TrainingOrchestrator(server, client_manager)
    
    # Initialize model
    initial_weights = {
        'layer1': np.array([[0.0, 0.0]], dtype=np.float32)
    }
    server.initialize_global_model(initial_weights)
    
    print("1. Starting orchestrated training round...")
    
    # In a real scenario, the orchestrator would coordinate the entire process
    # For this demo, we'll simulate the key steps
    try:
        round_id = server.start_training_round()
        
        # Simulate client updates (normally done by clients)
        for i in range(3):
            update = create_sample_update(f"client_{i:03d}", round_id, float(i))
            server.receive_client_update(update.client_id, update)
        
        # Complete the round
        server.aggregate_updates(round_id)
        
        # Check round status
        status = orchestrator.get_round_status(round_id)
        metrics = orchestrator.get_round_metrics(round_id)
        
        print(f"   Round status: {status}")
        print(f"   Round metrics: {metrics}")
        print("   ✓ Orchestrated round completed successfully")
        
    except Exception as e:
        print(f"   ✗ Orchestration failed: {e}")
    
    print("\n=== Orchestrator demo completed! ===")


if __name__ == "__main__":
    # Run the demonstrations
    demonstrate_federated_learning()
    demonstrate_orchestrator()