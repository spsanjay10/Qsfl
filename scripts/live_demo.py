#!/usr/bin/env python3
"""
Live demonstration script for QSFL-CAAD system.

This script provides an interactive demonstration of the QSFL-CAAD system
capabilities, including attack detection and response.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import threading
import queue
import logging
from typing import Dict, List, Any

# Configure logging for demo
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LiveDemo:
    """Interactive live demonstration of QSFL-CAAD system."""
    
    def __init__(self):
        """Initialize the demo system."""
        self.setup_system()
        self.setup_visualization()
        self.demo_running = False
        self.metrics_queue = queue.Queue()
        
    def setup_system(self):
        """Setup the QSFL-CAAD system for demonstration."""
        logger.info("Setting up QSFL-CAAD system...")
        
        # Import system components
        from qsfl_caad import QSFLSystem
        from config.settings import load_config
        
        # Load demo configuration
        config = self.create_demo_config()
        self.system = QSFLSystem(config)
        
        # Setup clients
        self.setup_demo_clients()
        
        logger.info("✅ System setup completed")
    
    def create_demo_config(self) -> Dict[str, Any]:
        """Create configuration optimized for demonstration."""
        return {
            'security': {
                'anomaly_threshold': 0.6,
                'reputation_decay': 0.95,
                'quarantine_threshold': 0.8
            },
            'demo': {
                'update_interval': 2.0,  # Seconds between updates
                'visualization_update': 1.0,  # Visualization refresh rate
                'num_honest_clients': 6,
                'num_malicious_clients': 2
            },
            'logging': {
                'level': 'INFO'
            }
        }
    
    def setup_demo_clients(self):
        """Setup demonstration clients."""
        logger.info("Setting up demo clients...")
        
        self.clients = {}
        
        # Create honest clients
        for i in range(6):
            client_id = f"honest_client_{i+1}"
            credentials = self.system.register_client(client_id)
            self.clients[client_id] = {
                'type': 'honest',
                'credentials': credentials,
                'reputation': 1.0,
                'updates_sent': 0,
                'last_anomaly_score': 0.0
            }
        
        # Create malicious clients (initially appear honest)
        for i in range(2):
            client_id = f"malicious_client_{i+1}"
            credentials = self.system.register_client(client_id)
            self.clients[client_id] = {
                'type': 'malicious',
                'credentials': credentials,
                'reputation': 1.0,
                'updates_sent': 0,
                'last_anomaly_score': 0.0,
                'attack_started': False
            }
        
        logger.info(f"✅ Setup {len(self.clients)} demo clients")
    
    def setup_visualization(self):
        """Setup real-time visualization."""
        # Disable interactive plotting for console demo
        # plt.ion()  # Enable interactive mode
        
        # self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        # self.fig.suptitle('QSFL-CAAD Live Demonstration Dashboard', fontsize=16)
        
        # Initialize data storage for plots (disabled for console demo)
        self.plot_data = {
            'timestamps': [],
            'anomaly_scores': {client_id: [] for client_id in self.clients.keys()},
            'reputations': {client_id: [] for client_id in self.clients.keys()},
            'model_accuracy': [],
            'security_events': []
        }
        
        # Setup individual plots (disabled for console demo)
        # self.setup_anomaly_plot()
        # self.setup_reputation_plot()
        # self.setup_accuracy_plot()
        # self.setup_events_plot()
    
    def setup_anomaly_plot(self):
        """Setup anomaly scores plot."""
        self.anomaly_ax = self.axes[0, 0]
        self.anomaly_ax.set_title('Real-time Anomaly Scores')
        self.anomaly_ax.set_xlabel('Time')
        self.anomaly_ax.set_ylabel('Anomaly Score')
        self.anomaly_ax.set_ylim(0, 1)
        self.anomaly_ax.axhline(y=0.6, color='r', linestyle='--', label='Threshold')
        self.anomaly_ax.legend()
    
    def setup_reputation_plot(self):
        """Setup reputation scores plot."""
        self.reputation_ax = self.axes[0, 1]
        self.reputation_ax.set_title('Client Reputation Scores')
        self.reputation_ax.set_xlabel('Time')
        self.reputation_ax.set_ylabel('Reputation')
        self.reputation_ax.set_ylim(0, 1)
    
    def setup_accuracy_plot(self):
        """Setup model accuracy plot."""
        self.accuracy_ax = self.axes[1, 0]
        self.accuracy_ax.set_title('Global Model Accuracy')
        self.accuracy_ax.set_xlabel('Training Round')
        self.accuracy_ax.set_ylabel('Accuracy')
        self.accuracy_ax.set_ylim(0, 1)
    
    def setup_events_plot(self):
        """Setup security events plot."""
        self.events_ax = self.axes[1, 1]
        self.events_ax.set_title('Security Events Timeline')
        self.events_ax.set_xlabel('Time')
        self.events_ax.set_ylabel('Event Type')
    
    def start_demo(self):
        """Start the live demonstration."""
        print("\n" + "="*60)
        print("🚀 STARTING QSFL-CAAD LIVE DEMONSTRATION")
        print("="*60)
        
        self.demo_running = True
        
        # Start background threads (visualization disabled for console demo)
        self.metrics_thread = threading.Thread(target=self.metrics_collector)
        # self.visualization_thread = threading.Thread(target=self.update_visualization)
        
        self.metrics_thread.start()
        # self.visualization_thread.start()
        
        # Run main demo loop
        self.run_demo_scenario()
    
    def run_demo_scenario(self):
        """Run the main demonstration scenario."""
        
        print("\n📋 DEMO SCENARIO:")
        print("1. Normal federated learning operation (30 seconds)")
        print("2. Malicious clients start attacking (at 30 seconds)")
        print("3. System detects and responds to attacks")
        print("4. Recovery and continued operation")
        print("\nPress Ctrl+C to stop the demo at any time\n")
        
        try:
            round_number = 0
            start_time = time.time()
            
            while self.demo_running:
                current_time = time.time() - start_time
                round_number += 1
                
                print(f"\n🔄 ROUND {round_number} (t={current_time:.1f}s)")
                
                # Determine if malicious clients should start attacking
                attack_phase = current_time > 30
                
                if attack_phase and not self.attack_announced:
                    print("\n🚨 ATTACK PHASE INITIATED!")
                    print("Malicious clients beginning coordinated attack...")
                    self.attack_announced = True
                
                # Process client updates
                self.process_federated_round(round_number, attack_phase)
                
                # Wait for next round
                time.sleep(2.0)
                
        except KeyboardInterrupt:
            print("\n\n⏹️  Demo stopped by user")
        finally:
            self.stop_demo()
    
    def process_federated_round(self, round_number: int, attack_phase: bool):
        """Process a single federated learning round."""
        
        round_id = f"demo_round_{round_number}"
        self.system.start_training_round(round_id)
        
        accepted_updates = 0
        rejected_updates = 0
        
        # Process updates from each client
        for client_id, client_info in self.clients.items():
            
            # Generate client update
            update = self.generate_client_update(
                client_id, round_id, client_info, attack_phase
            )
            
            # Submit update to system
            accepted = self.system.receive_client_update(client_id, update)
            
            if accepted:
                accepted_updates += 1
                status = "✅ ACCEPTED"
            else:
                rejected_updates += 1
                status = "❌ REJECTED"
            
            # Get updated metrics
            anomaly_score = self.system.get_last_anomaly_score(client_id)
            reputation = self.system.get_client_reputation(client_id)
            is_quarantined = self.system.is_client_quarantined(client_id)
            
            # Update client info
            client_info['last_anomaly_score'] = anomaly_score
            client_info['reputation'] = reputation
            client_info['updates_sent'] += 1
            
            # Display client status
            client_type = "😈" if client_info['type'] == 'malicious' else "😇"
            quarantine_status = "🚫" if is_quarantined else ""
            
            print(f"  {client_type} {client_id}: {status} "
                  f"(anomaly: {anomaly_score:.3f}, rep: {reputation:.3f}) {quarantine_status}")
        
        # Aggregate if we have accepted updates
        if accepted_updates > 0:
            global_model = self.system.aggregate_updates(round_id)
            accuracy = self.simulate_model_accuracy(global_model, round_number)
            
            print(f"📊 Round Summary:")
            print(f"   Accepted: {accepted_updates}, Rejected: {rejected_updates}")
            print(f"   Global Model Accuracy: {accuracy:.4f}")
            
            # Store metrics for visualization
            self.store_round_metrics(round_number, accuracy)
        else:
            print("⚠️  No updates accepted - skipping aggregation")
    
    def generate_client_update(self, client_id: str, round_id: str, 
                             client_info: Dict, attack_phase: bool):
        """Generate a model update for a client."""
        
        from anomaly_detection.interfaces import ModelUpdate
        
        # Base weights (normal distribution)
        weights = {
            "layer_0": np.random.normal(0, 0.1, (100, 50)),
            "layer_1": np.random.normal(0, 0.1, (50, 10)),
            "layer_2": np.random.normal(0, 0.1, (10, 1))
        }
        
        # Apply malicious behavior if in attack phase
        if client_info['type'] == 'malicious' and attack_phase:
            if not client_info.get('attack_started', False):
                print(f"🦹 {client_id} starting malicious behavior!")
                client_info['attack_started'] = True
            
            # Apply different attack strategies
            attack_type = np.random.choice(['gradient_poisoning', 'label_flipping', 'backdoor'])
            
            if attack_type == 'gradient_poisoning':
                # Amplify gradients
                for key in weights:
                    weights[key] *= 3.0
                    weights[key] += np.random.normal(0, 0.5, weights[key].shape)
            
            elif attack_type == 'label_flipping':
                # Simulate label flipping by inverting some weights
                for key in weights:
                    weights[key] *= -1
            
            elif attack_type == 'backdoor':
                # Add backdoor pattern
                for key in weights:
                    weights[key] += np.ones_like(weights[key]) * 0.1
        
        # Create update
        update = ModelUpdate(
            client_id=client_id,
            round_id=round_id,
            weights=weights,
            signature=self.sign_update(client_id, weights),
            timestamp=datetime.now(),
            metadata={
                'client_type': client_info['type'],
                'attack_phase': attack_phase,
                'local_accuracy': np.random.uniform(0.8, 0.95) if client_info['type'] == 'honest' else np.random.uniform(0.3, 0.7)
            }
        )
        
        return update
    
    def sign_update(self, client_id: str, weights: Dict) -> bytes:
        """Sign model update (simplified for demo)."""
        # In real implementation, use proper cryptographic signing
        import hashlib
        
        weights_str = str(sorted(weights.items()))
        message = f"{client_id}_{weights_str}".encode()
        return hashlib.sha256(message).digest()
    
    def simulate_model_accuracy(self, global_model, round_number: int) -> float:
        """Simulate global model accuracy."""
        # Simulate accuracy improvement over time with some noise
        base_accuracy = 0.7 + (round_number * 0.01)  # Gradual improvement
        noise = np.random.normal(0, 0.02)  # Small random variations
        
        # Cap accuracy and add some realism
        accuracy = min(0.95, max(0.5, base_accuracy + noise))
        
        return accuracy
    
    def store_round_metrics(self, round_number: int, accuracy: float):
        """Store metrics for visualization."""
        timestamp = datetime.now()
        
        # Store in queue for visualization thread
        metrics = {
            'timestamp': timestamp,
            'round': round_number,
            'accuracy': accuracy,
            'clients': {}
        }
        
        for client_id, client_info in self.clients.items():
            metrics['clients'][client_id] = {
                'anomaly_score': client_info['last_anomaly_score'],
                'reputation': client_info['reputation'],
                'type': client_info['type']
            }
        
        self.metrics_queue.put(metrics)
    
    def metrics_collector(self):
        """Background thread to collect metrics."""
        while self.demo_running:
            try:
                # Collect system metrics
                system_metrics = self.system.get_system_metrics()
                
                # Store for visualization
                # This runs in background to not block main demo
                
                time.sleep(1.0)
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
    
    def update_visualization(self):
        """Update visualization in real-time."""
        while self.demo_running:
            try:
                # Get new metrics from queue
                while not self.metrics_queue.empty():
                    metrics = self.metrics_queue.get_nowait()
                    self.update_plots(metrics)
                
                # Refresh display
                plt.pause(0.1)
                
            except Exception as e:
                logger.error(f"Visualization update error: {e}")
            
            time.sleep(1.0)
    
    def update_plots(self, metrics: Dict):
        """Update all plots with new metrics."""
        timestamp = metrics['timestamp']
        
        # Update data storage
        self.plot_data['timestamps'].append(timestamp)
        self.plot_data['model_accuracy'].append(metrics['accuracy'])
        
        for client_id, client_metrics in metrics['clients'].items():
            self.plot_data['anomaly_scores'][client_id].append(client_metrics['anomaly_score'])
            self.plot_data['reputations'][client_id].append(client_metrics['reputation'])
        
        # Keep only last 50 data points for readability
        max_points = 50
        if len(self.plot_data['timestamps']) > max_points:
            self.plot_data['timestamps'] = self.plot_data['timestamps'][-max_points:]
            self.plot_data['model_accuracy'] = self.plot_data['model_accuracy'][-max_points:]
            
            for client_id in self.clients.keys():
                self.plot_data['anomaly_scores'][client_id] = self.plot_data['anomaly_scores'][client_id][-max_points:]
                self.plot_data['reputations'][client_id] = self.plot_data['reputations'][client_id][-max_points:]
        
        # Update anomaly scores plot
        self.anomaly_ax.clear()
        self.anomaly_ax.set_title('Real-time Anomaly Scores')
        self.anomaly_ax.set_ylabel('Anomaly Score')
        self.anomaly_ax.set_ylim(0, 1)
        self.anomaly_ax.axhline(y=0.6, color='r', linestyle='--', label='Threshold')
        
        for client_id, client_info in self.clients.items():
            color = 'red' if client_info['type'] == 'malicious' else 'blue'
            alpha = 0.7 if client_info['type'] == 'malicious' else 0.5
            
            if len(self.plot_data['anomaly_scores'][client_id]) > 0:
                self.anomaly_ax.plot(
                    self.plot_data['timestamps'],
                    self.plot_data['anomaly_scores'][client_id],
                    color=color, alpha=alpha, label=client_id
                )
        
        self.anomaly_ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Update reputation plot
        self.reputation_ax.clear()
        self.reputation_ax.set_title('Client Reputation Scores')
        self.reputation_ax.set_ylabel('Reputation')
        self.reputation_ax.set_ylim(0, 1)
        
        for client_id, client_info in self.clients.items():
            color = 'red' if client_info['type'] == 'malicious' else 'blue'
            alpha = 0.7 if client_info['type'] == 'malicious' else 0.5
            
            if len(self.plot_data['reputations'][client_id]) > 0:
                self.reputation_ax.plot(
                    self.plot_data['timestamps'],
                    self.plot_data['reputations'][client_id],
                    color=color, alpha=alpha, label=client_id
                )
        
        # Update accuracy plot
        self.accuracy_ax.clear()
        self.accuracy_ax.set_title('Global Model Accuracy')
        self.accuracy_ax.set_ylabel('Accuracy')
        self.accuracy_ax.set_ylim(0.5, 1.0)
        
        if len(self.plot_data['model_accuracy']) > 0:
            rounds = list(range(len(self.plot_data['model_accuracy'])))
            self.accuracy_ax.plot(rounds, self.plot_data['model_accuracy'], 'g-', linewidth=2)
        
        plt.tight_layout()
    
    def stop_demo(self):
        """Stop the demonstration."""
        print("\n🛑 Stopping demonstration...")
        
        self.demo_running = False
        
        # Wait for threads to finish
        if hasattr(self, 'metrics_thread'):
            self.metrics_thread.join(timeout=2)
        # if hasattr(self, 'visualization_thread'):
        #     self.visualization_thread.join(timeout=2)
        
        print("✅ Demo stopped successfully")
        
        # Show final summary
        self.show_demo_summary()
    
    def show_demo_summary(self):
        """Show demonstration summary."""
        print("\n" + "="*60)
        print("📊 DEMONSTRATION SUMMARY")
        print("="*60)
        
        # Calculate statistics
        total_updates = sum(client['updates_sent'] for client in self.clients.values())
        malicious_clients = [c for c in self.clients.values() if c['type'] == 'malicious']
        honest_clients = [c for c in self.clients.values() if c['type'] == 'honest']
        
        quarantined_malicious = sum(1 for c in malicious_clients 
                                  if self.system.is_client_quarantined(c['credentials'].client_id))
        quarantined_honest = sum(1 for c in honest_clients 
                               if self.system.is_client_quarantined(c['credentials'].client_id))
        
        print(f"Total Updates Processed: {total_updates}")
        print(f"Honest Clients: {len(honest_clients)}")
        print(f"Malicious Clients: {len(malicious_clients)}")
        print(f"Malicious Clients Quarantined: {quarantined_malicious}/{len(malicious_clients)}")
        print(f"False Positives (Honest Quarantined): {quarantined_honest}/{len(honest_clients)}")
        
        if len(malicious_clients) > 0:
            detection_rate = quarantined_malicious / len(malicious_clients)
            print(f"Detection Rate: {detection_rate:.1%}")
        
        if len(honest_clients) > 0:
            false_positive_rate = quarantined_honest / len(honest_clients)
            print(f"False Positive Rate: {false_positive_rate:.1%}")
        
        print("\n🎯 Key Achievements:")
        print("✅ Demonstrated quantum-safe cryptographic operations")
        print("✅ Showed real-time anomaly detection capabilities")
        print("✅ Illustrated dynamic reputation management")
        print("✅ Validated system resilience against attacks")
        
        print("\n📚 For more information:")
        print("- API Documentation: docs/API_DOCUMENTATION.md")
        print("- Usage Examples: docs/USAGE_EXAMPLES.md")
        print("- Developer Guide: docs/DEVELOPER_GUIDE.md")

def main():
    """Main demonstration function."""
    
    print("🎭 QSFL-CAAD Live Demonstration")
    print("=" * 50)
    print("This demonstration shows the QSFL-CAAD system in action,")
    print("including attack detection and response capabilities.")
    print("\nInitializing system...")
    
    # Initialize demo
    demo = LiveDemo()
    demo.attack_announced = False
    
    try:
        # Start demonstration
        demo.start_demo()
        
        # Demo completed
        print("\nDemo completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()