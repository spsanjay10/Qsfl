#!/usr/bin/env python3
"""
QSFL-CAAD Demo Showcase Script
Comprehensive demonstration script for presentations and showcases
"""

import time
import json
import threading
from datetime import datetime
from typing import Dict, List
import numpy as np

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.text import Text

console = Console()

class DemoShowcase:
    """Interactive demo showcase for QSFL-CAAD."""
    
    def __init__(self):
        self.demo_data = {
            'clients': {},
            'metrics': {
                'timestamps': [],
                'anomaly_scores': {},
                'reputation_scores': {},
                'model_accuracy': [],
                'security_events': []
            },
            'system_status': 'stopped',
            'current_round': 0,
            'showcase_mode': False
        }
        
        self.showcase_steps = [
            {'name': 'System Initialization', 'duration': 3, 'action': self.init_system},
            {'name': 'Client Registration', 'duration': 5, 'action': self.register_clients},
            {'name': 'Training Simulation', 'duration': 8, 'action': self.simulate_training},
            {'name': 'Attack Detection', 'duration': 6, 'action': self.simulate_attack},
            {'name': 'Security Response', 'duration': 4, 'action': self.security_response},
            {'name': 'System Recovery', 'duration': 3, 'action': self.system_recovery},
        ]
        
        self.current_step = 0
        self.is_running = False
    
    def init_system(self):
        """Initialize the QSFL-CAAD system."""
        console.print("🚀 [bold blue]Initializing QSFL-CAAD System[/bold blue]")
        
        self.demo_data['system_status'] = 'initializing'
        time.sleep(1)
        
        console.print("✅ Quantum-safe cryptography modules loaded")
        console.print("✅ Anomaly detection algorithms initialized")
        console.print("✅ Federated learning server started")
        
        self.demo_data['system_status'] = 'ready'
        
    def register_clients(self):
        """Register demo clients."""
        console.print("👥 [bold green]Registering Federated Learning Clients[/bold green]")
        
        clients = [
            {'id': 'hospital_ny', 'type': 'honest', 'location': 'New York, USA'},
            {'id': 'bank_london', 'type': 'honest', 'location': 'London, UK'},
            {'id': 'research_tokyo', 'type': 'honest', 'location': 'Tokyo, Japan'},
            {'id': 'mobile_berlin', 'type': 'suspicious', 'location': 'Berlin, Germany'},
            {'id': 'attacker_001', 'type': 'malicious', 'location': 'Unknown'},
        ]
        
        for client in clients:
            self.demo_data['clients'][client['id']] = {
                'type': client['type'],
                'status': 'active',
                'reputation': 1.0,
                'last_anomaly_score': 0.0,
                'updates_sent': 0,
                'quarantined': False,
                'location': client['location']
            }
            
            color = 'green' if client['type'] == 'honest' else 'yellow' if client['type'] == 'suspicious' else 'red'
            console.print(f"  ✅ [{color}]{client['id']}[/{color}] - {client['location']}")
            time.sleep(0.5)
    
    def simulate_training(self):
        """Simulate federated learning training rounds."""
        console.print("🧠 [bold purple]Starting Federated Learning Training[/bold purple]")
        
        self.demo_data['system_status'] = 'training'
        
        for round_num in range(1, 6):
            self.demo_data['current_round'] = round_num
            timestamp = datetime.now().isoformat()
            self.demo_data['metrics']['timestamps'].append(timestamp)
            
            console.print(f"📊 Training Round {round_num}")
            
            # Simulate client updates
            for client_id, client in self.demo_data['clients'].items():
                if client['quarantined']:
                    continue
                
                # Generate realistic anomaly scores
                if client['type'] == 'honest':
                    anomaly_score = np.random.uniform(0.0, 0.3)
                elif client['type'] == 'suspicious':
                    anomaly_score = np.random.uniform(0.2, 0.5)
                else:  # malicious
                    anomaly_score = np.random.uniform(0.1, 0.4)  # Start low
                
                client['last_anomaly_score'] = anomaly_score
                client['updates_sent'] += 1
                
                # Initialize metrics if not exists
                if client_id not in self.demo_data['metrics']['anomaly_scores']:
                    self.demo_data['metrics']['anomaly_scores'][client_id] = []
                if client_id not in self.demo_data['metrics']['reputation_scores']:
                    self.demo_data['metrics']['reputation_scores'][client_id] = []
                
                self.demo_data['metrics']['anomaly_scores'][client_id].append(anomaly_score)
                self.demo_data['metrics']['reputation_scores'][client_id].append(client['reputation'])
            
            # Simulate model accuracy improvement
            base_accuracy = 0.75 + (round_num * 0.03)
            noise = np.random.normal(0, 0.01)
            accuracy = min(0.95, max(0.7, base_accuracy + noise))
            self.demo_data['metrics']['model_accuracy'].append(accuracy)
            
            console.print(f"  📈 Global Model Accuracy: {accuracy:.3f}")
            time.sleep(1.5)
    
    def simulate_attack(self):
        """Simulate a coordinated attack."""
        console.print("⚠️  [bold red]ATTACK DETECTED![/bold red]")
        console.print("🔍 Malicious clients launching coordinated attack...")
        
        # Increase anomaly scores for malicious clients
        for client_id, client in self.demo_data['clients'].items():
            if client['type'] == 'malicious':
                # Sudden spike in anomaly score
                anomaly_score = np.random.uniform(0.8, 0.95)
                client['last_anomaly_score'] = anomaly_score
                
                # Decrease reputation
                client['reputation'] *= 0.7
                
                # Add to metrics
                self.demo_data['metrics']['anomaly_scores'][client_id].append(anomaly_score)
                self.demo_data['metrics']['reputation_scores'][client_id].append(client['reputation'])
                
                # Create security event
                event = {
                    'timestamp': datetime.now().isoformat(),
                    'client_id': client_id,
                    'event_type': 'high_anomaly',
                    'anomaly_score': anomaly_score,
                    'severity': 'high',
                    'description': f'High anomaly score detected for {client_id}'
                }
                self.demo_data['metrics']['security_events'].append(event)
                
                console.print(f"  🚨 [{client_id}] Anomaly Score: {anomaly_score:.3f}")
                time.sleep(1)
    
    def security_response(self):
        """Demonstrate security response."""
        console.print("🛡️  [bold yellow]SECURITY RESPONSE ACTIVATED[/bold yellow]")
        
        # Quarantine malicious clients
        for client_id, client in self.demo_data['clients'].items():
            if client['type'] == 'malicious' and client['reputation'] < 0.5:
                client['quarantined'] = True
                
                event = {
                    'timestamp': datetime.now().isoformat(),
                    'client_id': client_id,
                    'event_type': 'quarantine',
                    'severity': 'high',
                    'description': f'Client {client_id} quarantined due to low reputation'
                }
                self.demo_data['metrics']['security_events'].append(event)
                
                console.print(f"  🔒 [{client_id}] QUARANTINED")
                time.sleep(1)
        
        console.print("✅ Malicious clients isolated")
        console.print("🔄 System continuing with trusted clients only")
    
    def system_recovery(self):
        """Demonstrate system recovery."""
        console.print("🔄 [bold green]SYSTEM RECOVERY[/bold green]")
        
        # Continue training with honest clients
        active_clients = [cid for cid, c in self.demo_data['clients'].items() if not c['quarantined']]
        console.print(f"📊 Continuing training with {len(active_clients)} trusted clients")
        
        # Simulate recovery round
        self.demo_data['current_round'] += 1
        
        # Improved accuracy after removing malicious clients
        recovery_accuracy = 0.92 + np.random.uniform(0, 0.03)
        self.demo_data['metrics']['model_accuracy'].append(recovery_accuracy)
        
        console.print(f"📈 Model Accuracy Recovered: {recovery_accuracy:.3f}")
        console.print("✅ System successfully defended against attack")
        
        self.demo_data['system_status'] = 'secure'
    
    def create_dashboard_layout(self):
        """Create a live dashboard layout."""
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )
        
        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        return layout
    
    def update_dashboard(self, layout):
        """Update the dashboard with current data."""
        # Header
        status_color = {
            'stopped': 'red',
            'initializing': 'yellow', 
            'ready': 'blue',
            'training': 'purple',
            'secure': 'green'
        }.get(self.demo_data['system_status'], 'white')
        
        header_text = Text()
        header_text.append("🛡️ QSFL-CAAD Dashboard ", style="bold blue")
        header_text.append(f"[{self.demo_data['system_status'].upper()}]", style=f"bold {status_color}")
        header_text.append(f" | Round: {self.demo_data['current_round']}")
        
        layout["header"].update(Panel(header_text, title="System Status"))
        
        # Client table
        client_table = Table(title="Federated Learning Clients")
        client_table.add_column("Client ID", style="cyan")
        client_table.add_column("Type", style="magenta")
        client_table.add_column("Status", style="green")
        client_table.add_column("Reputation", style="yellow")
        client_table.add_column("Anomaly Score", style="red")
        
        for client_id, client in self.demo_data['clients'].items():
            status = "🔒 QUARANTINED" if client['quarantined'] else "✅ ACTIVE"
            status_style = "red" if client['quarantined'] else "green"
            
            type_style = {
                'honest': 'green',
                'suspicious': 'yellow', 
                'malicious': 'red'
            }.get(client['type'], 'white')
            
            client_table.add_row(
                client_id,
                f"[{type_style}]{client['type'].upper()}[/{type_style}]",
                f"[{status_style}]{status}[/{status_style}]",
                f"{client['reputation']:.3f}",
                f"{client['last_anomaly_score']:.3f}"
            )
        
        layout["left"].update(client_table)
        
        # Security events
        events_text = Text()
        events_text.append("Recent Security Events:\n\n", style="bold red")
        
        recent_events = self.demo_data['metrics']['security_events'][-5:]
        for event in recent_events:
            timestamp = datetime.fromisoformat(event['timestamp']).strftime("%H:%M:%S")
            events_text.append(f"[{timestamp}] ", style="dim")
            events_text.append(f"{event['client_id']}: ", style="bold")
            events_text.append(f"{event['description']}\n", style="yellow")
        
        if not recent_events:
            events_text.append("No security events", style="dim")
        
        layout["right"].update(Panel(events_text, title="Security Monitor"))
        
        # Footer
        if self.demo_data['metrics']['model_accuracy']:
            latest_accuracy = self.demo_data['metrics']['model_accuracy'][-1]
            footer_text = f"📊 Latest Model Accuracy: {latest_accuracy:.3f} | "
            footer_text += f"👥 Active Clients: {len([c for c in self.demo_data['clients'].values() if not c['quarantined']])} | "
            footer_text += f"🔒 Quarantined: {len([c for c in self.demo_data['clients'].values() if c['quarantined']])}"
        else:
            footer_text = "System ready for demonstration"
        
        layout["footer"].update(Panel(footer_text, title="Metrics"))
    
    def run_showcase(self):
        """Run the complete showcase demonstration."""
        console.print(Panel.fit("🎭 QSFL-CAAD Interactive Showcase", style="bold blue"))
        console.print("Press Ctrl+C to stop at any time\n")
        
        self.is_running = True
        layout = self.create_dashboard_layout()
        
        try:
            with Live(layout, refresh_per_second=2, screen=True):
                for i, step in enumerate(self.showcase_steps):
                    if not self.is_running:
                        break
                    
                    self.current_step = i
                    console.print(f"\n🎬 Step {i+1}/{len(self.showcase_steps)}: {step['name']}")
                    
                    # Execute step
                    step['action']()
                    
                    # Update dashboard during step
                    for _ in range(step['duration']):
                        if not self.is_running:
                            break
                        self.update_dashboard(layout)
                        time.sleep(1)
                
                if self.is_running:
                    console.print("\n🎉 [bold green]Showcase Complete![/bold green]")
                    console.print("The QSFL-CAAD system successfully:")
                    console.print("  ✅ Detected malicious clients")
                    console.print("  ✅ Quarantined threats automatically") 
                    console.print("  ✅ Maintained system security")
                    console.print("  ✅ Recovered model performance")
                    
                    # Keep dashboard running
                    while self.is_running:
                        self.update_dashboard(layout)
                        time.sleep(1)
        
        except KeyboardInterrupt:
            console.print("\n🛑 Showcase stopped by user")
        finally:
            self.is_running = False
    
    def export_demo_data(self, filename: str = None):
        """Export demo data for frontend consumption."""
        if not filename:
            filename = f"demo_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.demo_data, f, indent=2)
        
        console.print(f"📁 Demo data exported to: {filename}")

def main():
    """Main function to run the showcase."""
    showcase = DemoShowcase()
    
    try:
        showcase.run_showcase()
    except KeyboardInterrupt:
        console.print("\n👋 Goodbye!")
    finally:
        # Export data for frontend
        showcase.export_demo_data("frontend_demo_data.json")

if __name__ == "__main__":
    main()