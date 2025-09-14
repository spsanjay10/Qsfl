#!/usr/bin/env python3
"""
Interactive Dashboard for QSFL-CAAD System Monitoring

This script provides a web-based dashboard for real-time monitoring
of the QSFL-CAAD system with interactive controls and visualizations.
"""

import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any
import numpy as np

# Web framework imports
try:
    from flask import Flask, render_template, jsonify, request
    from flask_socketio import SocketIO, emit
    FLASK_AVAILABLE = True
except ImportError:
    print("Flask not available. Install with: pip install flask flask-socketio")
    FLASK_AVAILABLE = False

# Plotting imports
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io
import base64

class QSFLDashboard:
    """Interactive dashboard for QSFL-CAAD system monitoring."""
    
    def __init__(self):
        """Initialize the dashboard."""
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'qsfl_caad_dashboard_secret'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        self.setup_system()
        self.setup_routes()
        self.setup_websockets()
        
        # Dashboard state
        self.dashboard_data = {
            'clients': {},
            'metrics': {
                'timestamps': [],
                'anomaly_scores': [],
                'reputation_scores': [],
                'model_accuracy': [],
                'security_events': []
            },
            'system_status': 'running',
            'current_round': 0
        }
        
        # Configuration
        self.config = {
            'anomaly_threshold': 0.6,
            'reputation_decay': 0.95,
            'quarantine_threshold': 0.8,
            'update_interval': 2.0
        }
        
        self.monitoring_active = False
    
    def setup_system(self):
        """Setup the QSFL-CAAD system."""
        try:
            from qsfl_caad import QSFLSystem
            self.system = QSFLSystem()
            
            # Setup demo clients
            self.setup_demo_clients()
            
        except ImportError:
            print("QSFL-CAAD system not available, using mock system")
            self.system = MockQSFLSystem()
            self.setup_demo_clients()
    
    def setup_demo_clients(self):
        """Setup demonstration clients."""
        client_configs = [
            ('honest_client_1', 'honest'),
            ('honest_client_2', 'honest'),
            ('honest_client_3', 'honest'),
            ('honest_client_4', 'honest'),
            ('malicious_client_1', 'malicious'),
            ('malicious_client_2', 'malicious')
        ]
        
        for client_id, client_type in client_configs:
            try:
                credentials = self.system.register_client(client_id)
                self.dashboard_data['clients'][client_id] = {
                    'type': client_type,
                    'status': 'active',
                    'reputation': 1.0,
                    'last_anomaly_score': 0.0,
                    'updates_sent': 0,
                    'last_update': None,
                    'quarantined': False
                }
            except Exception as e:
                print(f"Error registering client {client_id}: {e}")
    
    def setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/')
        def index():
            """Main dashboard page."""
            return self.render_dashboard()
        
        @self.app.route('/api/status')
        def api_status():
            """Get system status."""
            return jsonify({
                'status': self.dashboard_data['system_status'],
                'current_round': self.dashboard_data['current_round'],
                'clients_count': len(self.dashboard_data['clients']),
                'active_clients': len([c for c in self.dashboard_data['clients'].values() 
                                     if c['status'] == 'active']),
                'quarantined_clients': len([c for c in self.dashboard_data['clients'].values() 
                                          if c['quarantined']])
            })
        
        @self.app.route('/api/clients')
        def api_clients():
            """Get client information."""
            return jsonify(self.dashboard_data['clients'])
        
        @self.app.route('/api/metrics')
        def api_metrics():
            """Get system metrics."""
            return jsonify(self.dashboard_data['metrics'])
        
        @self.app.route('/api/config', methods=['GET', 'POST'])
        def api_config():
            """Get or update configuration."""
            if request.method == 'POST':
                new_config = request.json
                self.update_config(new_config)
                return jsonify({'status': 'updated', 'config': self.config})
            else:
                return jsonify(self.config)
        
        @self.app.route('/api/control/<action>', methods=['POST'])
        def api_control(action):
            """Control system operations."""
            if action == 'start':
                self.start_monitoring()
                return jsonify({'status': 'started'})
            elif action == 'stop':
                self.stop_monitoring()
                return jsonify({'status': 'stopped'})
            elif action == 'reset':
                self.reset_system()
                return jsonify({'status': 'reset'})
            else:
                return jsonify({'error': 'Unknown action'}), 400
        
        @self.app.route('/api/simulate/<attack_type>', methods=['POST'])
        def api_simulate_attack(attack_type):
            """Simulate different types of attacks."""
            result = self.simulate_attack(attack_type)
            return jsonify(result)
        
        @self.app.route('/api/plot/<plot_type>')
        def api_plot(plot_type):
            """Generate plots as base64 encoded images."""
            plot_data = self.generate_plot(plot_type)
            return jsonify({'image': plot_data})
    
    def setup_websockets(self):
        """Setup WebSocket event handlers."""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection."""
            print(f"Client connected: {request.sid}")
            emit('status', {'message': 'Connected to QSFL-CAAD Dashboard'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection."""
            print(f"Client disconnected: {request.sid}")
        
        @self.socketio.on('request_update')
        def handle_request_update():
            """Handle request for data update."""
            self.send_dashboard_update()
    
    def render_dashboard(self):
        """Render the main dashboard HTML."""
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QSFL-CAAD Interactive Dashboard</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }
        .dashboard-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        .card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .card h3 {
            margin-top: 0;
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }
        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        .status-card {
            background: white;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .status-value {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }
        .status-label {
            color: #666;
            margin-top: 5px;
        }
        .client-list {
            max-height: 300px;
            overflow-y: auto;
        }
        .client-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
            background: #f8f9fa;
        }
        .client-honest { border-left: 4px solid #28a745; }
        .client-malicious { border-left: 4px solid #dc3545; }
        .client-quarantined { background: #fff3cd; }
        .controls {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            transition: background-color 0.3s;
        }
        .btn-primary { background: #667eea; color: white; }
        .btn-success { background: #28a745; color: white; }
        .btn-danger { background: #dc3545; color: white; }
        .btn-warning { background: #ffc107; color: black; }
        .btn:hover { opacity: 0.8; }
        .config-panel {
            background: #e9ecef;
            padding: 15px;
            border-radius: 5px;
            margin-top: 15px;
        }
        .config-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 10px 0;
        }
        .config-item input {
            width: 100px;
            padding: 5px;
            border: 1px solid #ccc;
            border-radius: 3px;
        }
        .plot-container {
            height: 400px;
            margin: 20px 0;
        }
        .log-container {
            background: #1e1e1e;
            color: #00ff00;
            padding: 15px;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
            height: 200px;
            overflow-y: auto;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🛡️ QSFL-CAAD Interactive Dashboard</h1>
        <p>Quantum-Safe Federated Learning with Client Anomaly and Attack Detection</p>
    </div>

    <div class="status-grid">
        <div class="status-card">
            <div class="status-value" id="system-status">STOPPED</div>
            <div class="status-label">System Status</div>
        </div>
        <div class="status-card">
            <div class="status-value" id="current-round">0</div>
            <div class="status-label">Current Round</div>
        </div>
        <div class="status-card">
            <div class="status-value" id="active-clients">0</div>
            <div class="status-label">Active Clients</div>
        </div>
        <div class="status-card">
            <div class="status-value" id="quarantined-clients">0</div>
            <div class="status-label">Quarantined</div>
        </div>
    </div>

    <div class="controls">
        <button class="btn btn-success" onclick="startSystem()">▶️ Start</button>
        <button class="btn btn-danger" onclick="stopSystem()">⏹️ Stop</button>
        <button class="btn btn-warning" onclick="resetSystem()">🔄 Reset</button>
        <button class="btn btn-primary" onclick="simulateAttack('gradient_poisoning')">🦹 Simulate Attack</button>
    </div>

    <div class="dashboard-grid">
        <div class="card">
            <h3>📊 Real-time Metrics</h3>
            <div id="anomaly-plot" class="plot-container"></div>
        </div>
        
        <div class="card">
            <h3>👥 Client Status</h3>
            <div id="client-list" class="client-list"></div>
        </div>
    </div>

    <div class="dashboard-grid">
        <div class="card">
            <h3>📈 Model Performance</h3>
            <div id="accuracy-plot" class="plot-container"></div>
        </div>
        
        <div class="card">
            <h3>⚙️ Configuration</h3>
            <div class="config-panel">
                <div class="config-item">
                    <label>Anomaly Threshold:</label>
                    <input type="number" id="anomaly-threshold" step="0.1" min="0" max="1" value="0.6">
                </div>
                <div class="config-item">
                    <label>Reputation Decay:</label>
                    <input type="number" id="reputation-decay" step="0.01" min="0" max="1" value="0.95">
                </div>
                <div class="config-item">
                    <label>Quarantine Threshold:</label>
                    <input type="number" id="quarantine-threshold" step="0.1" min="0" max="1" value="0.8">
                </div>
                <button class="btn btn-primary" onclick="updateConfig()">Update Config</button>
            </div>
        </div>
    </div>

    <div class="card">
        <h3>📝 System Log</h3>
        <div id="system-log" class="log-container"></div>
    </div>

    <script>
        // WebSocket connection
        const socket = io();
        
        // Dashboard state
        let dashboardData = {
            clients: {},
            metrics: { timestamps: [], anomaly_scores: [], model_accuracy: [] }
        };

        // Socket event handlers
        socket.on('connect', function() {
            logMessage('Connected to QSFL-CAAD Dashboard');
            requestUpdate();
        });

        socket.on('dashboard_update', function(data) {
            updateDashboard(data);
        });

        socket.on('log_message', function(data) {
            logMessage(data.message, data.level);
        });

        // Control functions
        function startSystem() {
            fetch('/api/control/start', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    logMessage('System started', 'info');
                    requestUpdate();
                });
        }

        function stopSystem() {
            fetch('/api/control/stop', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    logMessage('System stopped', 'info');
                    requestUpdate();
                });
        }

        function resetSystem() {
            fetch('/api/control/reset', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    logMessage('System reset', 'info');
                    requestUpdate();
                });
        }

        function simulateAttack(attackType) {
            fetch(`/api/simulate/${attackType}`, { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    logMessage(`Simulated ${attackType} attack`, 'warning');
                });
        }

        function updateConfig() {
            const config = {
                anomaly_threshold: parseFloat(document.getElementById('anomaly-threshold').value),
                reputation_decay: parseFloat(document.getElementById('reputation-decay').value),
                quarantine_threshold: parseFloat(document.getElementById('quarantine-threshold').value)
            };

            fetch('/api/config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config)
            })
            .then(response => response.json())
            .then(data => {
                logMessage('Configuration updated', 'info');
            });
        }

        function requestUpdate() {
            socket.emit('request_update');
        }

        function updateDashboard(data) {
            dashboardData = data;
            
            // Update status cards
            document.getElementById('system-status').textContent = data.system_status.toUpperCase();
            document.getElementById('current-round').textContent = data.current_round;
            
            const activeClients = Object.values(data.clients).filter(c => c.status === 'active').length;
            const quarantinedClients = Object.values(data.clients).filter(c => c.quarantined).length;
            
            document.getElementById('active-clients').textContent = activeClients;
            document.getElementById('quarantined-clients').textContent = quarantinedClients;
            
            // Update client list
            updateClientList(data.clients);
            
            // Update plots
            updateAnomalyPlot(data.metrics);
            updateAccuracyPlot(data.metrics);
        }

        function updateClientList(clients) {
            const container = document.getElementById('client-list');
            container.innerHTML = '';
            
            Object.entries(clients).forEach(([clientId, client]) => {
                const item = document.createElement('div');
                item.className = `client-item client-${client.type}`;
                if (client.quarantined) item.classList.add('client-quarantined');
                
                item.innerHTML = `
                    <div>
                        <strong>${clientId}</strong>
                        <br>
                        <small>Rep: ${client.reputation.toFixed(3)} | Anomaly: ${client.last_anomaly_score.toFixed(3)}</small>
                    </div>
                    <div>
                        ${client.type === 'malicious' ? '😈' : '😇'}
                        ${client.quarantined ? '🚫' : '✅'}
                    </div>
                `;
                
                container.appendChild(item);
            });
        }

        function updateAnomalyPlot(metrics) {
            if (!metrics.timestamps || metrics.timestamps.length === 0) return;
            
            const traces = [];
            const clientIds = Object.keys(dashboardData.clients);
            
            clientIds.forEach(clientId => {
                const client = dashboardData.clients[clientId];
                const scores = metrics.anomaly_scores[clientId] || [];
                
                if (scores.length > 0) {
                    traces.push({
                        x: metrics.timestamps.slice(-scores.length),
                        y: scores,
                        name: clientId,
                        type: 'scatter',
                        mode: 'lines+markers',
                        line: { color: client.type === 'malicious' ? 'red' : 'blue' }
                    });
                }
            });
            
            // Add threshold line
            traces.push({
                x: metrics.timestamps,
                y: Array(metrics.timestamps.length).fill(0.6),
                name: 'Threshold',
                type: 'scatter',
                mode: 'lines',
                line: { color: 'red', dash: 'dash' }
            });
            
            const layout = {
                title: 'Anomaly Scores Over Time',
                xaxis: { title: 'Time' },
                yaxis: { title: 'Anomaly Score', range: [0, 1] },
                height: 350
            };
            
            Plotly.newPlot('anomaly-plot', traces, layout);
        }

        function updateAccuracyPlot(metrics) {
            if (!metrics.model_accuracy || metrics.model_accuracy.length === 0) return;
            
            const trace = {
                x: Array.from({length: metrics.model_accuracy.length}, (_, i) => i + 1),
                y: metrics.model_accuracy,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Model Accuracy',
                line: { color: 'green', width: 3 }
            };
            
            const layout = {
                title: 'Global Model Accuracy',
                xaxis: { title: 'Training Round' },
                yaxis: { title: 'Accuracy', range: [0, 1] },
                height: 350
            };
            
            Plotly.newPlot('accuracy-plot', [trace], layout);
        }

        function logMessage(message, level = 'info') {
            const logContainer = document.getElementById('system-log');
            const timestamp = new Date().toLocaleTimeString();
            const levelColor = {
                'info': '#00ff00',
                'warning': '#ffff00',
                'error': '#ff0000'
            }[level] || '#00ff00';
            
            const logEntry = document.createElement('div');
            logEntry.innerHTML = `<span style="color: #888">[${timestamp}]</span> <span style="color: ${levelColor}">${message}</span>`;
            
            logContainer.appendChild(logEntry);
            logContainer.scrollTop = logContainer.scrollHeight;
            
            // Keep only last 100 log entries
            while (logContainer.children.length > 100) {
                logContainer.removeChild(logContainer.firstChild);
            }
        }

        // Auto-refresh every 2 seconds
        setInterval(requestUpdate, 2000);
    </script>
</body>
</html>
        """
        return html_template
    
    def start_monitoring(self):
        """Start system monitoring."""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.dashboard_data['system_status'] = 'running'
            
            # Start monitoring thread
            self.monitor_thread = threading.Thread(target=self.monitoring_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            
            self.send_log_message("System monitoring started", "info")
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring_active = False
        self.dashboard_data['system_status'] = 'stopped'
        self.send_log_message("System monitoring stopped", "info")
    
    def reset_system(self):
        """Reset system state."""
        self.stop_monitoring()
        
        # Reset dashboard data
        self.dashboard_data['current_round'] = 0
        self.dashboard_data['metrics'] = {
            'timestamps': [],
            'anomaly_scores': {client_id: [] for client_id in self.dashboard_data['clients'].keys()},
            'reputation_scores': {client_id: [] for client_id in self.dashboard_data['clients'].keys()},
            'model_accuracy': [],
            'security_events': []
        }
        
        # Reset client states
        for client_id, client_info in self.dashboard_data['clients'].items():
            client_info.update({
                'reputation': 1.0,
                'last_anomaly_score': 0.0,
                'updates_sent': 0,
                'quarantined': False,
                'status': 'active'
            })
        
        self.send_log_message("System reset completed", "info")
        self.send_dashboard_update()
    
    def monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Simulate federated learning round
                self.simulate_federated_round()
                
                # Send updates to dashboard
                self.send_dashboard_update()
                
                # Wait for next iteration
                time.sleep(self.config['update_interval'])
                
            except Exception as e:
                self.send_log_message(f"Monitoring error: {e}", "error")
                time.sleep(1)
    
    def simulate_federated_round(self):
        """Simulate a federated learning round."""
        self.dashboard_data['current_round'] += 1
        round_number = self.dashboard_data['current_round']
        
        timestamp = datetime.now()
        self.dashboard_data['metrics']['timestamps'].append(timestamp.isoformat())
        
        # Process each client
        for client_id, client_info in self.dashboard_data['clients'].items():
            if client_info['status'] != 'active' or client_info['quarantined']:
                continue
            
            # Simulate anomaly score
            if client_info['type'] == 'malicious' and round_number > 10:
                # Malicious clients start showing anomalous behavior after round 10
                anomaly_score = np.random.uniform(0.7, 0.95)
            else:
                # Honest clients or early rounds
                anomaly_score = np.random.uniform(0.0, 0.3)
            
            client_info['last_anomaly_score'] = anomaly_score
            client_info['updates_sent'] += 1
            
            # Update reputation based on anomaly score
            if anomaly_score > self.config['anomaly_threshold']:
                client_info['reputation'] *= 0.8  # Decrease reputation
                self.send_log_message(f"High anomaly detected for {client_id}: {anomaly_score:.3f}", "warning")
            else:
                client_info['reputation'] = min(1.0, client_info['reputation'] * 1.01)  # Slowly increase
            
            # Check for quarantine
            if client_info['reputation'] < self.config['quarantine_threshold']:
                client_info['quarantined'] = True
                self.send_log_message(f"Client {client_id} quarantined (reputation: {client_info['reputation']:.3f})", "warning")
            
            # Store metrics
            if client_id not in self.dashboard_data['metrics']['anomaly_scores']:
                self.dashboard_data['metrics']['anomaly_scores'][client_id] = []
            if client_id not in self.dashboard_data['metrics']['reputation_scores']:
                self.dashboard_data['metrics']['reputation_scores'][client_id] = []
            
            self.dashboard_data['metrics']['anomaly_scores'][client_id].append(anomaly_score)
            self.dashboard_data['metrics']['reputation_scores'][client_id].append(client_info['reputation'])
        
        # Simulate model accuracy
        base_accuracy = 0.7 + (round_number * 0.005)  # Gradual improvement
        noise = np.random.normal(0, 0.02)
        accuracy = min(0.95, max(0.5, base_accuracy + noise))
        
        self.dashboard_data['metrics']['model_accuracy'].append(accuracy)
        
        # Keep only last 50 data points
        max_points = 50
        for key in ['timestamps', 'model_accuracy']:
            if len(self.dashboard_data['metrics'][key]) > max_points:
                self.dashboard_data['metrics'][key] = self.dashboard_data['metrics'][key][-max_points:]
        
        for client_id in self.dashboard_data['clients'].keys():
            for score_type in ['anomaly_scores', 'reputation_scores']:
                if client_id in self.dashboard_data['metrics'][score_type]:
                    if len(self.dashboard_data['metrics'][score_type][client_id]) > max_points:
                        self.dashboard_data['metrics'][score_type][client_id] = \
                            self.dashboard_data['metrics'][score_type][client_id][-max_points:]
    
    def simulate_attack(self, attack_type: str) -> Dict[str, Any]:
        """Simulate different types of attacks."""
        malicious_clients = [cid for cid, cinfo in self.dashboard_data['clients'].items() 
                           if cinfo['type'] == 'malicious']
        
        if not malicious_clients:
            return {'error': 'No malicious clients available'}
        
        # Force high anomaly scores for malicious clients
        for client_id in malicious_clients:
            client_info = self.dashboard_data['clients'][client_id]
            client_info['last_anomaly_score'] = np.random.uniform(0.8, 0.95)
            client_info['reputation'] *= 0.5  # Significant reputation drop
        
        self.send_log_message(f"Simulated {attack_type} attack from {len(malicious_clients)} clients", "warning")
        
        return {
            'attack_type': attack_type,
            'affected_clients': malicious_clients,
            'status': 'simulated'
        }
    
    def update_config(self, new_config: Dict[str, Any]):
        """Update system configuration."""
        self.config.update(new_config)
        self.send_log_message("Configuration updated", "info")
    
    def send_dashboard_update(self):
        """Send dashboard update via WebSocket."""
        self.socketio.emit('dashboard_update', self.dashboard_data)
    
    def send_log_message(self, message: str, level: str = "info"):
        """Send log message via WebSocket."""
        self.socketio.emit('log_message', {'message': message, 'level': level})
    
    def generate_plot(self, plot_type: str) -> str:
        """Generate plot as base64 encoded image."""
        plt.figure(figsize=(10, 6))
        
        if plot_type == 'anomaly_scores':
            for client_id, scores in self.dashboard_data['metrics']['anomaly_scores'].items():
                if scores:
                    client_type = self.dashboard_data['clients'][client_id]['type']
                    color = 'red' if client_type == 'malicious' else 'blue'
                    plt.plot(scores, label=client_id, color=color, alpha=0.7)
            
            plt.axhline(y=self.config['anomaly_threshold'], color='red', linestyle='--', label='Threshold')
            plt.title('Anomaly Scores Over Time')
            plt.ylabel('Anomaly Score')
            plt.legend()
        
        elif plot_type == 'model_accuracy':
            accuracy_data = self.dashboard_data['metrics']['model_accuracy']
            if accuracy_data:
                plt.plot(accuracy_data, 'g-', linewidth=2, label='Model Accuracy')
                plt.title('Global Model Accuracy')
                plt.ylabel('Accuracy')
                plt.legend()
        
        # Convert plot to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return plot_data
    
    def run(self, host='localhost', port=5000, debug=False):
        """Run the dashboard server."""
        if not FLASK_AVAILABLE:
            print("Flask not available. Please install Flask and Flask-SocketIO:")
            print("pip install flask flask-socketio")
            return
        
        print(f"🚀 Starting QSFL-CAAD Interactive Dashboard")
        print(f"📊 Dashboard URL: http://{host}:{port}")
        print(f"🛑 Press Ctrl+C to stop")
        
        try:
            self.socketio.run(self.app, host=host, port=port, debug=debug)
        except KeyboardInterrupt:
            print("\n🛑 Dashboard stopped")

class MockQSFLSystem:
    """Mock QSFL system for demonstration when real system is not available."""
    
    def __init__(self):
        self.clients = {}
    
    def register_client(self, client_id: str):
        """Mock client registration."""
        from collections import namedtuple
        Credentials = namedtuple('Credentials', ['client_id', 'public_key', 'private_key'])
        
        credentials = Credentials(
            client_id=client_id,
            public_key=b'mock_public_key',
            private_key=b'mock_private_key'
        )
        
        self.clients[client_id] = credentials
        return credentials
    
    def get_system_metrics(self):
        """Mock system metrics."""
        return {
            'cpu_usage': np.random.uniform(20, 80),
            'memory_usage': np.random.uniform(30, 70),
            'active_clients': len(self.clients)
        }

def main():
    """Main function to run the dashboard."""
    dashboard = QSFLDashboard()
    
    try:
        dashboard.run(host='0.0.0.0', port=5000, debug=False)
    except Exception as e:
        print(f"Error running dashboard: {e}")

if __name__ == "__main__":
    main()