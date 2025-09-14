#!/usr/bin/env python3
"""
Simple QSFL-CAAD Dashboard
"""
import json
import time
import threading
from datetime import datetime
from typing import Dict, List, Any
import numpy as np

try:
    from flask import Flask, render_template, jsonify, request
    from flask_socketio import SocketIO, emit
    FLASK_AVAILABLE = True
except ImportError:
    print("Flask not available. Install with: pip install flask flask-socketio")
    FLASK_AVAILABLE = False

class SimpleDashboard:
    """Simple web dashboard for QSFL-CAAD."""
    
    def __init__(self):
        """Initialize the dashboard."""
        if not FLASK_AVAILABLE:
            raise ImportError("Flask and Flask-SocketIO are required")
            
        self.app = Flask(__name__, 
                        template_folder='templates',
                        static_folder='static')
        self.app.config['SECRET_KEY'] = 'qsfl_caad_dashboard'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Dashboard data
        self.dashboard_data = {
            'clients': {},
            'metrics': {
                'timestamps': [],
                'anomaly_scores': {},
                'reputation_scores': {},
                'model_accuracy': [],
                'security_events': [],
                'system_performance': []
            },
            'system_status': 'stopped',
            'current_round': 0
        }
        
        self.config = {
            'anomaly_threshold': 0.6,
            'reputation_decay': 0.95,
            'update_interval': 2.0
        }
        
        self.monitoring_active = False
        
        self.setup_demo_clients()
        self.setup_routes()
        self.setup_websockets()
    
    def setup_demo_clients(self):
        """Setup demonstration clients."""
        client_configs = [
            ('honest_client_1', 'honest'),
            ('honest_client_2', 'honest'),
            ('honest_client_3', 'honest'),
            ('malicious_client_1', 'malicious'),
            ('malicious_client_2', 'malicious')
        ]
        
        for client_id, client_type in client_configs:
            self.dashboard_data['clients'][client_id] = {
                'type': client_type,
                'status': 'active',
                'reputation': 1.0,
                'last_anomaly_score': 0.0,
                'updates_sent': 0,
                'quarantined': False,
                'location': self._generate_mock_location()
            }
    
    def _generate_mock_location(self):
        """Generate mock geographical location."""
        locations = [
            {'city': 'New York', 'country': 'USA', 'lat': 40.7128, 'lng': -74.0060},
            {'city': 'London', 'country': 'UK', 'lat': 51.5074, 'lng': -0.1278},
            {'city': 'Tokyo', 'country': 'Japan', 'lat': 35.6762, 'lng': 139.6503},
            {'city': 'Sydney', 'country': 'Australia', 'lat': -33.8688, 'lng': 151.2093},
            {'city': 'Berlin', 'country': 'Germany', 'lat': 52.5200, 'lng': 13.4050}
        ]
        return np.random.choice(locations)
    
    def setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/')
        def dashboard():
            return render_template('dashboard.html')
        
        @self.app.route('/clients')
        def clients_page():
            return render_template('clients.html')
        
        @self.app.route('/security')
        def security_page():
            return render_template('security.html')
        
        @self.app.route('/analytics')
        def analytics_page():
            return render_template('analytics.html')
        
        @self.app.route('/api/dashboard_data')
        def api_dashboard_data():
            return jsonify(self.dashboard_data)
        
        @self.app.route('/api/system_status')
        def api_system_status():
            return jsonify({
                'status': self.dashboard_data['system_status'],
                'current_round': self.dashboard_data['current_round'],
                'clients_count': len(self.dashboard_data['clients']),
                'active_clients': len([c for c in self.dashboard_data['clients'].values() 
                                     if c['status'] == 'active' and not c['quarantined']])
            })
        
        @self.app.route('/api/control/<action>', methods=['POST'])
        def api_control(action):
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
    
    def setup_websockets(self):
        """Setup WebSocket event handlers."""
        
        @self.socketio.on('connect')
        def handle_connect():
            print(f"Client connected: {request.sid}")
            emit('connection_status', {'status': 'connected'})
            emit('dashboard_update', self.dashboard_data)
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            print(f"Client disconnected: {request.sid}")
        
        @self.socketio.on('request_update')
        def handle_request_update():
            emit('dashboard_update', self.dashboard_data)
        
        @self.socketio.on('client_action')
        def handle_client_action(data):
            client_id = data.get('client_id')
            action = data.get('action')
            
            if client_id in self.dashboard_data['clients']:
                if action == 'quarantine':
                    self.dashboard_data['clients'][client_id]['quarantined'] = True
                elif action == 'unquarantine':
                    self.dashboard_data['clients'][client_id]['quarantined'] = False
                
                emit('client_updated', {'client_id': client_id, 'action': action})
    
    def start_monitoring(self):
        """Start system monitoring."""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.dashboard_data['system_status'] = 'running'
            
            self.monitor_thread = threading.Thread(target=self.monitoring_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            
            self.broadcast_message("System monitoring started", "info")
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring_active = False
        self.dashboard_data['system_status'] = 'stopped'
        self.broadcast_message("System monitoring stopped", "info")
    
    def reset_system(self):
        """Reset system state."""
        self.stop_monitoring()
        self.dashboard_data['current_round'] = 0
        
        # Reset metrics
        for key in ['timestamps', 'model_accuracy', 'security_events']:
            self.dashboard_data['metrics'][key] = []
        
        for client_id in self.dashboard_data['clients'].keys():
            self.dashboard_data['metrics']['anomaly_scores'][client_id] = []
            self.dashboard_data['metrics']['reputation_scores'][client_id] = []
        
        # Reset client states
        for client_info in self.dashboard_data['clients'].values():
            client_info.update({
                'reputation': 1.0,
                'last_anomaly_score': 0.0,
                'updates_sent': 0,
                'quarantined': False
            })
        
        self.broadcast_message("System reset completed", "info")
    
    def monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                self.dashboard_data['current_round'] += 1
                round_number = self.dashboard_data['current_round']
                
                timestamp = datetime.now().isoformat()
                self.dashboard_data['metrics']['timestamps'].append(timestamp)
                
                # Process each client
                for client_id, client_info in self.dashboard_data['clients'].items():
                    if client_info['quarantined']:
                        continue
                    
                    # Generate anomaly score based on client type
                    if client_info['type'] == 'malicious' and round_number > 10:
                        anomaly_score = np.random.uniform(0.7, 0.95)
                    else:
                        anomaly_score = np.random.uniform(0.0, 0.4)
                    
                    client_info['last_anomaly_score'] = anomaly_score
                    client_info['updates_sent'] += 1
                    
                    # Update reputation
                    if anomaly_score > self.config['anomaly_threshold']:
                        client_info['reputation'] *= 0.85
                        if client_info['reputation'] < 0.3:
                            client_info['quarantined'] = True
                            self.create_security_event(client_id, 'quarantine', anomaly_score)
                    else:
                        client_info['reputation'] = min(1.0, client_info['reputation'] * 1.005)
                    
                    # Store metrics
                    if client_id not in self.dashboard_data['metrics']['anomaly_scores']:
                        self.dashboard_data['metrics']['anomaly_scores'][client_id] = []
                    if client_id not in self.dashboard_data['metrics']['reputation_scores']:
                        self.dashboard_data['metrics']['reputation_scores'][client_id] = []
                    
                    self.dashboard_data['metrics']['anomaly_scores'][client_id].append(anomaly_score)
                    self.dashboard_data['metrics']['reputation_scores'][client_id].append(client_info['reputation'])
                
                # Simulate model accuracy
                base_accuracy = 0.75 + (round_number * 0.003)
                noise = np.random.normal(0, 0.02)
                accuracy = min(0.98, max(0.6, base_accuracy + noise))
                self.dashboard_data['metrics']['model_accuracy'].append(accuracy)
                
                # Keep only last 100 data points
                self._trim_metrics(100)
                
                # Broadcast update
                self.socketio.emit('dashboard_update', self.dashboard_data)
                
                time.sleep(self.config['update_interval'])
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(1)
    
    def create_security_event(self, client_id: str, event_type: str, anomaly_score: float):
        """Create a security event."""
        event = {
            'timestamp': datetime.now().isoformat(),
            'client_id': client_id,
            'event_type': event_type,
            'anomaly_score': anomaly_score,
            'severity': 'high' if anomaly_score > 0.8 else 'medium',
            'description': f"Security event: {event_type} for client {client_id}"
        }
        
        self.dashboard_data['metrics']['security_events'].append(event)
        
        if len(self.dashboard_data['metrics']['security_events']) > 50:
            self.dashboard_data['metrics']['security_events'] = self.dashboard_data['metrics']['security_events'][-50:]
    
    def broadcast_message(self, message: str, level: str = "info"):
        """Broadcast message to all connected clients."""
        self.socketio.emit('system_message', {
            'message': message,
            'level': level,
            'timestamp': datetime.now().isoformat()
        })
    
    def _trim_metrics(self, max_points: int):
        """Trim metrics to keep only the last N points."""
        for key in ['timestamps', 'model_accuracy']:
            if len(self.dashboard_data['metrics'][key]) > max_points:
                self.dashboard_data['metrics'][key] = self.dashboard_data['metrics'][key][-max_points:]
        
        for client_id in self.dashboard_data['clients'].keys():
            for score_type in ['anomaly_scores', 'reputation_scores']:
                if client_id in self.dashboard_data['metrics'][score_type]:
                    if len(self.dashboard_data['metrics'][score_type][client_id]) > max_points:
                        self.dashboard_data['metrics'][score_type][client_id] = \
                            self.dashboard_data['metrics'][score_type][client_id][-max_points:]
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the dashboard server."""
        print(f"🚀 Starting Simple QSFL-CAAD Dashboard")
        print(f"📊 Dashboard URL: http://{host}:{port}")
        print(f"🛑 Press Ctrl+C to stop")
        
        try:
            self.socketio.run(self.app, host=host, port=port, debug=debug)
        except KeyboardInterrupt:
            print("\n🛑 Dashboard stopped")

def main():
    """Main function."""
    dashboard = SimpleDashboard()
    dashboard.run(host='127.0.0.1', port=5000, debug=True)

if __name__ == "__main__":
    main()