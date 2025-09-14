#!/usr/bin/env python3
"""
Enhanced Web Dashboard for QSFL-CAAD System

A modern, responsive web interface for monitoring and controlling
the QSFL-CAAD federated learning system.
"""

import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any
import numpy as np

# Web framework imports
try:
    from flask import Flask, render_template, jsonify, request, send_from_directory
    from flask_socketio import SocketIO, emit
    import plotly.graph_objs as go
    import plotly.utils
    FLASK_AVAILABLE = True
except ImportError:
    print("Flask or Plotly not available. Install with: pip install flask flask-socketio plotly")
    FLASK_AVAILABLE = False

# Import system components
try:
    from qsfl_caad.system import QSFLSystem
except ImportError:
    try:
        from qsfl_caad import QSFLSystem
    except ImportError:
        print("Warning: QSFLSystem not available. Running in demo mode.")
        QSFLSystem = None


class EnhancedQSFLDashboard:
    """Enhanced web dashboard with modern UI and advanced features."""
    
    def __init__(self):
        """Initialize the enhanced dashboard."""
        self.app = Flask(__name__, 
                        template_folder='templates',
                        static_folder='static')
        self.app.config['SECRET_KEY'] = 'qsfl_caad_enhanced_dashboard'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Enhanced dashboard state - initialize first
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
            'current_round': 0,
            'alerts': [],
            'attack_timeline': []
        }
        
        self.config = {
            'anomaly_threshold': 0.6,
            'reputation_decay': 0.95,
            'quarantine_threshold': 0.8,
            'update_interval': 2.0,
            'auto_start': False
        }
        
        self.monitoring_active = False
        
        # Now setup system after dashboard_data is initialized
        self.setup_system()
        self.setup_routes()
        self.setup_websockets()
    
    def setup_system(self):
        """Setup the QSFL-CAAD system."""
        try:
            if QSFLSystem:
                self.system = QSFLSystem()
            else:
                self.system = None
            self.setup_demo_clients()
        except Exception as e:
            print(f"System setup error: {e}")
            self.system = None
    
    def setup_demo_clients(self):
        """Setup demonstration clients."""
        client_configs = [
            ('honest_client_1', 'honest'),
            ('honest_client_2', 'honest'),
            ('honest_client_3', 'honest'),
            ('honest_client_4', 'honest'),
            ('suspicious_client_1', 'suspicious'),
            ('malicious_client_1', 'malicious'),
            ('malicious_client_2', 'malicious')
        ]
        
        for client_id, client_type in client_configs:
            try:
                # Try to register with system if available
                if self.system:
                    try:
                        credentials = self.system.register_client(client_id)
                    except:
                        pass  # Continue with demo mode
                
                self.dashboard_data['clients'][client_id] = {
                    'type': client_type,
                    'status': 'active',
                    'reputation': 1.0,
                    'last_anomaly_score': 0.0,
                    'updates_sent': 0,
                    'last_update': None,
                    'quarantined': False,
                    'location': self._generate_mock_location(),
                    'model_accuracy': np.random.uniform(0.85, 0.95)
                }
            except Exception as e:
                print(f"Error registering client {client_id}: {e}")
    
    def _generate_mock_location(self):
        """Generate mock geographical location for clients."""
        locations = [
            {'city': 'New York', 'country': 'USA', 'lat': 40.7128, 'lng': -74.0060},
            {'city': 'London', 'country': 'UK', 'lat': 51.5074, 'lng': -0.1278},
            {'city': 'Tokyo', 'country': 'Japan', 'lat': 35.6762, 'lng': 139.6503},
            {'city': 'Sydney', 'country': 'Australia', 'lat': -33.8688, 'lng': 151.2093},
            {'city': 'Berlin', 'country': 'Germany', 'lat': 52.5200, 'lng': 13.4050},
            {'city': 'Toronto', 'country': 'Canada', 'lat': 43.6532, 'lng': -79.3832},
            {'city': 'Singapore', 'country': 'Singapore', 'lat': 1.3521, 'lng': 103.8198}
        ]
        return np.random.choice(locations)
    
    def setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/')
        def dashboard():
            """Main dashboard page."""
            return render_template('dashboard.html')
        
        @self.app.route('/clients')
        def clients_page():
            """Clients management page."""
            return render_template('clients.html')
        
        @self.app.route('/security')
        def security_page():
            """Security monitoring page."""
            return render_template('security.html')
        
        @self.app.route('/analytics')
        def analytics_page():
            """Advanced analytics page."""
            return render_template('analytics.html')
        
        @self.app.route('/api/dashboard_data')
        def api_dashboard_data():
            """Get complete dashboard data."""
            return jsonify(self.dashboard_data)
        
        @self.app.route('/api/system_status')
        def api_system_status():
            """Get system status."""
            return jsonify({
                'status': self.dashboard_data['system_status'],
                'current_round': self.dashboard_data['current_round'],
                'clients_count': len(self.dashboard_data['clients']),
                'active_clients': len([c for c in self.dashboard_data['clients'].values() 
                                     if c['status'] == 'active' and not c['quarantined']]),
                'quarantined_clients': len([c for c in self.dashboard_data['clients'].values() 
                                          if c['quarantined']]),
                'model_accuracy': self.dashboard_data['metrics']['model_accuracy'][-1] if self.dashboard_data['metrics']['model_accuracy'] else 0,
                'total_updates': sum(c['updates_sent'] for c in self.dashboard_data['clients'].values())
            })
        
        @self.app.route('/api/clients_data')
        def api_clients_data():
            """Get detailed client data."""
            return jsonify(self.dashboard_data['clients'])
        
        @self.app.route('/api/security_events')
        def api_security_events():
            """Get security events."""
            return jsonify(self.dashboard_data['metrics']['security_events'][-50:])  # Last 50 events
        
        @self.app.route('/api/performance_metrics')
        def api_performance_metrics():
            """Get performance metrics."""
            return jsonify({
                'timestamps': self.dashboard_data['metrics']['timestamps'][-100:],
                'model_accuracy': self.dashboard_data['metrics']['model_accuracy'][-100:],
                'system_performance': self.dashboard_data['metrics']['system_performance'][-100:]
            })
        
        @self.app.route('/api/anomaly_data')
        def api_anomaly_data():
            """Get anomaly detection data for plotting."""
            plot_data = {}
            for client_id, scores in self.dashboard_data['metrics']['anomaly_scores'].items():
                if scores:
                    client_info = self.dashboard_data['clients'].get(client_id, {})
                    plot_data[client_id] = {
                        'scores': scores[-50:],  # Last 50 points
                        'type': client_info.get('type', 'unknown'),
                        'quarantined': client_info.get('quarantined', False)
                    }
            
            return jsonify({
                'timestamps': self.dashboard_data['metrics']['timestamps'][-50:],
                'clients': plot_data,
                'threshold': self.config['anomaly_threshold']
            })
        
        @self.app.route('/api/world_map_data')
        def api_world_map_data():
            """Get data for world map visualization."""
            map_data = []
            for client_id, client_info in self.dashboard_data['clients'].items():
                location = client_info.get('location', {})
                map_data.append({
                    'client_id': client_id,
                    'lat': location.get('lat', 0),
                    'lng': location.get('lng', 0),
                    'city': location.get('city', 'Unknown'),
                    'country': location.get('country', 'Unknown'),
                    'type': client_info['type'],
                    'status': 'quarantined' if client_info['quarantined'] else 'active',
                    'reputation': client_info['reputation'],
                    'anomaly_score': client_info['last_anomaly_score']
                })
            return jsonify(map_data)
        
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
            elif action == 'pause':
                self.monitoring_active = False
                self.dashboard_data['system_status'] = 'paused'
                return jsonify({'status': 'paused'})
            else:
                return jsonify({'error': 'Unknown action'}), 400
        
        @self.app.route('/api/config', methods=['GET', 'POST'])
        def api_config():
            """Get or update configuration."""
            if request.method == 'POST':
                new_config = request.json
                self.config.update(new_config)
                return jsonify({'status': 'updated', 'config': self.config})
            else:
                return jsonify(self.config)
        
        @self.app.route('/api/simulate_attack', methods=['POST'])
        def api_simulate_attack():
            """Simulate different types of attacks."""
            attack_data = request.json
            attack_type = attack_data.get('type', 'gradient_poisoning')
            intensity = attack_data.get('intensity', 'medium')
            
            result = self.simulate_attack(attack_type, intensity)
            return jsonify(result)
        
        @self.app.route('/static/<path:filename>')
        def static_files(filename):
            """Serve static files."""
            return send_from_directory('static', filename)
    
    def setup_websockets(self):
        """Setup WebSocket event handlers."""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection."""
            print(f"Dashboard client connected: {request.sid}")
            emit('connection_status', {'status': 'connected'})
            # Send initial data
            emit('dashboard_update', self.dashboard_data)
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection."""
            print(f"Dashboard client disconnected: {request.sid}")
        
        @self.socketio.on('request_update')
        def handle_request_update():
            """Handle request for data update."""
            emit('dashboard_update', self.dashboard_data)
        
        @self.socketio.on('client_action')
        def handle_client_action(data):
            """Handle client-specific actions."""
            client_id = data.get('client_id')
            action = data.get('action')
            
            if client_id in self.dashboard_data['clients']:
                if action == 'quarantine':
                    self.dashboard_data['clients'][client_id]['quarantined'] = True
                elif action == 'unquarantine':
                    self.dashboard_data['clients'][client_id]['quarantined'] = False
                elif action == 'reset_reputation':
                    self.dashboard_data['clients'][client_id]['reputation'] = 1.0
                
                emit('client_updated', {'client_id': client_id, 'action': action})
    
    def start_monitoring(self):
        """Start system monitoring."""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.dashboard_data['system_status'] = 'running'
            
            # Start monitoring thread
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
        
        # Reset metrics
        self.dashboard_data['current_round'] = 0
        self.dashboard_data['metrics'] = {
            'timestamps': [],
            'anomaly_scores': {client_id: [] for client_id in self.dashboard_data['clients'].keys()},
            'reputation_scores': {client_id: [] for client_id in self.dashboard_data['clients'].keys()},
            'model_accuracy': [],
            'security_events': [],
            'system_performance': []
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
        
        self.broadcast_message("System reset completed", "info")
    
    def monitoring_loop(self):
        """Main monitoring loop with enhanced features."""
        attack_phase_started = False
        
        while self.monitoring_active:
            try:
                self.dashboard_data['current_round'] += 1
                round_number = self.dashboard_data['current_round']
                
                timestamp = datetime.now()
                self.dashboard_data['metrics']['timestamps'].append(timestamp.isoformat())
                
                # Determine if attack phase should start
                if round_number > 15 and not attack_phase_started:
                    attack_phase_started = True
                    self.broadcast_message("🚨 Attack phase initiated!", "warning")
                    self.dashboard_data['attack_timeline'].append({
                        'timestamp': timestamp.isoformat(),
                        'event': 'Attack Phase Started',
                        'description': 'Malicious clients beginning coordinated attack'
                    })
                
                # Process each client
                for client_id, client_info in self.dashboard_data['clients'].items():
                    if client_info['status'] != 'active' or client_info['quarantined']:
                        continue
                    
                    # Generate anomaly score based on client type and phase
                    if client_info['type'] == 'malicious' and attack_phase_started:
                        anomaly_score = np.random.uniform(0.7, 0.95)
                        if anomaly_score > 0.8:
                            self.create_security_event(client_id, 'high_anomaly', anomaly_score)
                    elif client_info['type'] == 'suspicious' and attack_phase_started:
                        anomaly_score = np.random.uniform(0.5, 0.75)
                    else:
                        anomaly_score = np.random.uniform(0.0, 0.4)
                    
                    client_info['last_anomaly_score'] = anomaly_score
                    client_info['updates_sent'] += 1
                    
                    # Update reputation
                    if anomaly_score > self.config['anomaly_threshold']:
                        client_info['reputation'] *= 0.85
                        if client_info['reputation'] < self.config['quarantine_threshold']:
                            if not client_info['quarantined']:
                                client_info['quarantined'] = True
                                self.create_security_event(client_id, 'quarantine', anomaly_score)
                                self.broadcast_message(f"Client {client_id} quarantined", "warning")
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
                
                # System performance metrics
                performance = {
                    'cpu_usage': np.random.uniform(20, 80),
                    'memory_usage': np.random.uniform(40, 85),
                    'network_io': np.random.uniform(10, 100)
                }
                self.dashboard_data['metrics']['system_performance'].append(performance)
                
                # Keep only last 200 data points
                self._trim_metrics(200)
                
                # Broadcast update to all connected clients
                self.socketio.emit('dashboard_update', self.dashboard_data)
                
                # Wait for next iteration
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
        
        # Keep only last 100 events
        if len(self.dashboard_data['metrics']['security_events']) > 100:
            self.dashboard_data['metrics']['security_events'] = self.dashboard_data['metrics']['security_events'][-100:]
    
    def simulate_attack(self, attack_type: str, intensity: str) -> Dict[str, Any]:
        """Simulate different types of attacks."""
        malicious_clients = [cid for cid, cinfo in self.dashboard_data['clients'].items() 
                           if cinfo['type'] == 'malicious']
        
        if not malicious_clients:
            return {'error': 'No malicious clients available'}
        
        intensity_multiplier = {'low': 0.6, 'medium': 0.8, 'high': 1.0}.get(intensity, 0.8)
        
        for client_id in malicious_clients:
            client_info = self.dashboard_data['clients'][client_id]
            base_score = np.random.uniform(0.7, 0.95)
            client_info['last_anomaly_score'] = base_score * intensity_multiplier
            client_info['reputation'] *= 0.7
        
        self.broadcast_message(f"Simulated {attack_type} attack ({intensity} intensity)", "warning")
        
        return {
            'attack_type': attack_type,
            'intensity': intensity,
            'affected_clients': malicious_clients,
            'status': 'simulated'
        }
    
    def broadcast_message(self, message: str, level: str = "info"):
        """Broadcast message to all connected clients."""
        self.socketio.emit('system_message', {
            'message': message,
            'level': level,
            'timestamp': datetime.now().isoformat()
        })
    
    def _trim_metrics(self, max_points: int):
        """Trim metrics to keep only the last N points."""
        for key in ['timestamps', 'model_accuracy', 'system_performance']:
            if len(self.dashboard_data['metrics'][key]) > max_points:
                self.dashboard_data['metrics'][key] = self.dashboard_data['metrics'][key][-max_points:]
        
        for client_id in self.dashboard_data['clients'].keys():
            for score_type in ['anomaly_scores', 'reputation_scores']:
                if client_id in self.dashboard_data['metrics'][score_type]:
                    if len(self.dashboard_data['metrics'][score_type][client_id]) > max_points:
                        self.dashboard_data['metrics'][score_type][client_id] = \
                            self.dashboard_data['metrics'][score_type][client_id][-max_points:]
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the enhanced dashboard server."""
        if not FLASK_AVAILABLE:
            print("Flask or Plotly not available. Please install required packages:")
            print("pip install flask flask-socketio plotly")
            return
        
        print(f"🚀 Starting Enhanced QSFL-CAAD Dashboard")
        print(f"📊 Dashboard URL: http://{host}:{port}")
        print(f"🛑 Press Ctrl+C to stop")
        
        try:
            self.socketio.run(self.app, host=host, port=port, debug=debug)
        except KeyboardInterrupt:
            print("\n🛑 Dashboard stopped")


def main():
    """Main function to run the enhanced dashboard."""
    dashboard = EnhancedQSFLDashboard()
    dashboard.run(host='0.0.0.0', port=5000, debug=False)


if __name__ == "__main__":
    main()