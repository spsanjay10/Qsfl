#!/usr/bin/env python3
"""
Working QSFL-CAAD Dashboard
"""
import json
import time
import threading
from datetime import datetime
import numpy as np

from flask import Flask, render_template_string, jsonify, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS

# HTML Template embedded in Python
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QSFL-CAAD Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <style>
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .dashboard-container {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            margin: 20px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            text-align: center;
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 20px;
        }
        .metric-value {
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .metric-label {
            font-size: 0.9rem;
            opacity: 0.9;
        }
        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 8px;
        }
        .status-running { background-color: #28a745; }
        .status-stopped { background-color: #dc3545; }
        .status-paused { background-color: #ffc107; }
        .client-card {
            border-left: 4px solid #17a2b8;
            margin-bottom: 15px;
        }
        .client-card.honest { border-left-color: #28a745; }
        .client-card.malicious { border-left-color: #dc3545; }
        .client-card.suspicious { border-left-color: #ffc107; }
        .btn-gradient {
            background: linear-gradient(135deg, #667eea, #764ba2);
            border: none;
            color: white;
        }
        .btn-gradient:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            color: white;
        }
    </style>
</head>
<body>
    <div class="dashboard-container">
        <div class="d-flex justify-content-between align-items-center mb-4">
            <h1 class="h2">
                <i class="fas fa-shield-alt me-2"></i>
                QSFL-CAAD Dashboard
            </h1>
            <div class="btn-group">
                <button class="btn btn-success btn-sm" onclick="controlSystem('start')">
                    <i class="fas fa-play me-1"></i> Start
                </button>
                <button class="btn btn-warning btn-sm" onclick="controlSystem('pause')">
                    <i class="fas fa-pause me-1"></i> Pause
                </button>
                <button class="btn btn-danger btn-sm" onclick="controlSystem('stop')">
                    <i class="fas fa-stop me-1"></i> Stop
                </button>
                <button class="btn btn-info btn-sm" onclick="controlSystem('reset')">
                    <i class="fas fa-redo me-1"></i> Reset
                </button>
            </div>
        </div>

        <!-- Status Cards -->
        <div class="row mb-4">
            <div class="col-xl-3 col-md-6">
                <div class="metric-card">
                    <div class="metric-value" id="systemStatus">STOPPED</div>
                    <div class="metric-label">
                        <span class="status-indicator" id="statusIndicator"></span>
                        System Status
                    </div>
                </div>
            </div>
            <div class="col-xl-3 col-md-6">
                <div class="metric-card">
                    <div class="metric-value" id="currentRound">0</div>
                    <div class="metric-label">
                        <i class="fas fa-sync me-1"></i>
                        Training Round
                    </div>
                </div>
            </div>
            <div class="col-xl-3 col-md-6">
                <div class="metric-card">
                    <div class="metric-value" id="activeClients">0</div>
                    <div class="metric-label">
                        <i class="fas fa-users me-1"></i>
                        Active Clients
                    </div>
                </div>
            </div>
            <div class="col-xl-3 col-md-6">
                <div class="metric-card">
                    <div class="metric-value" id="modelAccuracy">0%</div>
                    <div class="metric-label">
                        <i class="fas fa-brain me-1"></i>
                        Model Accuracy
                    </div>
                </div>
            </div>
        </div>

        <!-- Charts -->
        <div class="row mb-4">
            <div class="col-lg-8">
                <div class="card">
                    <div class="card-header">
                        <h5 class="card-title mb-0">
                            <i class="fas fa-chart-line me-2"></i>
                            Real-time Anomaly Detection
                        </h5>
                    </div>
                    <div class="card-body">
                        <div id="anomalyChart" style="height: 400px;"></div>
                    </div>
                </div>
            </div>
            <div class="col-lg-4">
                <div class="card">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <h5 class="card-title mb-0">
                            <i class="fas fa-users me-2"></i>
                            Client Status
                        </h5>
                        <button class="btn btn-gradient btn-sm" onclick="showAddClientModal()">
                            <i class="fas fa-plus me-1"></i> Add Client
                        </button>
                    </div>
                    <div class="card-body" style="max-height: 400px; overflow-y: auto;">
                        <div id="clientList">
                            <div class="text-center text-muted">
                                <i class="fas fa-users fa-2x mb-2"></i>
                                <p>No clients connected</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Client Management Section -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <h5 class="card-title mb-0">
                            <i class="fas fa-network-wired me-2"></i>
                            Client Management
                        </h5>
                        <div class="btn-group">
                            <button class="btn btn-gradient btn-sm" onclick="showAddClientModal()">
                                <i class="fas fa-plus me-1"></i> Add Client
                            </button>
                            <button class="btn btn-outline-danger btn-sm" onclick="simulateAttack()">
                                <i class="fas fa-bolt me-1"></i> Simulate Attack
                            </button>
                        </div>
                    </div>
                    <div class="card-body">
                        <div id="clientGrid" class="row">
                            <!-- Client cards will be populated here -->
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Security Events -->
        <div class="row">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h5 class="card-title mb-0">
                            <i class="fas fa-exclamation-triangle me-2"></i>
                            Security Events
                        </h5>
                    </div>
                    <div class="card-body" style="max-height: 300px; overflow-y: auto;">
                        <div id="securityEvents">
                            <div class="text-center text-muted">
                                <i class="fas fa-shield-alt fa-2x mb-2"></i>
                                <p>No security events</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Add Client Modal -->
    <div class="modal fade" id="addClientModal" tabindex="-1">
        <div class="modal-dialog modal-lg">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">
                        <i class="fas fa-user-plus me-2"></i>
                        Add New Client
                    </h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body">
                    <form id="addClientForm">
                        <div class="row">
                            <div class="col-md-6">
                                <div class="mb-3">
                                    <label class="form-label">Client Name</label>
                                    <input type="text" class="form-control" id="clientName" required 
                                           placeholder="e.g., client_001, hospital_ny">
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="mb-3">
                                    <label class="form-label">Client Type</label>
                                    <select class="form-select" id="clientType" required>
                                        <option value="honest">Honest Client</option>
                                        <option value="suspicious">Suspicious Client</option>
                                        <option value="malicious">Malicious Client</option>
                                    </select>
                                </div>
                            </div>
                        </div>
                        <div class="row">
                            <div class="col-md-6">
                                <div class="mb-3">
                                    <label class="form-label">Location</label>
                                    <select class="form-select" id="clientLocation" required>
                                        <option value="New York,USA,40.7128,-74.0060">New York, USA</option>
                                        <option value="London,UK,51.5074,-0.1278">London, UK</option>
                                        <option value="Tokyo,Japan,35.6762,139.6503">Tokyo, Japan</option>
                                        <option value="Sydney,Australia,-33.8688,151.2093">Sydney, Australia</option>
                                        <option value="Berlin,Germany,52.5200,13.4050">Berlin, Germany</option>
                                        <option value="Toronto,Canada,43.6532,-79.3832">Toronto, Canada</option>
                                        <option value="Singapore,Singapore,1.3521,103.8198">Singapore</option>
                                        <option value="São Paulo,Brazil,-23.5505,-46.6333">São Paulo, Brazil</option>
                                        <option value="Mumbai,India,19.0760,72.8777">Mumbai, India</option>
                                        <option value="Dubai,UAE,25.2048,55.2708">Dubai, UAE</option>
                                    </select>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="mb-3">
                                    <label class="form-label">Capabilities</label>
                                    <div class="form-check-group" style="max-height: 120px; overflow-y: auto;">
                                        <div class="form-check">
                                            <input class="form-check-input" type="checkbox" value="High Performance Computing" id="cap1">
                                            <label class="form-check-label" for="cap1">High Performance Computing</label>
                                        </div>
                                        <div class="form-check">
                                            <input class="form-check-input" type="checkbox" value="GPU Acceleration" id="cap2">
                                            <label class="form-check-label" for="cap2">GPU Acceleration</label>
                                        </div>
                                        <div class="form-check">
                                            <input class="form-check-input" type="checkbox" value="Large Dataset Processing" id="cap3">
                                            <label class="form-check-label" for="cap3">Large Dataset Processing</label>
                                        </div>
                                        <div class="form-check">
                                            <input class="form-check-input" type="checkbox" value="Real-time Analytics" id="cap4">
                                            <label class="form-check-label" for="cap4">Real-time Analytics</label>
                                        </div>
                                        <div class="form-check">
                                            <input class="form-check-input" type="checkbox" value="Edge Computing" id="cap5">
                                            <label class="form-check-label" for="cap5">Edge Computing</label>
                                        </div>
                                        <div class="form-check">
                                            <input class="form-check-input" type="checkbox" value="IoT Integration" id="cap6">
                                            <label class="form-check-label" for="cap6">IoT Integration</label>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="mb-3">
                            <label class="form-label">Description (Optional)</label>
                            <textarea class="form-control" id="clientDescription" rows="3" 
                                      placeholder="Additional information about this client..."></textarea>
                        </div>
                    </form>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                    <button type="button" class="btn btn-gradient" onclick="addClient()">
                        <i class="fas fa-plus me-1"></i> Add Client
                    </button>
                </div>
            </div>
        </div>
    </div>

    <!-- Toast Container -->
    <div class="toast-container position-fixed top-0 end-0 p-3" id="toastContainer"></div>

    <script>
        let socket;
        let dashboardData = {};
        
        // Initialize socket connection
        function initializeSocket() {
            socket = io();
            
            socket.on('connect', function() {
                console.log('Connected to QSFL-CAAD Dashboard');
                showToast('Connected to dashboard', 'success');
            });
            
            socket.on('disconnect', function() {
                console.log('Disconnected from dashboard');
                showToast('Disconnected from dashboard', 'warning');
            });
            
            socket.on('dashboard_update', function(data) {
                dashboardData = data;
                updateDashboard(data);
            });
            
            socket.on('system_message', function(data) {
                showToast(data.message, data.level);
            });
        }
        
        // Control system
        function controlSystem(action) {
            fetch(`/api/control/${action}`, { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    showToast(`System ${action}ed`, 'success');
                })
                .catch(error => {
                    showToast(`Error: ${error}`, 'danger');
                });
        }
        
        // Show add client modal
        function showAddClientModal() {
            const modal = new bootstrap.Modal(document.getElementById('addClientModal'));
            modal.show();
        }
        
        // Add new client
        function addClient() {
            const name = document.getElementById('clientName').value.trim();
            const type = document.getElementById('clientType').value;
            const locationStr = document.getElementById('clientLocation').value;
            const description = document.getElementById('clientDescription').value.trim();
            
            if (!name) {
                showToast('Please enter a client name', 'danger');
                return;
            }
            
            // Parse location
            const [city, country, lat, lng] = locationStr.split(',');
            const location = {
                city: city,
                country: country,
                lat: parseFloat(lat),
                lng: parseFloat(lng)
            };
            
            // Get selected capabilities
            const capabilities = [];
            document.querySelectorAll('#addClientModal .form-check-input:checked').forEach(checkbox => {
                capabilities.push(checkbox.value);
            });
            
            const clientData = {
                name: name,
                type: type,
                location: location,
                description: description,
                capabilities: capabilities
            };
            
            fetch('/api/clients', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(clientData)
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    showToast(data.error, 'danger');
                } else {
                    showToast('Client added successfully!', 'success');
                    bootstrap.Modal.getInstance(document.getElementById('addClientModal')).hide();
                    document.getElementById('addClientForm').reset();
                }
            })
            .catch(error => {
                showToast('Error adding client', 'danger');
            });
        }
        
        // Delete client
        function deleteClient(clientId) {
            if (!confirm(`Are you sure you want to delete client "${clientId}"?`)) {
                return;
            }
            
            fetch(`/api/clients/${clientId}`, { method: 'DELETE' })
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        showToast(data.error, 'danger');
                    } else {
                        showToast('Client deleted successfully', 'success');
                    }
                })
                .catch(error => {
                    showToast('Error deleting client', 'danger');
                });
        }
        
        // Toggle client quarantine
        function toggleQuarantine(clientId) {
            fetch(`/api/clients/${clientId}/quarantine`, { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        showToast(data.error, 'danger');
                    } else {
                        const action = data.quarantined ? 'quarantined' : 'unquarantined';
                        showToast(`Client ${action}`, 'warning');
                    }
                })
                .catch(error => {
                    showToast('Error updating client', 'danger');
                });
        }
        
        // Simulate attack
        function simulateAttack() {
            fetch('/api/simulate_attack', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    type: 'model_poisoning',
                    intensity: 'medium'
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    showToast(data.error, 'danger');
                } else {
                    showToast('Attack simulation started!', 'warning');
                }
            })
            .catch(error => {
                showToast('Error simulating attack', 'danger');
            });
        }
        
        // Update dashboard
        function updateDashboard(data) {
            updateStatusCards(data);
            updateAnomalyChart(data);
            updateClientList(data);
            updateClientGrid(data);
            updateSecurityEvents(data);
        }
        
        // Update status cards
        function updateStatusCards(data) {
            const status = data.system_status || 'stopped';
            document.getElementById('systemStatus').textContent = status.toUpperCase();
            document.getElementById('currentRound').textContent = data.current_round || 0;
            
            const activeClients = Object.values(data.clients || {}).filter(c => 
                c.status === 'active' && !c.quarantined
            ).length;
            document.getElementById('activeClients').textContent = activeClients;
            
            const accuracy = data.metrics?.model_accuracy?.slice(-1)[0] || 0;
            document.getElementById('modelAccuracy').textContent = (accuracy * 100).toFixed(1) + '%';
            
            const indicator = document.getElementById('statusIndicator');
            indicator.className = `status-indicator status-${status}`;
        }
        
        // Update anomaly chart
        function updateAnomalyChart(data) {
            if (!data.metrics?.anomaly_scores) return;
            
            const traces = [];
            const timestamps = data.metrics.timestamps || [];
            
            // Add threshold line
            traces.push({
                x: timestamps,
                y: Array(timestamps.length).fill(0.6),
                name: 'Threshold',
                type: 'scatter',
                mode: 'lines',
                line: { color: 'red', dash: 'dash' }
            });
            
            // Add client traces
            Object.entries(data.metrics.anomaly_scores).forEach(([clientId, scores]) => {
                if (scores.length > 0) {
                    const client = data.clients[clientId] || {};
                    const color = client.type === 'malicious' ? 'red' : 
                                 client.type === 'suspicious' ? 'orange' : 'blue';
                    
                    traces.push({
                        x: timestamps.slice(-scores.length),
                        y: scores,
                        name: clientId,
                        type: 'scatter',
                        mode: 'lines+markers',
                        line: { color: color }
                    });
                }
            });
            
            const layout = {
                title: 'Client Anomaly Scores Over Time',
                xaxis: { title: 'Time' },
                yaxis: { title: 'Anomaly Score', range: [0, 1] },
                margin: { t: 40, r: 40, b: 40, l: 60 }
            };
            
            Plotly.redraw('anomalyChart', traces, layout);
        }
        
        // Update client list
        function updateClientList(data) {
            const container = document.getElementById('clientList');
            const clients = data.clients || {};
            
            if (Object.keys(clients).length === 0) {
                container.innerHTML = `
                    <div class="text-center text-muted">
                        <i class="fas fa-users fa-2x mb-2"></i>
                        <p>No clients connected</p>
                    </div>
                `;
                return;
            }
            
            const clientsHtml = Object.entries(clients).map(([clientId, client]) => {
                const typeClass = client.type;
                const statusIcon = client.quarantined ? 'ban' : 'check-circle';
                const statusColor = client.quarantined ? 'danger' : 'success';
                
                return `
                    <div class="card client-card ${typeClass} mb-2">
                        <div class="card-body p-3">
                            <div class="d-flex justify-content-between align-items-center">
                                <div>
                                    <h6 class="mb-1">${clientId}</h6>
                                    <small class="text-muted">${client.type}</small>
                                </div>
                                <i class="fas fa-${statusIcon} text-${statusColor}"></i>
                            </div>
                            <div class="mt-2">
                                <small>Reputation: ${client.reputation.toFixed(3)}</small><br>
                                <small>Anomaly: ${client.last_anomaly_score.toFixed(3)}</small>
                            </div>
                        </div>
                    </div>
                `;
            }).join('');
            
            container.innerHTML = clientsHtml;
        }
        
        // Update client grid
        function updateClientGrid(data) {
            const container = document.getElementById('clientGrid');
            const clients = data.clients || {};
            
            if (Object.keys(clients).length === 0) {
                container.innerHTML = `
                    <div class="col-12">
                        <div class="text-center text-muted py-5">
                            <i class="fas fa-users fa-3x mb-3"></i>
                            <h5>No Clients Connected</h5>
                            <p>Click "Add Client" to add your first client to the federated learning network.</p>
                        </div>
                    </div>
                `;
                return;
            }
            
            const clientCards = Object.entries(clients).map(([clientId, client]) => {
                const typeColors = {
                    'honest': 'success',
                    'suspicious': 'warning', 
                    'malicious': 'danger'
                };
                
                const typeColor = typeColors[client.type] || 'secondary';
                const statusIcon = client.quarantined ? 'ban' : 'check-circle';
                const statusColor = client.quarantined ? 'danger' : 'success';
                
                const location = client.location || { city: 'Unknown', country: 'Unknown' };
                const capabilities = client.capabilities || [];
                
                return `
                    <div class="col-lg-4 col-md-6 mb-3">
                        <div class="card h-100 client-card ${client.type}">
                            <div class="card-header d-flex justify-content-between align-items-center">
                                <h6 class="mb-0">
                                    <i class="fas fa-desktop me-2"></i>
                                    ${clientId}
                                </h6>
                                <span class="badge bg-${typeColor}">${client.type}</span>
                            </div>
                            <div class="card-body">
                                <div class="row mb-2">
                                    <div class="col-6">
                                        <small class="text-muted">Status</small><br>
                                        <i class="fas fa-${statusIcon} text-${statusColor} me-1"></i>
                                        <small>${client.quarantined ? 'Quarantined' : 'Active'}</small>
                                    </div>
                                    <div class="col-6">
                                        <small class="text-muted">Reputation</small><br>
                                        <small class="fw-bold">${client.reputation.toFixed(3)}</small>
                                    </div>
                                </div>
                                <div class="row mb-2">
                                    <div class="col-6">
                                        <small class="text-muted">Anomaly Score</small><br>
                                        <small class="fw-bold ${client.last_anomaly_score > 0.6 ? 'text-danger' : 'text-success'}">
                                            ${client.last_anomaly_score.toFixed(3)}
                                        </small>
                                    </div>
                                    <div class="col-6">
                                        <small class="text-muted">Updates</small><br>
                                        <small class="fw-bold">${client.updates_sent}</small>
                                    </div>
                                </div>
                                <div class="mb-2">
                                    <small class="text-muted">
                                        <i class="fas fa-map-marker-alt me-1"></i>
                                        ${location.city}, ${location.country}
                                    </small>
                                </div>
                                ${capabilities.length > 0 ? `
                                    <div class="mb-2">
                                        <small class="text-muted">Capabilities:</small><br>
                                        ${capabilities.slice(0, 2).map(cap => 
                                            `<span class="badge bg-light text-dark me-1" style="font-size: 0.7em;">${cap}</span>`
                                        ).join('')}
                                        ${capabilities.length > 2 ? `<small class="text-muted">+${capabilities.length - 2} more</small>` : ''}
                                    </div>
                                ` : ''}
                                ${client.description ? `
                                    <div class="mb-2">
                                        <small class="text-muted">${client.description.substring(0, 60)}${client.description.length > 60 ? '...' : ''}</small>
                                    </div>
                                ` : ''}
                            </div>
                            <div class="card-footer">
                                <div class="btn-group w-100">
                                    <button class="btn btn-sm btn-outline-${client.quarantined ? 'success' : 'warning'}" 
                                            onclick="toggleQuarantine('${clientId}')" title="${client.quarantined ? 'Unquarantine' : 'Quarantine'}">
                                        <i class="fas fa-${client.quarantined ? 'unlock' : 'lock'}"></i>
                                    </button>
                                    <button class="btn btn-sm btn-outline-danger" 
                                            onclick="deleteClient('${clientId}')" title="Delete Client">
                                        <i class="fas fa-trash"></i>
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
            }).join('');
            
            container.innerHTML = clientCards;
        }
        
        // Update security events
        function updateSecurityEvents(data) {
            const container = document.getElementById('securityEvents');
            const events = data.metrics?.security_events || [];
            
            if (events.length === 0) {
                container.innerHTML = `
                    <div class="text-center text-muted">
                        <i class="fas fa-shield-alt fa-2x mb-2"></i>
                        <p>No security events</p>
                    </div>
                `;
                return;
            }
            
            const eventsHtml = events.slice(-5).reverse().map(event => {
                const severityClass = event.severity === 'high' ? 'danger' : 'warning';
                
                return `
                    <div class="alert alert-${severityClass} mb-2">
                        <div class="d-flex justify-content-between">
                            <strong>${event.client_id}</strong>
                            <small>${new Date(event.timestamp).toLocaleTimeString()}</small>
                        </div>
                        <p class="mb-0">${event.description}</p>
                    </div>
                `;
            }).join('');
            
            container.innerHTML = eventsHtml;
        }
        
        // Show toast notification
        function showToast(message, type = 'info') {
            const toastContainer = document.getElementById('toastContainer');
            const toastId = 'toast_' + Date.now();
            
            const typeClass = {
                'success': 'bg-success',
                'warning': 'bg-warning',
                'danger': 'bg-danger',
                'info': 'bg-info'
            }[type] || 'bg-info';
            
            const toastHtml = `
                <div class="toast align-items-center text-white ${typeClass} border-0" role="alert" id="${toastId}">
                    <div class="d-flex">
                        <div class="toast-body">${message}</div>
                        <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
                    </div>
                </div>
            `;
            
            toastContainer.insertAdjacentHTML('beforeend', toastHtml);
            
            const toastElement = document.getElementById(toastId);
            const toast = new bootstrap.Toast(toastElement, { delay: 5000 });
            toast.show();
            
            toastElement.addEventListener('hidden.bs.toast', function() {
                toastElement.remove();
            });
        }
        
        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            initializeSocket();
            
            // Initialize empty chart
            const layout = {
                title: 'Client Anomaly Scores Over Time',
                xaxis: { title: 'Time' },
                yaxis: { title: 'Anomaly Score', range: [0, 1] },
                margin: { t: 40, r: 40, b: 40, l: 60 }
            };
            Plotly.newPlot('anomalyChart', [], layout, {responsive: true});
            
            // Request initial data
            if (socket) {
                socket.emit('request_update');
            }
        });
    </script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

class WorkingDashboard:
    """Working QSFL-CAAD Dashboard."""
    
    def __init__(self):
        """Initialize the dashboard."""
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'qsfl_caad_dashboard'
        CORS(self.app)  # Enable CORS for all routes
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Dashboard data
        self.dashboard_data = {
            'clients': {},
            'metrics': {
                'timestamps': [],
                'anomaly_scores': {},
                'reputation_scores': {},
                'model_accuracy': [],
                'security_events': []
            },
            'system_status': 'stopped',
            'current_round': 0
        }
        
        self.config = {
            'anomaly_threshold': 0.6,
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
                'quarantined': False
            }
    
    def setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/')
        def dashboard():
            return render_template_string(DASHBOARD_HTML)
        
        @self.app.route('/api/dashboard_data')
        def api_dashboard_data():
            return jsonify(self.dashboard_data)
        
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
            elif action == 'pause':
                self.monitoring_active = False
                self.dashboard_data['system_status'] = 'paused'
                return jsonify({'status': 'paused'})
            else:
                return jsonify({'error': 'Unknown action'}), 400
        
        @self.app.route('/api/clients', methods=['GET'])
        def api_get_clients():
            return jsonify(self.dashboard_data['clients'])
        
        @self.app.route('/api/clients', methods=['POST'])
        def api_add_client():
            try:
                data = request.get_json()
                
                # Validate required fields
                if not data or 'name' not in data:
                    return jsonify({'error': 'Client name is required'}), 400
                
                client_id = data['name']
                
                # Check if client already exists
                if client_id in self.dashboard_data['clients']:
                    return jsonify({'error': 'Client already exists'}), 409
                
                # Create new client
                new_client = {
                    'type': data.get('type', 'honest'),
                    'status': 'active',
                    'reputation': 1.0,
                    'last_anomaly_score': 0.0,
                    'updates_sent': 0,
                    'quarantined': False,
                    'location': data.get('location', {'city': 'Unknown', 'country': 'Unknown'}),
                    'description': data.get('description', ''),
                    'capabilities': data.get('capabilities', []),
                    'created_at': datetime.now().isoformat(),
                    'last_update': datetime.now().isoformat(),
                    'model_accuracy': 0.0,
                    'last_round': 0
                }
                
                # Add client to dashboard data
                self.dashboard_data['clients'][client_id] = new_client
                
                # Initialize metrics for new client
                self.dashboard_data['metrics']['anomaly_scores'][client_id] = []
                self.dashboard_data['metrics']['reputation_scores'][client_id] = []
                
                # Broadcast update
                self.socketio.emit('dashboard_update', self.dashboard_data)
                self.broadcast_message(f"✅ New client added: {client_id}", "success")
                
                return jsonify({'message': 'Client added successfully', 'client': new_client}), 201
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/clients/<client_id>', methods=['DELETE'])
        def api_delete_client(client_id):
            try:
                if client_id not in self.dashboard_data['clients']:
                    return jsonify({'error': 'Client not found'}), 404
                
                # Remove client
                del self.dashboard_data['clients'][client_id]
                
                # Remove client metrics
                if client_id in self.dashboard_data['metrics']['anomaly_scores']:
                    del self.dashboard_data['metrics']['anomaly_scores'][client_id]
                if client_id in self.dashboard_data['metrics']['reputation_scores']:
                    del self.dashboard_data['metrics']['reputation_scores'][client_id]
                
                # Broadcast update
                self.socketio.emit('dashboard_update', self.dashboard_data)
                self.broadcast_message(f"🗑️ Client removed: {client_id}", "info")
                
                return jsonify({'message': 'Client deleted successfully'}), 200
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/clients/<client_id>/quarantine', methods=['POST'])
        def api_quarantine_client(client_id):
            try:
                if client_id not in self.dashboard_data['clients']:
                    return jsonify({'error': 'Client not found'}), 404
                
                client = self.dashboard_data['clients'][client_id]
                client['quarantined'] = not client['quarantined']
                
                action = "quarantined" if client['quarantined'] else "unquarantined"
                
                # Create security event
                self.create_security_event(client_id, 'manual_quarantine', client['last_anomaly_score'])
                
                # Broadcast update
                self.socketio.emit('dashboard_update', self.dashboard_data)
                self.broadcast_message(f"⚠️ Client {client_id} {action}", "warning")
                
                return jsonify({'message': f'Client {action} successfully', 'quarantined': client['quarantined']}), 200
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/simulate_attack', methods=['POST'])
        def api_simulate_attack():
            try:
                data = request.get_json() or {}
                attack_type = data.get('type', 'model_poisoning')
                intensity = data.get('intensity', 'medium')
                
                # Find malicious clients or create one
                malicious_clients = [cid for cid, client in self.dashboard_data['clients'].items() 
                                   if client['type'] == 'malicious' and not client['quarantined']]
                
                if not malicious_clients:
                    # Create a temporary malicious client for the attack
                    attack_client_id = f"attacker_{int(time.time())}"
                    self.dashboard_data['clients'][attack_client_id] = {
                        'type': 'malicious',
                        'status': 'active',
                        'reputation': 0.5,
                        'last_anomaly_score': 0.8,
                        'updates_sent': 0,
                        'quarantined': False,
                        'location': {'city': 'Unknown', 'country': 'Unknown'},
                        'description': f'Simulated {attack_type} attacker',
                        'capabilities': ['Attack Simulation'],
                        'created_at': datetime.now().isoformat(),
                        'last_update': datetime.now().isoformat(),
                        'model_accuracy': 0.0,
                        'last_round': self.dashboard_data['current_round']
                    }
                    malicious_clients = [attack_client_id]
                
                # Simulate attack on random malicious client
                target_client = np.random.choice(malicious_clients)
                
                # Create high-severity security event
                self.create_security_event(target_client, attack_type, 0.9)
                
                # Broadcast update
                self.socketio.emit('dashboard_update', self.dashboard_data)
                self.broadcast_message(f"🚨 {attack_type.replace('_', ' ').title()} attack simulated!", "danger")
                
                return jsonify({
                    'message': 'Attack simulation started',
                    'attack_type': attack_type,
                    'target_client': target_client,
                    'intensity': intensity
                }), 200
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
    
    def setup_websockets(self):
        """Setup WebSocket event handlers."""
        
        @self.socketio.on('connect')
        def handle_connect():
            print(f"✅ Client connected: {request.sid}")
            emit('connection_status', {'status': 'connected'})
            emit('dashboard_update', self.dashboard_data)
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            print(f"❌ Client disconnected: {request.sid}")
        
        @self.socketio.on('request_update')
        def handle_request_update():
            emit('dashboard_update', self.dashboard_data)
    
    def start_monitoring(self):
        """Start system monitoring."""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.dashboard_data['system_status'] = 'running'
            
            self.monitor_thread = threading.Thread(target=self.monitoring_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            
            self.broadcast_message("🚀 System monitoring started", "success")
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring_active = False
        self.dashboard_data['system_status'] = 'stopped'
        self.broadcast_message("🛑 System monitoring stopped", "info")
    
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
        
        self.broadcast_message("🔄 System reset completed", "info")
    
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
                        if anomaly_score > 0.8:
                            self.create_security_event(client_id, 'high_anomaly', anomaly_score)
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
                            self.broadcast_message(f"⚠️ Client {client_id} quarantined", "warning")
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
                print(f"❌ Monitoring error: {e}")
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
    
    def run(self, host='127.0.0.1', port=5000, debug=False):
        """Run the dashboard server."""
        print(f"🚀 Starting QSFL-CAAD Dashboard")
        print(f"📊 Dashboard URL: http://{host}:{port}")
        print(f"🛑 Press Ctrl+C to stop")
        print(f"✨ Features: Real-time monitoring, anomaly detection, client management")
        
        try:
            self.socketio.run(self.app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)
        except KeyboardInterrupt:
            print("\n🛑 Dashboard stopped")

def main():
    """Main function."""
    dashboard = WorkingDashboard()
    dashboard.run(host='127.0.0.1', port=5000, debug=False)

if __name__ == "__main__":
    main()