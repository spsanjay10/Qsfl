/**
 * Enhanced Dashboard JavaScript for QSFL-CAAD
 * Provides advanced interactivity and real-time updates
 */

class QSFLDashboard {
    constructor() {
        this.socket = null;
        this.charts = {};
        this.data = {};
        this.config = {
            updateInterval: 2000,
            maxDataPoints: 100,
            animationDuration: 500
        };
        
        this.init();
    }
    
    init() {
        this.initializeSocket();
        this.setupEventListeners();
        this.initializeCharts();
        this.startPeriodicUpdates();
    }
    
    initializeSocket() {
        this.socket = io();
        
        this.socket.on('connect', () => {
            console.log('Connected to QSFL-CAAD Dashboard');
            this.showNotification('Connected to dashboard', 'success');
            this.updateConnectionStatus(true);
        });
        
        this.socket.on('disconnect', () => {
            console.log('Disconnected from dashboard');
            this.showNotification('Connection lost', 'warning');
            this.updateConnectionStatus(false);
        });
        
        this.socket.on('dashboard_update', (data) => {
            this.data = data;
            this.updateAllComponents();
        });
        
        this.socket.on('system_message', (data) => {
            this.showNotification(data.message, data.level);
        });
        
        this.socket.on('client_updated', (data) => {
            this.showNotification(`Client ${data.client_id} ${data.action}`, 'info');
        });
    }
    
    setupEventListeners() {
        // System control buttons
        document.addEventListener('click', (e) => {
            if (e.target.matches('[data-action]')) {
                const action = e.target.dataset.action;
                this.controlSystem(action);
            }
        });
        
        // Configuration changes
        document.addEventListener('change', (e) => {
            if (e.target.matches('[data-config]')) {
                this.updateConfiguration();
            }
        });
        
        // Chart interactions
        document.addEventListener('plotly_click', (eventData) => {
            this.handleChartClick(eventData);
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey || e.metaKey) {
                switch(e.key) {
                    case 's':
                        e.preventDefault();
                        this.controlSystem('start');
                        break;
                    case 'p':
                        e.preventDefault();
                        this.controlSystem('pause');
                        break;
                    case 'r':
                        e.preventDefault();
                        this.controlSystem('reset');
                        break;
                }
            }
        });
    }
    
    initializeCharts() {
        // Initialize all Plotly charts with common configuration
        const commonConfig = {
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
            displaylogo: false
        };
        
        // Anomaly Detection Chart
        if (document.getElementById('anomalyChart')) {
            const layout = {
                title: 'Real-time Anomaly Detection',
                xaxis: { title: 'Time', type: 'date' },
                yaxis: { title: 'Anomaly Score', range: [0, 1] },
                hovermode: 'x unified',
                showlegend: true,
                margin: { t: 50, r: 50, b: 50, l: 60 }
            };
            
            Plotly.newPlot('anomalyChart', [], layout, commonConfig);
        }
        
        // Performance Chart
        if (document.getElementById('performanceChart')) {
            const layout = {
                title: 'Model Performance Trend',
                xaxis: { title: 'Training Round' },
                yaxis: { title: 'Accuracy', range: [0, 1] },
                hovermode: 'x unified',
                margin: { t: 50, r: 50, b: 50, l: 60 }
            };
            
            Plotly.newPlot('performanceChart', [], layout, commonConfig);
        }
        
        // World Map
        if (document.getElementById('worldMap')) {
            const layout = {
                geo: {
                    projection: { type: 'natural earth' },
                    showland: true,
                    landcolor: 'rgb(243, 243, 243)',
                    coastlinecolor: 'rgb(204, 204, 204)',
                    showlakes: true,
                    lakecolor: 'rgb(255, 255, 255)',
                    showocean: true,
                    oceancolor: 'rgb(230, 245, 255)'
                },
                margin: { t: 0, r: 0, b: 0, l: 0 }
            };
            
            Plotly.newPlot('worldMap', [], layout, commonConfig);
        }
    }
    
    updateAllComponents() {
        this.updateStatusCards();
        this.updateCharts();
        this.updateClientList();
        this.updateSecurityEvents();
        this.updateSystemMetrics();
    }
    
    updateStatusCards() {
        const data = this.data;
        
        // System status
        const status = data.system_status || 'stopped';
        const statusElement = document.getElementById('systemStatus');
        if (statusElement) {
            statusElement.textContent = status.toUpperCase();
            statusElement.className = `metric-value text-${this.getStatusColor(status)}`;
        }
        
        // Current round
        const roundElement = document.getElementById('currentRound');
        if (roundElement) {
            roundElement.textContent = data.current_round || 0;
        }
        
        // Active clients
        const activeClients = Object.values(data.clients || {}).filter(c => 
            c.status === 'active' && !c.quarantined
        ).length;
        const activeElement = document.getElementById('activeClients');
        if (activeElement) {
            activeElement.textContent = activeClients;
        }
        
        // Model accuracy
        const accuracy = data.metrics?.model_accuracy?.slice(-1)[0] || 0;
        const accuracyElement = document.getElementById('modelAccuracy');
        if (accuracyElement) {
            accuracyElement.textContent = (accuracy * 100).toFixed(1) + '%';
        }
        
        // Update status indicator
        const indicator = document.getElementById('statusIndicator');
        if (indicator) {
            indicator.className = `status-indicator status-${status}`;
        }
    }
    
    updateCharts() {
        this.updateAnomalyChart();
        this.updatePerformanceChart();
        this.updateWorldMap();
    }
    
    updateAnomalyChart() {
        const chartElement = document.getElementById('anomalyChart');
        if (!chartElement || !this.data.metrics?.anomaly_scores) return;
        
        const traces = [];
        const timestamps = this.data.metrics.timestamps || [];
        
        // Add threshold line
        traces.push({
            x: timestamps,
            y: Array(timestamps.length).fill(0.6),
            name: 'Threshold',
            type: 'scatter',
            mode: 'lines',
            line: { color: 'red', dash: 'dash', width: 2 },
            hovertemplate: 'Threshold: 0.6<extra></extra>'
        });
        
        // Add client traces
        Object.entries(this.data.metrics.anomaly_scores).forEach(([clientId, scores]) => {
            if (scores.length > 0) {
                const client = this.data.clients[clientId] || {};
                const color = this.getClientColor(client.type);
                const opacity = client.quarantined ? 0.5 : 1.0;
                
                traces.push({
                    x: timestamps.slice(-scores.length),
                    y: scores,
                    name: clientId,
                    type: 'scatter',
                    mode: 'lines+markers',
                    line: { color: color, width: 2 },
                    marker: { size: 6 },
                    opacity: opacity,
                    hovertemplate: `${clientId}<br>Score: %{y:.3f}<br>Time: %{x}<extra></extra>`
                });
            }
        });
        
        Plotly.redraw(chartElement, traces);
    }
    
    updatePerformanceChart() {
        const chartElement = document.getElementById('performanceChart');
        if (!chartElement || !this.data.metrics?.model_accuracy) return;
        
        const accuracy = this.data.metrics.model_accuracy;
        const rounds = Array.from({length: accuracy.length}, (_, i) => i + 1);
        
        const trace = {
            x: rounds,
            y: accuracy,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Model Accuracy',
            line: { color: '#28a745', width: 3 },
            marker: { size: 8, color: '#28a745' },
            fill: 'tonexty',
            fillcolor: 'rgba(40, 167, 69, 0.1)',
            hovertemplate: 'Round: %{x}<br>Accuracy: %{y:.3f}<extra></extra>'
        };
        
        Plotly.redraw(chartElement, [trace]);
    }
    
    updateWorldMap() {
        const chartElement = document.getElementById('worldMap');
        if (!chartElement) return;
        
        fetch('/api/world_map_data')
            .then(response => response.json())
            .then(mapData => {
                const trace = {
                    type: 'scattergeo',
                    mode: 'markers',
                    lat: mapData.map(d => d.lat),
                    lon: mapData.map(d => d.lng),
                    text: mapData.map(d => 
                        `<b>${d.client_id}</b><br>` +
                        `${d.city}, ${d.country}<br>` +
                        `Type: ${d.type}<br>` +
                        `Status: ${d.status}<br>` +
                        `Reputation: ${d.reputation.toFixed(3)}<br>` +
                        `Anomaly: ${d.anomaly_score.toFixed(3)}`
                    ),
                    hovertemplate: '%{text}<extra></extra>',
                    marker: {
                        size: mapData.map(d => Math.max(8, d.reputation * 25)),
                        color: mapData.map(d => this.getClientColor(d.type)),
                        opacity: mapData.map(d => d.status === 'quarantined' ? 0.5 : 0.8),
                        line: { color: 'white', width: 2 },
                        symbol: mapData.map(d => d.status === 'quarantined' ? 'x' : 'circle')
                    }
                };
                
                Plotly.redraw(chartElement, [trace]);
            })
            .catch(error => console.error('Error updating world map:', error));
    }
    
    updateClientList() {
        const container = document.getElementById('clientCards');
        if (!container) return;
        
        const clients = this.data.clients || {};
        
        const cardsHtml = Object.entries(clients).map(([clientId, client]) => {
            const typeClass = client.type;
            const statusIcon = client.quarantined ? 'ban' : 'check-circle';
            const statusColor = client.quarantined ? 'danger' : 'success';
            const reputationColor = client.reputation > 0.8 ? 'success' : 
                                   client.reputation > 0.5 ? 'warning' : 'danger';
            
            return `
                <div class="col-lg-4 col-md-6 mb-3 client-filter-item animate__animated animate__fadeInUp" data-type="${client.type}">
                    <div class="card client-card ${typeClass} ${client.quarantined ? 'quarantined' : ''}" data-client-id="${clientId}">
                        <div class="card-body">
                            <div class="d-flex justify-content-between align-items-start mb-3">
                                <div>
                                    <h6 class="card-title mb-1">${clientId}</h6>
                                    <span class="badge bg-${this.getTypeColor(client.type)}">${client.type}</span>
                                </div>
                                <div class="text-end">
                                    <i class="fas fa-${statusIcon} text-${statusColor} fa-lg"></i>
                                    ${client.quarantined ? '<span class="badge bg-warning ms-1">Q</span>' : ''}
                                </div>
                            </div>
                            
                            <div class="row text-center mb-3">
                                <div class="col-4">
                                    <small class="text-muted d-block">Reputation</small>
                                    <strong class="text-${reputationColor}">${client.reputation.toFixed(3)}</strong>
                                </div>
                                <div class="col-4">
                                    <small class="text-muted d-block">Anomaly</small>
                                    <strong class="${client.last_anomaly_score > 0.6 ? 'text-danger' : 'text-success'}">${client.last_anomaly_score.toFixed(3)}</strong>
                                </div>
                                <div class="col-4">
                                    <small class="text-muted d-block">Updates</small>
                                    <strong class="text-info">${client.updates_sent}</strong>
                                </div>
                            </div>
                            
                            <div class="progress mb-2" style="height: 6px;">
                                <div class="progress-bar bg-${reputationColor}" 
                                     style="width: ${client.reputation * 100}%"
                                     data-bs-toggle="tooltip" 
                                     title="Reputation: ${client.reputation.toFixed(3)}"></div>
                            </div>
                            
                            <div class="d-flex justify-content-between align-items-center">
                                <small class="text-muted">
                                    <i class="fas fa-map-marker-alt me-1"></i>
                                    ${client.location?.city || 'Unknown'}
                                </small>
                                <div class="btn-group btn-group-sm">
                                    <button class="btn btn-outline-primary btn-sm" onclick="dashboard.viewClientDetails('${clientId}')">
                                        <i class="fas fa-eye"></i>
                                    </button>
                                    <button class="btn btn-outline-${client.quarantined ? 'success' : 'warning'} btn-sm" 
                                            onclick="dashboard.toggleQuarantine('${clientId}')">
                                        <i class="fas fa-${client.quarantined ? 'unlock' : 'ban'}"></i>
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }).join('');
        
        container.innerHTML = cardsHtml;
        
        // Initialize tooltips
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(tooltipTriggerEl => new bootstrap.Tooltip(tooltipTriggerEl));
    }
    
    updateSecurityEvents() {
        const container = document.getElementById('securityEvents');
        if (!container) return;
        
        const events = this.data.metrics?.security_events || [];
        
        if (events.length === 0) {
            container.innerHTML = `
                <div class="text-center text-muted py-4">
                    <i class="fas fa-shield-alt fa-3x mb-3 text-success"></i>
                    <p>No security events detected</p>
                    <small>System is operating normally</small>
                </div>
            `;
            return;
        }
        
        const eventsHtml = events.slice(-10).reverse().map((event, index) => {
            const severityClass = this.getSeverityColor(event.severity);
            const icon = this.getEventIcon(event.event_type);
            const timeAgo = this.getTimeAgo(event.timestamp);
            
            return `
                <div class="alert alert-${severityClass} alert-custom mb-2 animate__animated animate__fadeInLeft" 
                     style="animation-delay: ${index * 0.1}s">
                    <div class="d-flex align-items-start">
                        <div class="me-3">
                            <i class="fas fa-${icon} fa-lg"></i>
                        </div>
                        <div class="flex-grow-1">
                            <div class="d-flex justify-content-between align-items-start mb-1">
                                <strong>${event.client_id || 'System'}</strong>
                                <small class="text-muted">${timeAgo}</small>
                            </div>
                            <p class="mb-1">${event.description}</p>
                            <div class="d-flex justify-content-between align-items-center">
                                <span class="badge bg-${severityClass}">${event.severity.toUpperCase()}</span>
                                ${event.anomaly_score ? `<small>Score: ${event.anomaly_score.toFixed(3)}</small>` : ''}
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }).join('');
        
        container.innerHTML = eventsHtml;
    }
    
    updateSystemMetrics() {
        // Update various system metrics displays
        const performance = this.data.metrics?.system_performance?.slice(-1)[0] || {};
        
        // CPU usage
        const cpuElement = document.getElementById('cpuUsage');
        if (cpuElement) {
            cpuElement.textContent = (performance.cpu_usage || 0).toFixed(1) + '%';
        }
        
        // Memory usage
        const memoryElement = document.getElementById('memoryUsage');
        if (memoryElement) {
            memoryElement.textContent = (performance.memory_usage || 0).toFixed(1) + '%';
        }
        
        // Network I/O
        const networkElement = document.getElementById('networkIO');
        if (networkElement) {
            networkElement.textContent = this.formatBytes(performance.network_io || 0);
        }
    }
    
    controlSystem(action) {
        fetch(`/api/control/${action}`, { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                this.showNotification(`System ${action}ed successfully`, 'success');
                this.updateSystemStatus(action);
            })
            .catch(error => {
                this.showNotification(`Error: ${error}`, 'danger');
            });
    }
    
    updateConfiguration() {
        const configData = {
            anomaly_threshold: parseFloat(document.getElementById('anomalyThreshold')?.value || 0.6),
            reputation_decay: parseFloat(document.getElementById('reputationDecay')?.value || 0.95),
            quarantine_threshold: parseFloat(document.getElementById('quarantineThreshold')?.value || 0.8),
            update_interval: parseFloat(document.getElementById('updateInterval')?.value || 2.0)
        };
        
        fetch('/api/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(configData)
        })
        .then(response => response.json())
        .then(data => {
            this.showNotification('Configuration updated', 'success');
        })
        .catch(error => {
            this.showNotification(`Configuration error: ${error}`, 'danger');
        });
    }
    
    viewClientDetails(clientId) {
        const client = this.data.clients[clientId];
        if (!client) return;
        
        // Populate modal with client details
        this.populateClientModal(clientId, client);
        
        // Show modal
        const modal = new bootstrap.Modal(document.getElementById('clientModal'));
        modal.show();
    }
    
    toggleQuarantine(clientId) {
        const client = this.data.clients[clientId];
        if (!client) return;
        
        const action = client.quarantined ? 'unquarantine' : 'quarantine';
        
        this.socket.emit('client_action', {
            client_id: clientId,
            action: action
        });
    }
    
    simulateAttack(attackType, intensity = 'medium') {
        fetch('/api/simulate_attack', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ type: attackType, intensity: intensity })
        })
        .then(response => response.json())
        .then(data => {
            this.showNotification(`Simulated ${attackType} attack`, 'warning');
        })
        .catch(error => {
            this.showNotification(`Attack simulation error: ${error}`, 'danger');
        });
    }
    
    // Utility functions
    getStatusColor(status) {
        const colors = {
            'running': 'success',
            'stopped': 'danger',
            'paused': 'warning'
        };
        return colors[status] || 'secondary';
    }
    
    getClientColor(type) {
        const colors = {
            'honest': '#28a745',
            'suspicious': '#ffc107',
            'malicious': '#dc3545'
        };
        return colors[type] || '#6c757d';
    }
    
    getTypeColor(type) {
        const colors = {
            'honest': 'success',
            'suspicious': 'warning',
            'malicious': 'danger'
        };
        return colors[type] || 'secondary';
    }
    
    getSeverityColor(severity) {
        const colors = {
            'low': 'info',
            'medium': 'warning',
            'high': 'danger',
            'critical': 'dark'
        };
        return colors[severity] || 'secondary';
    }
    
    getEventIcon(eventType) {
        const icons = {
            'quarantine': 'ban',
            'high_anomaly': 'exclamation-triangle',
            'authentication_failure': 'lock',
            'crypto_error': 'key',
            'system_error': 'exclamation-circle'
        };
        return icons[eventType] || 'shield-alt';
    }
    
    getTimeAgo(timestamp) {
        const now = new Date();
        const eventTime = new Date(timestamp);
        const diffMs = now - eventTime;
        const diffMins = Math.floor(diffMs / 60000);
        
        if (diffMins < 1) return 'Just now';
        if (diffMins < 60) return `${diffMins}m ago`;
        
        const diffHours = Math.floor(diffMins / 60);
        if (diffHours < 24) return `${diffHours}h ago`;
        
        const diffDays = Math.floor(diffHours / 24);
        return `${diffDays}d ago`;
    }
    
    formatBytes(bytes) {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    showNotification(message, type = 'info', duration = 5000) {
        const toastContainer = document.getElementById('toastContainer') || this.createToastContainer();
        const toastId = 'toast_' + Date.now();
        
        const typeClass = {
            'success': 'bg-success',
            'warning': 'bg-warning',
            'danger': 'bg-danger',
            'info': 'bg-info'
        }[type] || 'bg-info';
        
        const toastHtml = `
            <div class="toast align-items-center text-white ${typeClass} border-0 animate__animated animate__slideInRight" 
                 role="alert" id="${toastId}">
                <div class="d-flex">
                    <div class="toast-body">
                        <i class="fas fa-${this.getNotificationIcon(type)} me-2"></i>
                        ${message}
                    </div>
                    <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
                </div>
            </div>
        `;
        
        toastContainer.insertAdjacentHTML('beforeend', toastHtml);
        
        const toastElement = document.getElementById(toastId);
        const toast = new bootstrap.Toast(toastElement, { delay: duration });
        toast.show();
        
        toastElement.addEventListener('hidden.bs.toast', function() {
            toastElement.remove();
        });
    }
    
    createToastContainer() {
        const container = document.createElement('div');
        container.id = 'toastContainer';
        container.className = 'toast-container position-fixed top-0 end-0 p-3';
        document.body.appendChild(container);
        return container;
    }
    
    getNotificationIcon(type) {
        const icons = {
            'success': 'check-circle',
            'warning': 'exclamation-triangle',
            'danger': 'exclamation-circle',
            'info': 'info-circle'
        };
        return icons[type] || 'info-circle';
    }
    
    updateConnectionStatus(connected) {
        const indicator = document.getElementById('connectionIndicator');
        if (indicator) {
            indicator.className = `status-dot ${connected ? 'online' : 'offline'}`;
        }
    }
    
    updateSystemStatus(action) {
        const statusMap = {
            'start': 'running',
            'stop': 'stopped',
            'pause': 'paused',
            'reset': 'stopped'
        };
        
        this.data.system_status = statusMap[action] || this.data.system_status;
        this.updateStatusCards();
    }
    
    startPeriodicUpdates() {
        setInterval(() => {
            if (this.socket && this.socket.connected) {
                this.socket.emit('request_update');
            }
        }, this.config.updateInterval);
    }
    
    handleChartClick(eventData) {
        // Handle chart click events for drill-down functionality
        console.log('Chart clicked:', eventData);
        
        if (eventData.points && eventData.points.length > 0) {
            const point = eventData.points[0];
            
            // If clicking on a client in anomaly chart, show client details
            if (point.data.name && point.data.name !== 'Threshold') {
                this.viewClientDetails(point.data.name);
            }
        }
    }
    
    exportData(format = 'json') {
        const exportData = {
            timestamp: new Date().toISOString(),
            system_data: this.data,
            configuration: this.config
        };
        
        let blob, filename;
        
        if (format === 'json') {
            blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
            filename = `qsfl-caad-export-${new Date().toISOString().split('T')[0]}.json`;
        } else if (format === 'csv') {
            const csv = this.convertToCSV(exportData);
            blob = new Blob([csv], { type: 'text/csv' });
            filename = `qsfl-caad-export-${new Date().toISOString().split('T')[0]}.csv`;
        }
        
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
        URL.revokeObjectURL(url);
        
        this.showNotification(`Data exported as ${format.toUpperCase()}`, 'success');
    }
    
    convertToCSV(data) {
        // Convert dashboard data to CSV format
        const clients = data.system_data.clients || {};
        const headers = ['Client ID', 'Type', 'Status', 'Reputation', 'Anomaly Score', 'Updates Sent', 'Quarantined'];
        
        const rows = Object.entries(clients).map(([id, client]) => [
            id,
            client.type,
            client.status,
            client.reputation.toFixed(3),
            client.last_anomaly_score.toFixed(3),
            client.updates_sent,
            client.quarantined ? 'Yes' : 'No'
        ]);
        
        return [headers, ...rows].map(row => row.join(',')).join('\n');
    }
}

// Initialize dashboard when DOM is loaded
let dashboard;
document.addEventListener('DOMContentLoaded', function() {
    dashboard = new QSFLDashboard();
});

// Export dashboard instance for global access
window.dashboard = dashboard;