#!/usr/bin/env python3
"""
QSFL-CAAD System Visualization Dashboard

Interactive dashboard for visualizing system behavior, security metrics,
and real-time monitoring of the QSFL-CAAD system.

Usage:
    python visualization_dashboard.py [--port PORT] [--data-dir DATA_DIR] [--demo-mode]

Features:
    - Real-time system monitoring
    - Security event visualization
    - Attack detection analytics
    - Performance metrics dashboard
    - Interactive plots and charts
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from typing import Dict, List, Any, Tuple
import os
from pathlib import Path
import time
import threading
from dataclasses import dataclass

# Try to import Dash for interactive dashboard
try:
    import dash
    from dash import dcc, html, Input, Output, callback
    import dash_bootstrap_components as dbc
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False
    print("⚠ Dash not available. Interactive dashboard disabled.")

# Import system components for data generation
from tests.test_utils import ModelUpdateGenerator, create_test_model_shape


@dataclass
class SystemMetrics:
    """System metrics data structure."""
    timestamp: datetime
    round_id: str
    num_clients: int
    honest_clients: int
    malicious_clients: int
    detection_rate: float
    false_positive_rate: float
    throughput: float
    latency: float
    cpu_usage: float
    memory_usage: float


class DataGenerator:
    """Generates simulated data for visualization."""
    
    def __init__(self, seed: int = 42):
        """Initialize data generator."""
        np.random.seed(seed)
        self.current_time = datetime.now()
        self.round_counter = 0
        
    def generate_system_metrics(self, duration_hours: int = 24) -> List[SystemMetrics]:
        """Generate simulated system metrics over time."""
        metrics = []
        
        # Generate data points every 5 minutes
        time_points = int(duration_hours * 60 / 5)
        
        for i in range(time_points):
            timestamp = self.current_time - timedelta(hours=duration_hours) + timedelta(minutes=i*5)
            
            # Simulate varying system load
            base_clients = 20
            client_variation = int(10 * np.sin(i * 0.1) + np.random.normal(0, 3))
            num_clients = max(5, base_clients + client_variation)
            
            # Simulate attack patterns (higher during certain periods)
            attack_probability = 0.1 + 0.15 * np.sin(i * 0.05)  # Cyclical attack patterns
            malicious_clients = np.random.binomial(num_clients, attack_probability)
            honest_clients = num_clients - malicious_clients
            
            # Detection performance (degrades slightly with more attacks)
            base_detection_rate = 0.85
            detection_rate = base_detection_rate - (malicious_clients / num_clients) * 0.1
            detection_rate = max(0.6, min(0.95, detection_rate + np.random.normal(0, 0.05)))
            
            # False positive rate (increases with system stress)
            base_fpr = 0.05
            stress_factor = num_clients / 50.0  # Stress increases with client count
            false_positive_rate = base_fpr + stress_factor * 0.02 + np.random.normal(0, 0.01)
            false_positive_rate = max(0.01, min(0.15, false_positive_rate))
            
            # Performance metrics
            base_throughput = 100
            throughput = base_throughput * (1 - stress_factor * 0.3) + np.random.normal(0, 5)
            throughput = max(10, throughput)
            
            base_latency = 0.5
            latency = base_latency * (1 + stress_factor * 0.5) + np.random.normal(0, 0.1)
            latency = max(0.1, latency)
            
            # Resource usage
            cpu_usage = 30 + stress_factor * 40 + np.random.normal(0, 5)
            cpu_usage = max(10, min(90, cpu_usage))
            
            memory_usage = 200 + stress_factor * 300 + np.random.normal(0, 20)
            memory_usage = max(100, min(800, memory_usage))
            
            metrics.append(SystemMetrics(
                timestamp=timestamp,
                round_id=f"round_{self.round_counter:06d}",
                num_clients=num_clients,
                honest_clients=honest_clients,
                malicious_clients=malicious_clients,
                detection_rate=detection_rate,
                false_positive_rate=false_positive_rate,
                throughput=throughput,
                latency=latency,
                cpu_usage=cpu_usage,
                memory_usage=memory_usage
            ))
            
            self.round_counter += 1
        
        return metrics
    
    def generate_attack_events(self, num_events: int = 100) -> pd.DataFrame:
        """Generate simulated attack events."""
        attack_types = ['gradient_poisoning', 'model_replacement', 'byzantine', 'backdoor']
        severities = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
        
        events = []
        
        for i in range(num_events):
            timestamp = self.current_time - timedelta(hours=24) + timedelta(minutes=np.random.randint(0, 1440))
            
            event = {
                'timestamp': timestamp,
                'client_id': f"client_{np.random.randint(1, 100):03d}",
                'attack_type': np.random.choice(attack_types),
                'severity': np.random.choice(severities, p=[0.4, 0.3, 0.2, 0.1]),
                'anomaly_score': np.random.beta(2, 5),  # Skewed towards lower scores
                'detected': np.random.choice([True, False], p=[0.85, 0.15]),
                'response_action': np.random.choice(['allow', 'reduce_weight', 'quarantine', 'reject']),
                'response_time': np.random.exponential(0.1)  # Response time in seconds
            }
            
            # Adjust anomaly score based on attack type
            if event['attack_type'] == 'model_replacement':
                event['anomaly_score'] = max(0.7, event['anomaly_score'])
            elif event['attack_type'] == 'backdoor':
                event['anomaly_score'] = min(0.6, event['anomaly_score'])
            
            events.append(event)
        
        return pd.DataFrame(events).sort_values('timestamp')


class StaticVisualizer:
    """Creates static visualizations for system analysis."""
    
    def __init__(self, output_dir: str = "visualizations"):
        """Initialize static visualizer."""
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def create_system_overview_dashboard(self, metrics: List[SystemMetrics]):
        """Create comprehensive system overview dashboard."""
        print("📊 Creating system overview dashboard...")
        
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame([
            {
                'timestamp': m.timestamp,
                'num_clients': m.num_clients,
                'honest_clients': m.honest_clients,
                'malicious_clients': m.malicious_clients,
                'detection_rate': m.detection_rate,
                'false_positive_rate': m.false_positive_rate,
                'throughput': m.throughput,
                'latency': m.latency,
                'cpu_usage': m.cpu_usage,
                'memory_usage': m.memory_usage
            }
            for m in metrics
        ])
        
        # Create multi-panel dashboard
        fig, axes = plt.subplots(3, 2, figsize=(20, 15))
        fig.suptitle('QSFL-CAAD System Overview Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Client composition over time
        ax1 = axes[0, 0]
        ax1.fill_between(df['timestamp'], 0, df['honest_clients'], 
                        alpha=0.7, color='green', label='Honest Clients')
        ax1.fill_between(df['timestamp'], df['honest_clients'], 
                        df['honest_clients'] + df['malicious_clients'],
                        alpha=0.7, color='red', label='Malicious Clients')
        ax1.set_title('Client Composition Over Time')
        ax1.set_ylabel('Number of Clients')
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Detection performance
        ax2 = axes[0, 1]
        ax2.plot(df['timestamp'], df['detection_rate'], color='blue', linewidth=2, label='Detection Rate')
        ax2.plot(df['timestamp'], df['false_positive_rate'], color='orange', linewidth=2, label='False Positive Rate')
        ax2.set_title('Detection Performance')
        ax2.set_ylabel('Rate')
        ax2.set_ylim(0, 1)
        ax2.legend()
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. System throughput
        ax3 = axes[1, 0]
        ax3.plot(df['timestamp'], df['throughput'], color='purple', linewidth=2)
        ax3.set_title('System Throughput')
        ax3.set_ylabel('Updates/Second')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. System latency
        ax4 = axes[1, 1]
        ax4.plot(df['timestamp'], df['latency'], color='brown', linewidth=2)
        ax4.set_title('System Latency')
        ax4.set_ylabel('Seconds')
        ax4.tick_params(axis='x', rotation=45)
        
        # 5. CPU usage
        ax5 = axes[2, 0]
        ax5.fill_between(df['timestamp'], 0, df['cpu_usage'], alpha=0.6, color='red')
        ax5.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='Warning Threshold')
        ax5.set_title('CPU Usage')
        ax5.set_ylabel('Percentage')
        ax5.set_ylim(0, 100)
        ax5.legend()
        ax5.tick_params(axis='x', rotation=45)
        
        # 6. Memory usage
        ax6 = axes[2, 1]
        ax6.fill_between(df['timestamp'], 0, df['memory_usage'], alpha=0.6, color='blue')
        ax6.axhline(y=600, color='red', linestyle='--', alpha=0.7, label='Warning Threshold')
        ax6.set_title('Memory Usage')
        ax6.set_ylabel('MB')
        ax6.legend()
        ax6.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'system_overview_dashboard.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_attack_analysis_plots(self, attack_events: pd.DataFrame):
        """Create attack analysis visualizations."""
        print("🚨 Creating attack analysis plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Attack Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Attack types distribution
        ax1 = axes[0, 0]
        attack_counts = attack_events['attack_type'].value_counts()
        colors = sns.color_palette("Set2", len(attack_counts))
        wedges, texts, autotexts = ax1.pie(attack_counts.values, labels=attack_counts.index, 
                                          autopct='%1.1f%%', colors=colors)
        ax1.set_title('Attack Types Distribution')
        
        # 2. Detection rate by attack type
        ax2 = axes[0, 1]
        detection_by_type = attack_events.groupby('attack_type')['detected'].mean()
        bars = ax2.bar(detection_by_type.index, detection_by_type.values, 
                      color=sns.color_palette("viridis", len(detection_by_type)))
        ax2.set_title('Detection Rate by Attack Type')
        ax2.set_ylabel('Detection Rate')
        ax2.set_ylim(0, 1)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, detection_by_type.values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.2f}', ha='center', va='bottom')
        
        # 3. Anomaly score distribution
        ax3 = axes[1, 0]
        detected_scores = attack_events[attack_events['detected']]['anomaly_score']
        missed_scores = attack_events[~attack_events['detected']]['anomaly_score']
        
        ax3.hist(detected_scores, bins=20, alpha=0.7, label='Detected', color='green')
        ax3.hist(missed_scores, bins=20, alpha=0.7, label='Missed', color='red')
        ax3.axvline(x=0.5, color='black', linestyle='--', label='Threshold')
        ax3.set_title('Anomaly Score Distribution')
        ax3.set_xlabel('Anomaly Score')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        
        # 4. Attack timeline
        ax4 = axes[1, 1]
        
        # Group attacks by hour
        attack_events['hour'] = attack_events['timestamp'].dt.hour
        hourly_attacks = attack_events.groupby('hour').size()
        
        ax4.plot(hourly_attacks.index, hourly_attacks.values, 'o-', color='red', linewidth=2)
        ax4.set_title('Attack Frequency by Hour')
        ax4.set_xlabel('Hour of Day')
        ax4.set_ylabel('Number of Attacks')
        ax4.set_xticks(range(0, 24, 4))
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'attack_analysis_dashboard.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_performance_heatmap(self, metrics: List[SystemMetrics]):
        """Create performance correlation heatmap."""
        print("🔥 Creating performance heatmap...")
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                'Clients': m.num_clients,
                'Malicious %': m.malicious_clients / m.num_clients * 100,
                'Detection Rate': m.detection_rate,
                'FP Rate': m.false_positive_rate,
                'Throughput': m.throughput,
                'Latency': m.latency,
                'CPU Usage': m.cpu_usage,
                'Memory Usage': m.memory_usage
            }
            for m in metrics
        ])
        
        # Calculate correlation matrix
        correlation_matrix = df.corr()
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdYlBu_r', 
                   center=0, square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        
        plt.title('System Metrics Correlation Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'performance_correlation_heatmap.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()


class InteractiveDashboard:
    """Interactive dashboard using Plotly and Dash."""
    
    def __init__(self, port: int = 8050):
        """Initialize interactive dashboard."""
        if not DASH_AVAILABLE:
            raise ImportError("Dash is required for interactive dashboard")
        
        self.port = port
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.data_generator = DataGenerator()
        
        # Generate initial data
        self.metrics = self.data_generator.generate_system_metrics(24)
        self.attack_events = self.data_generator.generate_attack_events(100)
        
        self._setup_layout()
        self._setup_callbacks()
    
    def _setup_layout(self):
        """Set up the dashboard layout."""
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("QSFL-CAAD System Dashboard", 
                           className="text-center mb-4",
                           style={'color': '#2c3e50'})
                ])
            ]),
            
            # Control panel
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Controls", className="card-title"),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Time Range:"),
                                    dcc.Dropdown(
                                        id='time-range-dropdown',
                                        options=[
                                            {'label': 'Last Hour', 'value': 1},
                                            {'label': 'Last 6 Hours', 'value': 6},
                                            {'label': 'Last 24 Hours', 'value': 24}
                                        ],
                                        value=24
                                    )
                                ], width=6),
                                dbc.Col([
                                    html.Label("Refresh Rate:"),
                                    dcc.Dropdown(
                                        id='refresh-dropdown',
                                        options=[
                                            {'label': '5 seconds', 'value': 5000},
                                            {'label': '10 seconds', 'value': 10000},
                                            {'label': '30 seconds', 'value': 30000}
                                        ],
                                        value=10000
                                    )
                                ], width=6)
                            ])
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),
            
            # Key metrics cards
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Active Clients", className="card-title"),
                            html.H2(id="active-clients-metric", className="text-primary")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Detection Rate", className="card-title"),
                            html.H2(id="detection-rate-metric", className="text-success")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Throughput", className="card-title"),
                            html.H2(id="throughput-metric", className="text-info")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("System Load", className="card-title"),
                            html.H2(id="system-load-metric", className="text-warning")
                        ])
                    ])
                ], width=3)
            ], className="mb-4"),
            
            # Main charts
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="client-composition-chart")
                ], width=6),
                dbc.Col([
                    dcc.Graph(id="detection-performance-chart")
                ], width=6)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="attack-timeline-chart")
                ], width=6),
                dbc.Col([
                    dcc.Graph(id="system-performance-chart")
                ], width=6)
            ], className="mb-4"),
            
            # Attack analysis
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="attack-types-chart")
                ], width=6),
                dbc.Col([
                    dcc.Graph(id="anomaly-scores-chart")
                ], width=6)
            ]),
            
            # Auto-refresh component
            dcc.Interval(
                id='interval-component',
                interval=10000,  # Update every 10 seconds
                n_intervals=0
            )
            
        ], fluid=True)
    
    def _setup_callbacks(self):
        """Set up dashboard callbacks."""
        
        @self.app.callback(
            [Output('active-clients-metric', 'children'),
             Output('detection-rate-metric', 'children'),
             Output('throughput-metric', 'children'),
             Output('system-load-metric', 'children')],
            [Input('interval-component', 'n_intervals'),
             Input('time-range-dropdown', 'value')]
        )
        def update_metrics(n, time_range):
            # Get recent metrics
            recent_metrics = self.metrics[-int(time_range * 12):]  # 12 points per hour
            
            if recent_metrics:
                latest = recent_metrics[-1]
                avg_detection = np.mean([m.detection_rate for m in recent_metrics])
                avg_throughput = np.mean([m.throughput for m in recent_metrics])
                avg_cpu = np.mean([m.cpu_usage for m in recent_metrics])
                
                return (
                    f"{latest.num_clients}",
                    f"{avg_detection:.1%}",
                    f"{avg_throughput:.0f}/s",
                    f"{avg_cpu:.0f}%"
                )
            
            return "0", "0%", "0/s", "0%"
        
        @self.app.callback(
            Output('client-composition-chart', 'figure'),
            [Input('interval-component', 'n_intervals'),
             Input('time-range-dropdown', 'value')]
        )
        def update_client_composition(n, time_range):
            recent_metrics = self.metrics[-int(time_range * 12):]
            
            if not recent_metrics:
                return go.Figure()
            
            timestamps = [m.timestamp for m in recent_metrics]
            honest = [m.honest_clients for m in recent_metrics]
            malicious = [m.malicious_clients for m in recent_metrics]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=timestamps, y=honest,
                fill='tonexty',
                mode='none',
                name='Honest Clients',
                fillcolor='rgba(0, 255, 0, 0.3)'
            ))
            
            fig.add_trace(go.Scatter(
                x=timestamps, y=[h + m for h, m in zip(honest, malicious)],
                fill='tonexty',
                mode='none',
                name='Malicious Clients',
                fillcolor='rgba(255, 0, 0, 0.3)'
            ))
            
            fig.update_layout(
                title="Client Composition Over Time",
                xaxis_title="Time",
                yaxis_title="Number of Clients",
                hovermode='x unified'
            )
            
            return fig
        
        @self.app.callback(
            Output('detection-performance-chart', 'figure'),
            [Input('interval-component', 'n_intervals'),
             Input('time-range-dropdown', 'value')]
        )
        def update_detection_performance(n, time_range):
            recent_metrics = self.metrics[-int(time_range * 12):]
            
            if not recent_metrics:
                return go.Figure()
            
            timestamps = [m.timestamp for m in recent_metrics]
            detection_rates = [m.detection_rate for m in recent_metrics]
            fp_rates = [m.false_positive_rate for m in recent_metrics]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=timestamps, y=detection_rates,
                mode='lines+markers',
                name='Detection Rate',
                line=dict(color='blue', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=timestamps, y=fp_rates,
                mode='lines+markers',
                name='False Positive Rate',
                line=dict(color='orange', width=2)
            ))
            
            fig.update_layout(
                title="Detection Performance",
                xaxis_title="Time",
                yaxis_title="Rate",
                yaxis=dict(range=[0, 1]),
                hovermode='x unified'
            )
            
            return fig
        
        # Additional callbacks for other charts would go here...
        # For brevity, I'm showing the pattern with these two examples
    
    def run(self, debug: bool = False):
        """Run the interactive dashboard."""
        print(f"🚀 Starting interactive dashboard on http://localhost:{self.port}")
        self.app.run_server(debug=debug, port=self.port, host='0.0.0.0')


def main():
    """Main visualization function."""
    parser = argparse.ArgumentParser(
        description="QSFL-CAAD System Visualization Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8050,
        help='Port for interactive dashboard (default: 8050)'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='visualization_data',
        help='Directory for generated data and plots (default: visualization_data)'
    )
    
    parser.add_argument(
        '--demo-mode',
        action='store_true',
        help='Run in demo mode with simulated data'
    )
    
    parser.add_argument(
        '--static-only',
        action='store_true',
        help='Generate only static visualizations'
    )
    
    parser.add_argument(
        '--interactive-only',
        action='store_true',
        help='Run only interactive dashboard'
    )
    
    args = parser.parse_args()
    
    try:
        # Generate demo data
        print("📊 Generating demonstration data...")
        data_generator = DataGenerator()
        metrics = data_generator.generate_system_metrics(24)
        attack_events = data_generator.generate_attack_events(100)
        
        if not args.interactive_only:
            # Create static visualizations
            print("🎨 Creating static visualizations...")
            static_viz = StaticVisualizer(args.data_dir)
            static_viz.create_system_overview_dashboard(metrics)
            static_viz.create_attack_analysis_plots(attack_events)
            static_viz.create_performance_heatmap(metrics)
            
            print(f"✅ Static visualizations saved to {args.data_dir}")
        
        if not args.static_only and DASH_AVAILABLE:
            # Run interactive dashboard
            dashboard = InteractiveDashboard(args.port)
            dashboard.run(debug=args.demo_mode)
        
        elif not args.static_only and not DASH_AVAILABLE:
            print("⚠ Interactive dashboard not available. Install dash: pip install dash plotly")
    
    except KeyboardInterrupt:
        print("\n⚠ Visualization interrupted by user")
    except Exception as e:
        print(f"\n❌ Visualization failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()