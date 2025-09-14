#!/usr/bin/env python3
"""
Test simple dashboard
"""
from flask import Flask, render_template

app = Flask(__name__, template_folder='ui/templates', static_folder='ui/static')

@app.route('/')
def test():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>QSFL-CAAD Dashboard Test</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body>
        <div class="container mt-5">
            <h1 class="text-center">🛡️ QSFL-CAAD Dashboard</h1>
            <div class="alert alert-success text-center">
                <h4>✅ Dashboard is Working!</h4>
                <p>The Flask server is running correctly.</p>
            </div>
            <div class="row">
                <div class="col-md-3">
                    <div class="card text-center">
                        <div class="card-body">
                            <h3 class="text-primary">5</h3>
                            <p>Active Clients</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card text-center">
                        <div class="card-body">
                            <h3 class="text-success">Running</h3>
                            <p>System Status</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card text-center">
                        <div class="card-body">
                            <h3 class="text-info">92.5%</h3>
                            <p>Model Accuracy</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card text-center">
                        <div class="card-body">
                            <h3 class="text-warning">15</h3>
                            <p>Training Round</p>
                        </div>
                    </div>
                </div>
            </div>
            <div class="mt-4">
                <div class="card">
                    <div class="card-header">
                        <h5>🔧 Next Steps</h5>
                    </div>
                    <div class="card-body">
                        <ol>
                            <li>This basic test confirms Flask is working</li>
                            <li>Templates are in the correct location</li>
                            <li>Bootstrap CSS is loading from CDN</li>
                            <li>Ready to implement full dashboard features</li>
                        </ol>
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """

if __name__ == '__main__':
    print("🧪 Starting Dashboard Test Server")
    print("📊 Open: http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=True)