#!/usr/bin/env python3
"""
QSFL-CAAD Enhanced UI Startup Script
"""
import os
import sys
import json
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def load_ui_config():
    """Load UI configuration."""
    config_path = project_root / "config" / "ui_config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    else:
        # Default configuration
        return {
            "ui": {
                "host": "0.0.0.0",
                "port": 5000,
                "debug": False
            }
        }

def main():
    """Start the QSFL-CAAD Enhanced UI."""
    print("🚀 Starting QSFL-CAAD Enhanced UI...")
    
    # Load configuration
    config = load_ui_config()
    ui_config = config.get("ui", {})
    
    # Import and start the web dashboard
    try:
        from ui.web_dashboard import EnhancedQSFLDashboard
        dashboard = EnhancedQSFLDashboard()
        
        host = ui_config.get("host", "0.0.0.0")
        port = ui_config.get("port", 5000)
        debug = ui_config.get("debug", False)
        
        print(f"🌐 Dashboard URL: http://{host}:{port}")
        print("🛑 Press Ctrl+C to stop")
        
        dashboard.run(host=host, port=port, debug=debug)
        
    except ImportError as e:
        print(f"❌ Failed to import web dashboard: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements-ui.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to start UI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()