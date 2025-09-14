#!/usr/bin/env python3
"""
Setup script for QSFL-CAAD Enhanced UI
"""
import os
import sys
import subprocess
import json
from pathlib import Path

def install_ui_dependencies():
    """Install UI-specific dependencies."""
    print("🔧 Installing UI dependencies...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "-r", "requirements-ui.txt"
        ])
        print("✅ UI dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install UI dependencies: {e}")
        return False
    return True

def create_directories():
    """Create necessary directories for the UI."""
    print("📁 Creating UI directories...")
    directories = [
        "ui/static/css",
        "ui/static/js", 
        "ui/static/images",
        "ui/templates",
        "logs",
        "data/exports"
    ]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   Created: {directory}")
    print("✅ Directories created successfully")

def create_sample_config():
    """Create sample configuration for the UI."""
    print("⚙️ Creating UI configuration...")
    ui_config = {
        "ui": {
            "host": "0.0.0.0",
            "port": 5000,
            "debug": False,
            "secret_key": "your-secret-key-here-change-in-production",
            "update_interval": 2000,
            "max_data_points": 100
        },
        "dashboard": {
            "theme": "light",
            "auto_refresh": True,
            "show_notifications": True,
            "export_formats": ["json", "csv"]
        }
    }
    
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    config_path = config_dir / "ui_config.json"
    
    with open(config_path, 'w') as f:
        json.dump(ui_config, f, indent=2)
    print(f"✅ UI configuration created: {config_path}")

def main():
    """Main setup function."""
    print("🎨 QSFL-CAAD Enhanced UI Setup")
    print("=" * 40)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    # Run setup steps
    steps = [
        create_directories,
        install_ui_dependencies,
        create_sample_config
    ]
    
    for step in steps:
        try:
            step()
        except Exception as e:
            print(f"❌ Setup step failed: {e}")
            sys.exit(1)
    
    print("\n🎉 UI Setup Complete!")
    print("\nNext steps:")
    print("1. Review and customize config/ui_config.json")
    print("2. Start the UI: python scripts/start_ui.py")
    print("3. Open browser to http://localhost:5000")

if __name__ == "__main__":
    main()