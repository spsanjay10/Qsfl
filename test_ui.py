#!/usr/bin/env python3
"""
Simple test for QSFL-CAAD UI
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from ui.web_dashboard import EnhancedQSFLDashboard
    print("✅ Successfully imported EnhancedQSFLDashboard")
    
    dashboard = EnhancedQSFLDashboard()
    print("✅ Successfully created dashboard instance")
    
    print("🚀 Starting test server on http://localhost:5000")
    print("🛑 Press Ctrl+C to stop")
    
    dashboard.run(host='127.0.0.1', port=5000, debug=True)
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()