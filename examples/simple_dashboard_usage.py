#!/usr/bin/env python3
"""
Simple examples of using the Dashboard class from ExplainIt.

This script shows the most common usage patterns for the interactive dashboard.
"""

import sys
import os

# Add the current directory to the path so we can import explainit
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    print("📊 Simple Dashboard Class Usage Examples")
    print("=" * 45)
    
    try:
        # Import the Dashboard class
        from explainit.dashboard import Dashboard, run_dashboard
        
        print("✅ Successfully imported Dashboard class")
        
        # Method 1: Basic Dashboard usage
        print("\n1. Basic Dashboard usage:")
        print("   from explainit.dashboard import Dashboard")
        print("   dashboard = Dashboard(explainer)")
        print("   dashboard.run()")
        
        # Method 2: Using run_dashboard function
        print("\n2. Using run_dashboard function:")
        print("   from explainit.dashboard import run_dashboard")
        print("   run_dashboard(explainer)")
        
        # Method 3: Dashboard features
        print("\n3. Dashboard features:")
        print("   📊 Overview Tab:")
        print("      - Model summary and information")
        print("      - Dataset preview")
        print("      - Feature names list")
        print("      - Explanation status")
        
        print("   🌍 Global Analysis Tab:")
        print("      - Feature importance plots")
        print("      - Feature importance tables")
        print("      - Download feature importance data")
        
        print("   🎯 Local Analysis Tab:")
        print("      - Individual prediction explanations")
        print("      - Feature contribution plots")
        print("      - Sample values display")
        print("      - Prediction probabilities")
        
        print("   📄 Reports Tab:")
        print("      - Generate PDF/HTML reports")
        print("      - Customize report content")
        print("      - Download generated reports")
        
        # Method 4: Sidebar controls
        print("\n4. Sidebar controls:")
        print("   🎛️ Model Information:")
        print("      - Model type and dataset shape")
        print("      - Number of features")
        
        print("   🌍 Global Analysis Controls:")
        print("      - Method selection (SHAP/Permutation)")
        print("      - Generate global explanation button")
        
        print("   🎯 Local Analysis Controls:")
        print("      - Sample index selection")
        print("      - Method selection (SHAP/LIME)")
        print("      - Generate local explanation button")
        
        # Method 5: Requirements and setup
        print("\n5. Requirements and setup:")
        print("   📦 Install Streamlit:")
        print("      pip install streamlit")
        
        print("   🚀 Run dashboard:")
        print("      streamlit run your_script.py")
        
        print("   🌐 Access dashboard:")
        print("      Open browser to provided URL (usually http://localhost:8501)")
        
        # Method 6: Example script structure
        print("\n6. Example script structure:")
        example_script = '''
# Example dashboard script
import streamlit as st
from explainit import ExplainIt
from explainit.dashboard import run_dashboard

# Your model and data setup here
# explainer = ExplainIt(model, X, y, feature_names)

# Run the dashboard
run_dashboard(explainer)
'''
        print(example_script)
        
        print("\n🎉 Simple dashboard examples completed!")
        
        print("\n💡 Key points about the Dashboard class:")
        print("   1. Interactive web interface using Streamlit")
        print("   2. No-code exploration of model explanations")
        print("   3. Real-time generation of explanations")
        print("   4. Built-in visualization and reporting")
        print("   5. Download capabilities for data and reports")
        
        return True
        
    except ImportError as e:
        if "streamlit" in str(e):
            print(f"\n❌ Error: {e}")
            print("\n💡 To enable dashboard functionality, install Streamlit:")
            print("   pip install streamlit")
            return False
        else:
            print(f"\n❌ Import Error: {e}")
            return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 