#!/usr/bin/env python3
"""
Example demonstrating how to use the Dashboard class from ExplainIt.

This script shows how to create and run an interactive Streamlit dashboard
for model explanation and visualization.

"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Add the current directory to the path so we can import explainit
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    print("🚀 ExplainIt Dashboard Example")
    print("=" * 40)
    
    try:
        # Import ExplainIt components
        from explainit import ExplainIt
        from explainit.dashboard import Dashboard, run_dashboard
        
        print("✅ Successfully imported ExplainIt and Dashboard")
        
        # Load and prepare data
        print("\n1. Loading breast cancer dataset...")
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = data.target
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"   Dataset shape: {X.shape}")
        print(f"   Features: {len(data.feature_names)}")
        
        # Train model
        print("\n2. Training Random Forest model...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        print(f"   Model accuracy: {accuracy:.3f}")
        
        # Create ExplainIt explainer
        print("\n3. Creating ExplainIt explainer...")
        explainer = ExplainIt(
            model=model, 
            X=X_test, 
            y=y_test, 
            feature_names=data.feature_names
        )
        
        print("   ✅ ExplainIt explainer created successfully")
        
        # Method 1: Create Dashboard instance
        print("\n4. Method 1: Creating Dashboard instance")
        print("-" * 40)
        
        try:
            dashboard = Dashboard(explainer)
            print("   ✅ Dashboard instance created successfully")
            print("   📊 Dashboard features:")
            print("      - Interactive sidebar with controls")
            print("      - Overview tab with model summary")
            print("      - Global analysis tab with feature importance")
            print("      - Local analysis tab with individual predictions")
            print("      - Reports tab for generating PDF/HTML reports")
            
            # Note: We won't actually run the dashboard here as it requires Streamlit
            print("   💡 To run the dashboard, use: streamlit run dashboard_example.py")
            
        except ImportError as e:
            print(f"   ⚠️ Streamlit not available: {e}")
            print("   💡 Install Streamlit: pip install streamlit")
        
        # Method 2: Using run_dashboard function
        print("\n5. Method 2: Using run_dashboard function")
        print("-" * 40)
        
        try:
            print("   run_dashboard(explainer) - This would launch the Streamlit app")
            print("   ✅ run_dashboard function available")
        except ImportError as e:
            print(f"   ⚠️ Streamlit not available: {e}")
        
        # Method 3: Pre-generate explanations for dashboard
        print("\n6. Method 3: Pre-generating explanations")
        print("-" * 40)
        
        print("   Generating global explanation...")
        global_exp = explainer.explain_global(method="shap")
        print("   ✅ Global explanation generated")
        
        print("   Generating local explanations...")
        for i in range(5):  # Generate 5 local explanations
            explainer.explain_local(sample_index=i, method="shap")
        print("   ✅ Local explanations generated")
        
        # Show what's available in the dashboard
        summary = explainer.get_summary()
        print(f"\n   📊 Dashboard will show:")
        print(f"      - Model type: {summary['model_type']}")
        print(f"      - Dataset: {summary['dataset_shape'][0]} samples, {summary['dataset_shape'][1]} features")
        print(f"      - Global explanation: {summary['has_global_explanation']}")
        print(f"      - Local explanations: {summary['local_explanations_count']}")
        
        # Method 4: Create a simple dashboard script
        print("\n7. Method 4: Creating dashboard script")
        print("-" * 40)
        
        dashboard_script = """#!/usr/bin/env python3
\"\"\"
Simple dashboard script for ExplainIt.
Run this with: streamlit run dashboard_script.py
\"\"\"

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Import ExplainIt
from explainit import ExplainIt
from explainit.dashboard import run_dashboard

def main():
    st.title("ExplainIt Dashboard Example")
    
    # Load data
    with st.spinner("Loading data..."):
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = data.target
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
    # Train model
    with st.spinner("Training model..."):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
    
    # Create explainer
    explainer = ExplainIt(
        model=model, 
        X=X_test, 
        y=y_test, 
        feature_names=data.feature_names
    )
    
    # Run dashboard
    run_dashboard(explainer)

if __name__ == "__main__":
    main()
"""
        
        # Write the dashboard script
        with open("dashboard_script.py", "w", encoding="utf-8") as f:
            f.write(dashboard_script)
        
        print("   ✅ Created dashboard_script.py")
        print("   💡 Run with: streamlit run dashboard_script.py")
        
        # Summary
        print("\n🎉 Dashboard example completed!")
        print("\n📁 Generated files:")
        print("   - dashboard_script.py (ready to run with Streamlit)")
        
        print("\n💡 Key points about the Dashboard class:")
        print("   1. Requires Streamlit: pip install streamlit")
        print("   2. Interactive web interface for model explanations")
        print("   3. Multiple tabs: Overview, Global Analysis, Local Analysis, Reports")
        print("   4. Sidebar controls for generating explanations")
        print("   5. Download capabilities for data and reports")
        print("   6. Real-time visualization and analysis")
        
        print("\n🚀 To run the dashboard:")
        print("   1. Install Streamlit: pip install streamlit")
        print("   2. Run: streamlit run dashboard_script.py")
        print("   3. Open browser to the provided URL")
        
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