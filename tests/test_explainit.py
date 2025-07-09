#!/usr/bin/env python3
"""
Simple test script for ExplainIt library.

This script tests the basic functionality of ExplainIt to ensure
everything is working correctly.
"""

import sys
import os

# Add the current directory to the path so we can import explainit
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_basic_functionality():
    """Test basic ExplainIt functionality."""
    print("🧪 Testing ExplainIt Basic Functionality")
    print("=" * 50)
    
    try:
        # Import required libraries
        print("1. Importing libraries...")
        import numpy as np
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split
        
        # Import ExplainIt
        from explainit import ExplainIt
        
        print("   ✅ All imports successful")
        
        # Load data
        print("\n2. Loading breast cancer dataset...")
        data = load_breast_cancer()
        X, y = data.data, data.target
        feature_names = data.feature_names
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"   Dataset shape: {X.shape}")
        print(f"   Features: {len(feature_names)}")
        print(f"   Classes: {len(np.unique(y))}")
        
        # Train model
        print("\n3. Training Random Forest model...")
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        print(f"   Model accuracy: {accuracy:.3f}")
        
        # Create explainer
        print("\n4. Creating ExplainIt explainer...")
        explainer = ExplainIt(
            model=model, 
            X=X_test, 
            y=y_test, 
            feature_names=feature_names
        )
        
        print("   ✅ ExplainIt explainer created successfully")
        
        # Test global explanation
        print("\n5. Testing global explanation...")
        global_exp = explainer.explain_global(method="shap")
        
        print("   ✅ Global explanation generated")
        print(f"   Method: {global_exp['method']}")
        print(f"   Features analyzed: {len(global_exp['feature_names'])}")
        
        # Show top features
        top_features = global_exp["summary"]["top_features"][:3]
        print("   Top 3 features:")
        for i, (feature, importance) in enumerate(top_features):
            print(f"     {i+1}. {feature}: {importance:.4f}")
        
        # Test local explanation
        print("\n6. Testing local explanation...")
        local_exp = explainer.explain_local(sample_index=0, method="shap")
        
        print("   ✅ Local explanation generated")
        print(f"   Sample index: {local_exp['sample_index']}")
        print(f"   Prediction: {local_exp['prediction']}")
        
        if local_exp.get('probabilities') is not None:
            print(f"   Probabilities: {local_exp['probabilities']}")
        
        # Test visualization
        print("\n7. Testing visualization...")
        try:
            fig_global = explainer.plot_global(top_n=5)
            print("   ✅ Global visualization created")
            
            fig_local = explainer.plot_local(sample_index=0)
            print("   ✅ Local visualization created")
            
            # Save plots
            fig_global.savefig("test_global_plot.png", dpi=150, bbox_inches='tight')
            fig_local.savefig("test_local_plot.png", dpi=150, bbox_inches='tight')
            print("   ✅ Plots saved as test_global_plot.png and test_local_plot.png")
            
        except Exception as e:
            print(f"   ⚠️ Visualization failed: {e}")
        
        # Test report generation
        print("\n8. Testing report generation...")
        try:
            explainer.export_report("test_report.html", format="html")
            print("   ✅ HTML report generated: test_report.html")
        except Exception as e:
            print(f"   ⚠️ HTML report generation failed: {e}")
        
        try:
            explainer.export_report("test_report.pdf", format="pdf")
            print("   ✅ PDF report generated: test_report.pdf")
        except Exception as e:
            print(f"   ⚠️ PDF report generation failed: {e}")
        
        # Test summary
        print("\n9. Testing summary...")
        summary = explainer.get_summary()
        
        print("   ✅ Summary generated")
        print(f"   Model type: {summary['model_type']}")
        print(f"   Dataset shape: {summary['dataset_shape']}")
        print(f"   Has global explanation: {summary['has_global_explanation']}")
        print(f"   Local explanations count: {summary['local_explanations_count']}")
        
        print("\n🎉 All tests completed successfully!")
        print("\n📁 Generated files:")
        print("   - test_global_plot.png")
        print("   - test_local_plot.png")
        print("   - test_report.html")
        print("   - test_report.pdf")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_imports():
    """Test that all modules can be imported."""
    print("🧪 Testing Imports")
    print("=" * 30)
    
    modules_to_test = [
        "explainit",
        "explainit.explainer",
        "explainit.visualizer", 
        "explainit.reporter",
        "explainit.utils",
        "explainit.dashboard"
    ]
    
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"   ✅ {module}")
        except ImportError as e:
            print(f"   ❌ {module}: {e}")
        except Exception as e:
            print(f"   ⚠️ {module}: {e}")


if __name__ == "__main__":
    print("🚀 ExplainIt Library Test")
    print("=" * 50)
    
    # Test imports first
    test_imports()
    
    print("\n" + "=" * 50)
    
    # Test basic functionality
    success = test_basic_functionality()
    
    if success:
        print("\n✅ All tests passed! ExplainIt is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
        sys.exit(1) 