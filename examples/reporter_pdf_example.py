#!/usr/bin/env python3
"""
Example demonstrating direct PDF export using the Reporter class from ExplainIt.

This script shows how to use the Reporter class to generate PDF reports directly,
without needing separate conversion functions.
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
    print("🚀 ExplainIt Reporter PDF Export Example")
    print("=" * 50)
    
    try:
        # Import ExplainIt components
        from explainit import ExplainIt
        from explainit.reporter import Reporter
        
        print("✅ Successfully imported ExplainIt and Reporter")
        
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
        model = RandomForestClassifier(n_estimators=50, random_state=42)
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
        
        # Generate explanations
        print("\n4. Generating explanations...")
        global_exp = explainer.explain_global(method="shap")
        local_exp1 = explainer.explain_local(sample_index=0, method="shap")
        local_exp2 = explainer.explain_local(sample_index=1, method="shap")
        
        print("   ✅ Explanations generated")
        
        # Method 1: Direct PDF export using ExplainIt
        print("\n5. Method 1: Direct PDF export using ExplainIt")
        print("-" * 40)
        
        print("   Generating PDF report via ExplainIt...")
        pdf_file1 = explainer.export_report(
            "breast_cancer_analysis", 
            format="pdf",
            title="Breast Cancer Model Analysis Report"
        )
        print(f"   ✅ PDF report saved: {pdf_file1}")
        
        # Method 2: Direct PDF export using Reporter class
        print("\n6. Method 2: Direct PDF export using Reporter class")
        print("-" * 40)
        
        # Create reporter instance
        reporter = Reporter()
        
        print("   Generating PDF report via Reporter class...")
        pdf_file2 = reporter.export_report(
            global_explanation=global_exp,
            local_explanations=explainer.local_explanations,
            filename="breast_cancer_analysis_direct",
            format="pdf",
            title="Breast Cancer Model Analysis Report (Direct)"
        )
        print(f"   ✅ PDF report saved: {pdf_file2}")
        
        # Method 3: Custom data with Reporter class
        print("\n7. Method 3: Custom data with Reporter class")
        print("-" * 40)
        
        # Create custom explanation data
        custom_global = {
            "method": "custom_analysis",
            "feature_importances": {
                "mean radius": 0.18,
                "mean texture": 0.15,
                "mean perimeter": 0.22,
                "mean area": 0.16,
                "mean smoothness": 0.10,
                "mean compactness": 0.12,
                "mean concavity": 0.07
            },
            "feature_names": [
                "mean radius", "mean texture", "mean perimeter", "mean area",
                "mean smoothness", "mean compactness", "mean concavity"
            ],
            "summary": {
                "total_features": 7,
                "top_features": [
                    ("mean perimeter", 0.22),
                    ("mean radius", 0.18),
                    ("mean area", 0.16),
                    ("mean texture", 0.15),
                    ("mean compactness", 0.12)
                ]
            }
        }
        
        custom_local = {
            0: {
                "method": "custom_analysis",
                "sample_index": 0,
                "prediction": 1,
                "probabilities": [0.25, 0.75],
                "feature_contributions": {
                    "mean radius": 0.06,
                    "mean texture": -0.02,
                    "mean perimeter": 0.09,
                    "mean area": 0.07,
                    "mean smoothness": -0.01,
                    "mean compactness": 0.04,
                    "mean concavity": 0.02
                },
                "feature_names": [
                    "mean radius", "mean texture", "mean perimeter", "mean area",
                    "mean smoothness", "mean compactness", "mean concavity"
                ]
            }
        }
        
        print("   Generating custom PDF report...")
        pdf_file3 = reporter.export_report(
            global_explanation=custom_global,
            local_explanations=custom_local,
            filename="custom_analysis_report",
            format="pdf",
            title="Custom Analysis Report"
        )
        print(f"   ✅ Custom PDF report saved: {pdf_file3}")
        
        # Summary
        print("\n🎉 PDF export examples completed successfully!")
        print("\n📁 Generated PDF files:")
        print(f"   - {pdf_file1}")
        print(f"   - {pdf_file2}")
        print(f"   - {pdf_file3}")
        
        print("\n💡 Key points about PDF export:")
        print("   1. PDF export requires ReportLab: pip install reportlab")
        print("   2. Use format='pdf' in export_report() method")
        print("   3. Reports include executive summary, methodology, and explanations")
        print("   4. Works with both ExplainIt and Reporter classes")
        
        return True
        
    except ImportError as e:
        if "reportlab" in str(e):
            print(f"\n❌ Error: {e}")
            print("\n💡 To enable PDF export, install ReportLab:")
            print("   pip install reportlab")
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