#!/usr/bin/env python3
"""
Example demonstrating how to use the Reporter class from ExplainIt.

This script shows:
1. Using Reporter class directly
2. Using Reporter through ExplainIt class
3. Different output formats (PDF, HTML, Markdown)
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
    print("🚀 ExplainIt Reporter Class Usage Examples")
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
        
        # Example 1: Using Reporter class directly
        print("\n3. Example 1: Using Reporter class directly")
        print("-" * 40)
        
        # Create reporter instance
        reporter = Reporter()
        
        # Create sample explanation data
        global_explanation = {
            "method": "shap",
            "feature_importances": {
                "mean radius": 0.15,
                "mean texture": 0.12,
                "mean perimeter": 0.18,
                "mean area": 0.14,
                "mean smoothness": 0.08,
                "mean compactness": 0.11,
                "mean concavity": 0.09,
                "mean concave points": 0.13
            },
            "feature_names": [
                "mean radius", "mean texture", "mean perimeter", "mean area",
                "mean smoothness", "mean compactness", "mean concavity", "mean concave points"
            ],
            "summary": {
                "total_features": 8,
                "top_features": [
                    ("mean perimeter", 0.18),
                    ("mean radius", 0.15),
                    ("mean area", 0.14),
                    ("mean concave points", 0.13),
                    ("mean texture", 0.12)
                ]
            }
        }
        
        local_explanations = {
            0: {
                "method": "shap",
                "sample_index": 0,
                "prediction": 1,
                "probabilities": [0.2, 0.8],
                "feature_contributions": {
                    "mean radius": 0.05,
                    "mean texture": -0.02,
                    "mean perimeter": 0.08,
                    "mean area": 0.06,
                    "mean smoothness": -0.01,
                    "mean compactness": 0.03,
                    "mean concavity": 0.02,
                    "mean concave points": 0.04
                },
                "feature_names": [
                    "mean radius", "mean texture", "mean perimeter", "mean area",
                    "mean smoothness", "mean compactness", "mean concavity", "mean concave points"
                ]
            },
            1: {
                "method": "shap",
                "sample_index": 1,
                "prediction": 0,
                "probabilities": [0.7, 0.3],
                "feature_contributions": {
                    "mean radius": -0.03,
                    "mean texture": 0.01,
                    "mean perimeter": -0.06,
                    "mean area": -0.04,
                    "mean smoothness": 0.02,
                    "mean compactness": -0.02,
                    "mean concavity": -0.01,
                    "mean concave points": -0.03
                },
                "feature_names": [
                    "mean radius", "mean texture", "mean perimeter", "mean area",
                    "mean smoothness", "mean compactness", "mean concavity", "mean concave points"
                ]
            }
        }
        
        # Generate reports using Reporter directly
        print("   Generating markdown report...")
        markdown_file = reporter.export_report(
            global_explanation=global_explanation,
            local_explanations=local_explanations,
            filename="direct_reporter_example",
            format="markdown",
            title="Direct Reporter Example Report"
        )
        print(f"   ✅ Markdown report: {markdown_file}")
        
        print("   Generating HTML report...")
        html_file = reporter.export_report(
            global_explanation=global_explanation,
            local_explanations=local_explanations,
            filename="direct_reporter_example",
            format="html",
            title="Direct Reporter Example Report"
        )
        print(f"   ✅ HTML report: {html_file}")
        
        # Example 2: Using Reporter through ExplainIt class
        print("\n4. Example 2: Using Reporter through ExplainIt class")
        print("-" * 40)
        
        # Create ExplainIt explainer
        explainer = ExplainIt(
            model=model, 
            X=X_test, 
            y=y_test, 
            feature_names=data.feature_names
        )
        
        print("   ✅ ExplainIt explainer created")
        
        # Generate explanations
        print("   Generating global explanation...")
        global_exp = explainer.explain_global(method="shap")
        
        print("   Generating local explanations...")
        local_exp1 = explainer.explain_local(sample_index=0, method="shap")
        local_exp2 = explainer.explain_local(sample_index=1, method="shap")
        
        print("   ✅ Explanations generated")
        
        # Generate reports using ExplainIt's export_report method
        print("   Generating markdown report...")
        markdown_file2 = explainer.export_report(
            "explainit_reporter_example", 
            format="markdown",
            title="ExplainIt Reporter Example Report"
        )
        print(f"   ✅ Markdown report: {markdown_file2}")
        
        print("   Generating HTML report...")
        html_file2 = explainer.export_report(
            "explainit_reporter_example", 
            format="html",
            title="ExplainIt Reporter Example Report"
        )
        print(f"   ✅ HTML report: {html_file2}")
        
        # Example 3: Accessing Reporter directly from ExplainIt
        print("\n5. Example 3: Accessing Reporter directly from ExplainIt")
        print("-" * 40)
        
        # Access the reporter instance
        explainer_reporter = explainer.reporter
        
        # Use it directly with custom data
        custom_global = {
            "method": "custom",
            "feature_importances": {"feature1": 0.5, "feature2": 0.3, "feature3": 0.2},
            "feature_names": ["feature1", "feature2", "feature3"],
            "summary": {
                "total_features": 3,
                "top_features": [("feature1", 0.5), ("feature2", 0.3), ("feature3", 0.2)]
            }
        }
        
        custom_local = {
            0: {
                "method": "custom",
                "sample_index": 0,
                "prediction": 1,
                "feature_contributions": {"feature1": 0.2, "feature2": 0.1, "feature3": 0.05},
                "feature_names": ["feature1", "feature2", "feature3"]
            }
        }
        
        print("   Generating custom report...")
        custom_file = explainer_reporter.export_report(
            global_explanation=custom_global,
            local_explanations=custom_local,
            filename="custom_reporter_example",
            format="markdown",
            title="Custom Reporter Example"
        )
        print(f"   ✅ Custom report: {custom_file}")
        
        # Summary
        print("\n🎉 All examples completed successfully!")
        print("\n📁 Generated files:")
        print(f"   - {markdown_file}")
        print(f"   - {html_file}")
        print(f"   - {markdown_file2}")
        print(f"   - {html_file2}")
        print(f"   - {custom_file}")
        
        print("\n💡 Key points about using the Reporter class:")
        print("   1. You can use Reporter directly: reporter = Reporter()")
        print("   2. You can access it through ExplainIt: explainer.reporter")
        print("   3. Supported formats: 'pdf', 'html', 'markdown'")
        print("   4. Reports include executive summary, methodology, and explanations")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 