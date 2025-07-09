"""
Basic Usage Example for ExplainIt

This example demonstrates the core functionality of ExplainIt using a simple
classification problem with the breast cancer dataset.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Import ExplainIt
from explainit import ExplainIt

def main():
    print("🚀 ExplainIt Basic Usage Example")
    print("=" * 50)
    
    # 1. Load and prepare data
    print("\n1. Loading breast cancer dataset...")
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"   Dataset shape: {X.shape}")
    print(f"   Features: {len(feature_names)}")
    print(f"   Classes: {len(np.unique(y))}")
    
    # 2. Train a model
    print("\n2. Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"   Model accuracy: {accuracy:.3f}")
    
    # 3. Create ExplainIt explainer
    print("\n3. Creating ExplainIt explainer...")
    explainer = ExplainIt(
        model=model, 
        X=X_test, 
        y=y_test, 
        feature_names=feature_names
    )
    
    # 4. Generate global explanation
    print("\n4. Generating global explanation...")
    global_exp = explainer.explain_global(method="shap")
    
    print("   Top 5 most important features:")
    for i, (feature, importance) in enumerate(global_exp["summary"]["top_features"][:5]):
        print(f"   {i+1}. {feature}: {importance:.4f}")
    
    # 5. Generate local explanation
    print("\n5. Generating local explanation for sample 0...")
    local_exp = explainer.explain_local(sample_index=0, method="shap")
    
    print(f"   Prediction: {local_exp['prediction']}")
    if local_exp.get('probabilities') is not None:
        print(f"   Probabilities: {local_exp['probabilities']}")
    
    print("   Top 3 contributing features:")
    sorted_contributions = sorted(
        local_exp["feature_contributions"].items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    for i, (feature, contribution) in enumerate(sorted_contributions[:3]):
        direction = "positively" if contribution > 0 else "negatively"
        print(f"   {i+1}. {feature}: {direction} ({contribution:.4f})")
    
    # 6. Create visualizations
    print("\n6. Creating visualizations...")
    
    # Global feature importance plot
    fig_global = explainer.plot_global(top_n=10)
    fig_global.savefig("global_feature_importance.png", dpi=300, bbox_inches='tight')
    print("   Saved: global_feature_importance.png")
    
    # Local explanation plot
    fig_local = explainer.plot_local(sample_index=0)
    fig_local.savefig("local_explanation.png", dpi=300, bbox_inches='tight')
    print("   Saved: local_explanation.png")
    
    # 7. Export report
    print("\n7. Exporting report...")
    try:
        explainer.export_report("model_explanation_report.html", format="html")
        print("   Saved: model_explanation_report.html")
    except Exception as e:
        print(f"   HTML report generation failed: {e}")
    
    try:
        explainer.export_report("model_explanation_report.pdf", format="pdf")
        print("   Saved: model_explanation_report.pdf")
    except Exception as e:
        print(f"   PDF report generation failed: {e}")
    
    # 8. Get summary
    print("\n8. Model explanation summary:")
    summary = explainer.get_summary()
    print(f"   Model type: {summary['model_type']}")
    print(f"   Dataset shape: {summary['dataset_shape']}")
    print(f"   Has global explanation: {summary['has_global_explanation']}")
    print(f"   Local explanations count: {summary['local_explanations_count']}")
    
    print("\n✅ Example completed successfully!")
    print("\n📁 Generated files:")
    print("   - global_feature_importance.png")
    print("   - local_explanation.png")
    print("   - model_explanation_report.html")
    print("   - model_explanation_report.pdf")

if __name__ == "__main__":
    main() 