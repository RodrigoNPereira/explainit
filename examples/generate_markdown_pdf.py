#!/usr/bin/env python3
"""
Generate Markdown report and convert to PDF using ExplainIt.
"""

import sys
import os
import subprocess

# Add the current directory to the path so we can import explainit
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    print("🚀 Generate Markdown Report and Convert to PDF")
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
        
        # Generate explanations
        print("\n5. Generating explanations...")
        global_exp = explainer.explain_global(method="shap")
        local_exp = explainer.explain_local(sample_index=0, method="shap")
        
        print("   ✅ Explanations generated")
        
        # Generate markdown report
        print("\n6. Generating markdown report...")
        markdown_file = explainer.export_report("model_explanation", format="markdown")
        print(f"   ✅ Markdown report saved: {markdown_file}")
        
        # Convert markdown to PDF
        print("\n7. Converting markdown to PDF...")
        pdf_file = convert_markdown_to_pdf(markdown_file)
        
        if pdf_file:
            print(f"   ✅ PDF report saved: {pdf_file}")
        else:
            print("   ⚠️ PDF conversion failed - markdown file is ready")
        
        print("\n🎉 Report generation completed!")
        print(f"\n📁 Generated files:")
        print(f"   - {markdown_file}")
        if pdf_file:
            print(f"   - {pdf_file}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def convert_markdown_to_pdf(markdown_file):
    """Convert markdown file to PDF using pandoc."""
    try:
        # Check if pandoc is available
        result = subprocess.run(['pandoc', '--version'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            # Convert markdown to PDF using pandoc
            pdf_file = markdown_file.replace('.md', '.pdf')
            cmd = [
                'pandoc', 
                markdown_file, 
                '-o', pdf_file,
                '--pdf-engine=xelatex',
                '--variable=geometry:margin=1in'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                return pdf_file
            else:
                print(f"   Pandoc error: {result.stderr}")
                return None
        else:
            print("   Pandoc not found. Install pandoc to convert to PDF.")
            print("   Download from: https://pandoc.org/installing.html")
            return None
            
    except FileNotFoundError:
        print("   Pandoc not found. Install pandoc to convert to PDF.")
        print("   Download from: https://pandoc.org/installing.html")
        return None
    except Exception as e:
        print(f"   PDF conversion error: {e}")
        return None

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 