#!/usr/bin/env python3
"""
Example: Using the Visualizer class from ExplainIt

This script demonstrates how to use the Visualizer class to plot
global and local explanations for a machine learning model.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Import ExplainIt and Visualizer
from explainit import ExplainIt
from explainit.visualizer import Visualizer

def reduce_importances(importances):
    # If importances is a dict of arrays, reduce each to a scalar
    reduced = {}
    for k, v in importances.items():
        arr = np.asarray(v)
        if arr.ndim > 0:
            reduced[k] = float(np.mean(arr))
        else:
            reduced[k] = float(arr)
    return reduced

def main():
    # Load data
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    # Create ExplainIt explainer and generate explanations
    explainer = ExplainIt(model=model, X=X_test, y=y_test, feature_names=data.feature_names)
    global_exp = explainer.explain_global(method="shap")

    # Fix for multi-dimensional feature importances
    if "feature_importances" in global_exp:
        global_exp["feature_importances"] = reduce_importances(global_exp["feature_importances"])

    local_exp = explainer.explain_local(sample_index=0, method="shap")

    # Create Visualizer instance
    visualizer = Visualizer(style="seaborn")

    # Plot global explanation
    fig_global = visualizer.plot_global_explanation(global_exp, top_n=10, title="Top 10 Feature Importances")
    plt.show()

    # Plot local explanation
    fig_local = visualizer.plot_local_explanation(local_exp, top_n=10, title="Local Feature Contributions (Sample 0)")
    plt.show()

if __name__ == "__main__":
    main() 