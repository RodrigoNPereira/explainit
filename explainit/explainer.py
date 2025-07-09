"""
Core explainer engine for ExplainIt library.

This module provides the main ExplainIt class that handles model explanations
using SHAP and other explainability techniques.
"""

import numpy as np
import pandas as pd
import shap
from typing import Union, Optional, Dict, Any
import warnings

from .utils import validate_model, validate_data
from .visualizer import Visualizer
from .reporter import Reporter


class ExplainIt:
    """
    Main ExplainIt class for generating model explanations.
    
    This class provides a simple interface for explaining machine learning models
    using various explainability techniques, with a focus on clear visualizations
    and exportable reports.
    """
    
    def __init__(
        self, 
        model: Any, 
        X: Union[np.ndarray, pd.DataFrame], 
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        feature_names: Optional[list] = None
    ):
        """
        Initialize the ExplainIt explainer.
        
        Args:
            model: Trained machine learning model (sklearn, XGBoost, etc.)
            X: Feature data (numpy array or pandas DataFrame)
            y: Target data (optional, for validation)
            feature_names: List of feature names (optional)
        """
        self.model = validate_model(model)
        self.X = validate_data(X)
        self.y = y
        self.feature_names = feature_names if feature_names is not None else self._get_feature_names()
        
        # Initialize SHAP explainer
        self._shap_explainer = None
        self._shap_values = None
        self._shap_base_values = None
        
        # Initialize components
        self.visualizer = Visualizer()
        self.reporter = Reporter()
        
        # Store explanations
        self.global_explanation = None
        self.local_explanations = {}
        
    def _get_feature_names(self) -> list:
        """Extract feature names from the data."""
        if hasattr(self.X, 'columns'):
            return list(self.X.columns)
        else:
            return [f"Feature_{i}" for i in range(self.X.shape[1])]
    
    def _get_shap_explainer(self):
        """Get or create SHAP explainer for the model."""
        if self._shap_explainer is None:
            try:
                # Try TreeExplainer first (for tree-based models)
                self._shap_explainer = shap.TreeExplainer(self.model)
            except:
                try:
                    # Fall back to KernelExplainer
                    self._shap_explainer = shap.KernelExplainer(
                        self.model.predict, 
                        shap.sample(self.X, min(100, len(self.X)))
                    )
                except Exception as e:
                    raise ValueError(f"Could not create SHAP explainer: {e}")
        
        return self._shap_explainer
    
    def explain_global(self, method: str = "shap") -> Dict[str, Any]:
        """
        Generate global explanation (feature importances).
        
        Args:
            method: Explanation method ("shap", "permutation")
            
        Returns:
            Dictionary containing global explanation data
        """
        if method.lower() == "shap":
            return self._explain_global_shap()
        elif method.lower() == "permutation":
            return self._explain_global_permutation()
        else:
            raise ValueError(f"Unsupported method: {method}")
    
    def _explain_global_shap(self) -> Dict[str, Any]:
        """Generate global explanation using SHAP."""
        explainer = self._get_shap_explainer()
        
        # Calculate SHAP values
        if self._shap_values is None:
            self._shap_values = explainer.shap_values(self.X)
            self._shap_base_values = explainer.expected_value
        
        # For classification, use the first class or average across classes
        if isinstance(self._shap_values, list):
            shap_values = np.array(self._shap_values)
            if len(shap_values.shape) == 3:  # Multi-class
                shap_values = np.mean(np.abs(shap_values), axis=0)
            else:  # Binary classification
                shap_values = np.abs(shap_values[0])
        else:
            shap_values = np.abs(self._shap_values)
        
        # Calculate feature importances
        feature_importances = np.mean(shap_values, axis=0)
        
        # Create explanation data
        explanation = {
            "method": "shap",
            "feature_importances": dict(zip(self.feature_names, feature_importances)),
            "feature_names": self.feature_names,
            "shap_values": self._shap_values,
            "base_values": self._shap_base_values,
            "summary": {
                "total_features": len(self.feature_names),
                "top_features": sorted(
                    [
                        (name, float(np.ravel(importance)[0]))
                        for name, importance in zip(self.feature_names, feature_importances)
                    ],
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
            }
        }
        
        self.global_explanation = explanation
        return explanation
    
    def _explain_global_permutation(self) -> Dict[str, Any]:
        """Generate global explanation using permutation importance."""
        from sklearn.inspection import permutation_importance
        
        # Calculate permutation importance
        result = permutation_importance(
            self.model, self.X, self.y, 
            n_repeats=10, random_state=42
        )
        
        feature_importances = result.importances_mean
        
        explanation = {
            "method": "permutation",
            "feature_importances": dict(zip(self.feature_names, feature_importances)),
            "feature_names": self.feature_names,
            "importance_std": dict(zip(self.feature_names, result.importances_std)),
            "summary": {
                "total_features": len(self.feature_names),
                "top_features": sorted(
                    [
                        (name, float(np.ravel(importance)[0]))
                        for name, importance in zip(self.feature_names, feature_importances)
                    ],
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
            }
        }
        
        self.global_explanation = explanation
        return explanation
    
    def explain_local(self, sample_index: int, method: str = "shap") -> Dict[str, Any]:
        """
        Generate local explanation for a specific sample.
        
        Args:
            sample_index: Index of the sample to explain
            method: Explanation method ("shap", "lime")
            
        Returns:
            Dictionary containing local explanation data
        """
        if method.lower() == "shap":
            return self._explain_local_shap(sample_index)
        elif method.lower() == "lime":
            return self._explain_local_lime(sample_index)
        else:
            raise ValueError(f"Unsupported method: {method}")
    
    def _explain_local_shap(self, sample_index: int) -> Dict[str, Any]:
        """Generate local explanation using SHAP."""
        explainer = self._get_shap_explainer()
        
        # Get the sample
        sample = self.X.iloc[sample_index:sample_index+1] if hasattr(self.X, 'iloc') else self.X[sample_index:sample_index+1]
        
        # Calculate SHAP values for this sample
        shap_values = explainer.shap_values(sample)
        
        # Handle different model types
        if isinstance(shap_values, list):
            if len(shap_values) == 2:  # Binary classification
                shap_values = shap_values[1]  # Use positive class
            else:  # Multi-class
                shap_values = shap_values[0]  # Use first class
        
        # Get prediction
        prediction = self.model.predict(sample)[0]
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(sample)[0]
        else:
            probabilities = None
        
        explanation = {
            "method": "shap",
            "sample_index": sample_index,
            "sample": sample.iloc[0] if hasattr(sample, 'iloc') else sample[0],
            "prediction": prediction,
            "probabilities": probabilities,
            "shap_values": shap_values[0],
            "base_value": explainer.expected_value,
            "feature_contributions": dict(zip(self.feature_names, shap_values[0])),
            "feature_names": self.feature_names
        }
        
        self.local_explanations[sample_index] = explanation
        return explanation
    
    def _explain_local_lime(self, sample_index: int) -> Dict[str, Any]:
        """Generate local explanation using LIME."""
        try:
            from lime.lime_tabular import LimeTabularExplainer
        except ImportError:
            raise ImportError("LIME is not installed. Install with: pip install lime")
        
        # Get the sample
        sample = self.X.iloc[sample_index] if hasattr(self.X, 'iloc') else self.X[sample_index]
        
        # Create LIME explainer
        explainer = LimeTabularExplainer(
            self.X.values,
            feature_names=self.feature_names,
            class_names=['class_0', 'class_1'] if hasattr(self.model, 'classes_') else None,
            mode='classification' if hasattr(self.model, 'classes_') else 'regression'
        )
        
        # Generate explanation
        exp = explainer.explain_instance(
            sample.values if hasattr(sample, 'values') else sample,
            self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict,
            num_features=len(self.feature_names)
        )
        
        # Extract feature contributions
        feature_contributions = dict(exp.as_list())
        
        explanation = {
            "method": "lime",
            "sample_index": sample_index,
            "sample": sample,
            "feature_contributions": feature_contributions,
            "feature_names": self.feature_names,
            "explanation": exp
        }
        
        self.local_explanations[sample_index] = explanation
        return explanation
    
    def plot_global(self, top_n: int = 10, **kwargs):
        """Plot global explanation (feature importances)."""
        if self.global_explanation is None:
            self.explain_global()
        
        return self.visualizer.plot_global_explanation(
            self.global_explanation, top_n=top_n, **kwargs
        )
    
    def plot_local(self, sample_index: int, **kwargs):
        """Plot local explanation for a specific sample."""
        if sample_index not in self.local_explanations:
            self.explain_local(sample_index)
        
        return self.visualizer.plot_local_explanation(
            self.local_explanations[sample_index], **kwargs
        )
    
    def export_report(self, filename: str, format: str = "pdf", **kwargs):
        """Export explanation report to file."""
        # Ensure we have explanations
        if self.global_explanation is None:
            self.explain_global()
        
        return self.reporter.export_report(
            self.global_explanation,
            self.local_explanations,
            filename,
            format=format,
            **kwargs
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all explanations."""
        summary = {
            "model_type": type(self.model).__name__,
            "dataset_shape": self.X.shape,
            "feature_names": self.feature_names,
            "has_global_explanation": self.global_explanation is not None,
            "local_explanations_count": len(self.local_explanations)
        }
        
        if self.global_explanation:
            summary["global_method"] = self.global_explanation["method"]
            summary["top_features"] = self.global_explanation["summary"]["top_features"][:5]
        
        return summary 