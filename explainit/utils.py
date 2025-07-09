"""
Utility functions for ExplainIt library.

This module provides helper functions for validation, data processing,
and other common operations.
"""

import numpy as np
import pandas as pd
from typing import Union, Any
import warnings


def validate_model(model: Any) -> Any:
    """
    Validate that the model has the required methods.
    
    Args:
        model: Machine learning model to validate
        
    Returns:
        The validated model
        
    Raises:
        ValueError: If model doesn't have required methods
    """
    if not hasattr(model, 'predict'):
        raise ValueError("Model must have a 'predict' method")
    
    # Check for common model types
    model_type = type(model).__name__.lower()
    
    # Log model type for debugging
    print(f"Detected model type: {type(model).__name__}")
    
    return model


def validate_data(X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
    """
    Validate and preprocess input data.
    
    Args:
        X: Input data (numpy array or pandas DataFrame)
        
    Returns:
        Validated and preprocessed data
    """
    if X is None:
        raise ValueError("Input data cannot be None")
    
    if isinstance(X, pd.DataFrame):
        # Check for missing values
        if X.isnull().any().any():
            warnings.warn("Data contains missing values. Consider handling them before explanation.")
        
        # Check for infinite values
        if np.isinf(X.select_dtypes(include=[np.number])).any().any():
            warnings.warn("Data contains infinite values. Consider handling them before explanation.")
    
    elif isinstance(X, np.ndarray):
        # Check for missing values
        if np.isnan(X).any():
            warnings.warn("Data contains missing values. Consider handling them before explanation.")
        
        # Check for infinite values
        if np.isinf(X).any():
            warnings.warn("Data contains infinite values. Consider handling them before explanation.")
    
    else:
        raise ValueError("Input data must be a numpy array or pandas DataFrame")
    
    return X


def format_feature_names(feature_names: list) -> list:
    """
    Format feature names for better readability.
    
    Args:
        feature_names: List of feature names
        
    Returns:
        Formatted feature names
    """
    formatted_names = []
    
    for name in feature_names:
        # Convert to string if not already
        name = str(name)
        
        # Replace underscores with spaces
        name = name.replace('_', ' ')
        
        # Capitalize first letter of each word
        name = name.title()
        
        # Limit length
        if len(name) > 30:
            name = name[:27] + "..."
        
        formatted_names.append(name)
    
    return formatted_names


def get_model_info(model: Any) -> dict:
    """
    Extract basic information about the model.
    
    Args:
        model: Machine learning model
        
    Returns:
        Dictionary with model information
    """
    info = {
        "type": type(model).__name__,
        "module": type(model).__module__,
        "has_predict_proba": hasattr(model, 'predict_proba'),
        "has_feature_importances_": hasattr(model, 'feature_importances_'),
        "has_coef_": hasattr(model, 'coef_'),
    }
    
    # Try to get feature importances
    if hasattr(model, 'feature_importances_'):
        info["feature_importances_available"] = True
        info["n_features"] = len(model.feature_importances_)
    elif hasattr(model, 'coef_'):
        info["coefficients_available"] = True
        info["n_features"] = model.coef_.shape[1] if len(model.coef_.shape) > 1 else len(model.coef_)
    else:
        info["feature_importances_available"] = False
        info["coefficients_available"] = False
    
    # Try to get classes for classification
    if hasattr(model, 'classes_'):
        info["classes"] = list(model.classes_)
        info["n_classes"] = len(model.classes_)
        info["is_classifier"] = True
    else:
        info["is_classifier"] = False
    
    return info


def safe_predict(model: Any, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
    """
    Safely make predictions with error handling.
    
    Args:
        model: Machine learning model
        X: Input data
        
    Returns:
        Predictions as numpy array
    """
    try:
        predictions = model.predict(X)
        return np.array(predictions)
    except Exception as e:
        raise ValueError(f"Error making predictions: {e}")


def safe_predict_proba(model: Any, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
    """
    Safely make probability predictions with error handling.
    
    Args:
        model: Machine learning model
        X: Input data
        
    Returns:
        Probability predictions as numpy array
    """
    if not hasattr(model, 'predict_proba'):
        raise ValueError("Model does not support probability predictions")
    
    try:
        probabilities = model.predict_proba(X)
        return np.array(probabilities)
    except Exception as e:
        raise ValueError(f"Error making probability predictions: {e}")


def calculate_prediction_confidence(predictions: np.ndarray, probabilities: np.ndarray = None) -> np.ndarray:
    """
    Calculate confidence scores for predictions.
    
    Args:
        predictions: Model predictions
        probabilities: Prediction probabilities (optional)
        
    Returns:
        Confidence scores
    """
    if probabilities is not None:
        # Use maximum probability as confidence
        return np.max(probabilities, axis=1)
    else:
        # Return ones if no probabilities available
        return np.ones(len(predictions))


def get_feature_statistics(X: Union[np.ndarray, pd.DataFrame]) -> dict:
    """
    Calculate basic statistics for features.
    
    Args:
        X: Input data
        
    Returns:
        Dictionary with feature statistics
    """
    if isinstance(X, pd.DataFrame):
        stats = {
            "n_features": X.shape[1],
            "n_samples": X.shape[0],
            "feature_names": list(X.columns),
            "dtypes": X.dtypes.to_dict(),
            "missing_values": X.isnull().sum().to_dict(),
            "numeric_features": list(X.select_dtypes(include=[np.number]).columns),
            "categorical_features": list(X.select_dtypes(include=['object', 'category']).columns)
        }
        
        # Add numeric statistics
        if len(stats["numeric_features"]) > 0:
            numeric_stats = X[stats["numeric_features"]].describe()
            stats["numeric_statistics"] = numeric_stats.to_dict()
    
    else:  # numpy array
        stats = {
            "n_features": X.shape[1],
            "n_samples": X.shape[0],
            "feature_names": [f"Feature_{i}" for i in range(X.shape[1])],
            "dtype": str(X.dtype),
            "missing_values": np.isnan(X).sum(axis=0).tolist(),
            "numeric_statistics": {
                "mean": np.mean(X, axis=0).tolist(),
                "std": np.std(X, axis=0).tolist(),
                "min": np.min(X, axis=0).tolist(),
                "max": np.max(X, axis=0).tolist()
            }
        }
    
    return stats 