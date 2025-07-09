"""
Visualization module for ExplainIt library.

This module provides functions for creating clear and informative visualizations
of model explanations using matplotlib and seaborn.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union
import warnings

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")


class Visualizer:
    """
    Visualization class for creating explanation plots.
    
    This class provides methods for creating various types of plots
    to visualize model explanations in a clear and informative way.
    """
    
    def __init__(self, style: str = "default"):
        """
        Initialize the visualizer.
        
        Args:
            style: Plot style ("default", "seaborn", "minimal")
        """
        self.style = style
        self._setup_style()
    
    def _setup_style(self):
        """Setup the plotting style."""
        if self.style == "seaborn":
            sns.set_style("whitegrid")
            sns.set_context("notebook", font_scale=1.2)
        elif self.style == "minimal":
            plt.style.use('default')
            plt.rcParams['figure.facecolor'] = 'white'
            plt.rcParams['axes.facecolor'] = 'white'
        else:  # default
            plt.style.use('default')
            sns.set_palette("husl")
    
    def plot_global_explanation(
        self, 
        explanation: Dict[str, Any], 
        top_n: int = 10,
        figsize: tuple = (10, 6),
        title: str = "Feature Importance Analysis",
        **kwargs
    ) -> plt.Figure:
        """
        Plot global explanation (feature importances).
        
        Args:
            explanation: Global explanation dictionary
            top_n: Number of top features to show
            figsize: Figure size
            title: Plot title
            **kwargs: Additional plotting arguments
            
        Returns:
            Matplotlib figure
        """
        feature_importances = explanation["feature_importances"]
        
        # Sort features by importance
        sorted_features = sorted(
            feature_importances.items(), 
            key=lambda x: float(np.ravel(x[1])[0]) if hasattr(x[1], '__iter__') else float(x[1]), 
            reverse=True
        )[:top_n]
        
        features, importances = zip(*sorted_features)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create horizontal bar plot
        bars = ax.barh(range(len(features)), importances, color='skyblue', alpha=0.7)
        
        # Customize plot
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features, fontsize=10)
        ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Add value labels on bars
        for i, (bar, importance) in enumerate(zip(bars, importances)):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{importance:.3f}', va='center', fontsize=9)
        
        # Invert y-axis for better readability
        ax.invert_yaxis()
        
        # Add grid
        ax.grid(axis='x', alpha=0.3)
        
        # Remove spines
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def plot_local_explanation(
        self, 
        explanation: Dict[str, Any],
        top_n: int = 10,
        figsize: tuple = (12, 8),
        title: str = "Local Feature Contribution",
        **kwargs
    ) -> plt.Figure:
        """
        Plot local explanation for a specific sample.
        
        Args:
            explanation: Local explanation dictionary
            top_n: Number of top features to show
            figsize: Figure size
            title: Plot title
            **kwargs: Additional plotting arguments
            
        Returns:
            Matplotlib figure
        """
        feature_contributions = explanation["feature_contributions"]
        
        # Sort features by absolute contribution
        sorted_features = sorted(
            feature_contributions.items(), 
            key=lambda x: float(np.ravel(x[1])[0]) if hasattr(x[1], '__iter__') else float(x[1]), 
            reverse=True
        )[:top_n]
        
        features, contributions = zip(*sorted_features)
        contributions = [float(np.ravel(c)[0]) if hasattr(c, '__iter__') else float(c) for c in contributions]
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[1, 2])
        
        # Plot 1: Prediction summary
        prediction = explanation.get("prediction", "N/A")
        probabilities = explanation.get("probabilities")
        
        if probabilities is not None:
            # Classification case
            classes = [f"Class {i}" for i in range(len(probabilities))]
            bars = ax1.bar(classes, probabilities, color=['lightcoral', 'lightblue'][:len(classes)])
            ax1.set_ylabel('Probability', fontweight='bold')
            ax1.set_title(f'Prediction: {prediction}', fontweight='bold')
            
            # Add probability labels
            for bar, prob in zip(bars, probabilities):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{prob:.3f}', ha='center', va='bottom')
        else:
            # Regression case
            ax1.text(0.5, 0.5, f'Prediction: {prediction:.3f}', 
                    ha='center', va='center', transform=ax1.transAxes, 
                    fontsize=14, fontweight='bold')
            ax1.set_title('Model Prediction', fontweight='bold')
        
        ax1.set_ylim(0, 1 if probabilities is not None else None)
        
        # Plot 2: Feature contributions
        colors = ['red' if c < 0 else 'green' for c in contributions]
        bars = ax2.barh(range(len(features)), contributions, color=colors, alpha=0.7)
        
        # Add zero line
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Customize plot
        ax2.set_yticks(range(len(features)))
        ax2.set_yticklabels(features, fontsize=10)
        ax2.set_xlabel('Feature Contribution', fontsize=12, fontweight='bold')
        ax2.set_title('Feature Contributions to Prediction', fontsize=12, fontweight='bold')
        
        # Add value labels on bars
        for i, (bar, contribution) in enumerate(zip(bars, contributions)):
            ax2.text(bar.get_width() + (0.01 if contribution >= 0 else -0.01), 
                    bar.get_y() + bar.get_height()/2, 
                    f'{contribution:.3f}', 
                    va='center', 
                    ha='left' if contribution >= 0 else 'right',
                    fontsize=9)
        
        # Invert y-axis for better readability
        ax2.invert_yaxis()
        
        # Add grid
        ax2.grid(axis='x', alpha=0.3)
        
        # Remove spines
        for ax in [ax1, ax2]:
            for spine in ax.spines.values():
                spine.set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def plot_shap_summary(
        self, 
        shap_values: np.ndarray,
        X: Union[np.ndarray, pd.DataFrame],
        feature_names: List[str],
        figsize: tuple = (12, 8),
        **kwargs
    ) -> plt.Figure:
        """
        Plot SHAP summary plot.
        
        Args:
            shap_values: SHAP values
            X: Feature data
            feature_names: List of feature names
            figsize: Figure size
            **kwargs: Additional plotting arguments
            
        Returns:
            Matplotlib figure
        """
        try:
            import shap
            
            # Create figure
            fig = plt.figure(figsize=figsize)
            
            # Create SHAP summary plot
            shap.summary_plot(
                shap_values, 
                X, 
                feature_names=feature_names,
                show=False,
                **kwargs
            )
            
            plt.tight_layout()
            return fig
            
        except ImportError:
            warnings.warn("SHAP plotting requires shap library")
            return self._fallback_shap_plot(shap_values, feature_names, figsize)
    
    def _fallback_shap_plot(
        self, 
        shap_values: np.ndarray, 
        feature_names: List[str],
        figsize: tuple = (12, 8)
    ) -> plt.Figure:
        """Fallback SHAP plot when shap library is not available."""
        # Calculate mean absolute SHAP values
        mean_shap = np.mean(np.abs(shap_values), axis=0)
        
        # Sort features
        sorted_indices = np.argsort(mean_shap)[::-1]
        sorted_features = [feature_names[i] for i in sorted_indices]
        sorted_values = mean_shap[sorted_indices]
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        bars = ax.barh(range(len(sorted_features)), sorted_values, color='lightblue', alpha=0.7)
        
        ax.set_yticks(range(len(sorted_features)))
        ax.set_yticklabels(sorted_features)
        ax.set_xlabel('Mean |SHAP Value|', fontweight='bold')
        ax.set_title('SHAP Feature Importance (Fallback)', fontweight='bold')
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, sorted_values)):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{value:.3f}', va='center', fontsize=9)
        
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def plot_feature_interactions(
        self, 
        explanation: Dict[str, Any],
        top_features: int = 5,
        figsize: tuple = (10, 8),
        **kwargs
    ) -> plt.Figure:
        """
        Plot feature interaction analysis (placeholder for future implementation).
        
        Args:
            explanation: Explanation dictionary
            top_features: Number of top features to analyze
            figsize: Figure size
            **kwargs: Additional plotting arguments
            
        Returns:
            Matplotlib figure
        """
        # Placeholder for feature interaction visualization
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.text(0.5, 0.5, 'Feature Interaction Analysis\n(Coming Soon)', 
                ha='center', va='center', transform=ax.transAxes, 
                fontsize=16, fontweight='bold')
        
        ax.set_title('Feature Interactions', fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    def plot_model_performance(
        self, 
        y_true: np.ndarray,
        y_pred: np.ndarray,
        figsize: tuple = (12, 4),
        **kwargs
    ) -> plt.Figure:
        """
        Plot model performance metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            figsize: Figure size
            **kwargs: Additional plotting arguments
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Scatter plot of predictions vs actual
        axes[0].scatter(y_true, y_pred, alpha=0.6, color='skyblue')
        axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                    'r--', lw=2, label='Perfect Prediction')
        axes[0].set_xlabel('Actual Values', fontweight='bold')
        axes[0].set_ylabel('Predicted Values', fontweight='bold')
        axes[0].set_title('Predictions vs Actual', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Residuals
        residuals = y_true - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.6, color='lightcoral')
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted Values', fontweight='bold')
        axes[1].set_ylabel('Residuals', fontweight='bold')
        axes[1].set_title('Residual Plot', fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        # Remove spines
        for ax in axes:
            for spine in ax.spines.values():
                spine.set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def save_plot(self, fig: plt.Figure, filename: str, dpi: int = 300, **kwargs):
        """
        Save plot to file.
        
        Args:
            fig: Matplotlib figure
            filename: Output filename
            dpi: Resolution
            **kwargs: Additional save arguments
        """
        fig.savefig(filename, dpi=dpi, bbox_inches='tight', **kwargs)
        print(f"Plot saved to {filename}")
    
    def show_plot(self, fig: plt.Figure):
        """
        Display plot.
        
        Args:
            fig: Matplotlib figure
        """
        plt.show() 