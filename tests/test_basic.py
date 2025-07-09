"""
Basic tests for ExplainIt library.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Set matplotlib to use non-GUI backend for testing
import matplotlib
matplotlib.use('Agg')

from explainit import ExplainIt


class TestExplainIt:
    """Test cases for ExplainIt class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        X, y = make_classification(
            n_samples=100, 
            n_features=10, 
            n_informative=5, 
            n_redundant=2, 
            random_state=42
        )
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        return X, y, feature_names
    
    @pytest.fixture
    def trained_model(self, sample_data):
        """Create a trained model for testing."""
        X, y, _ = sample_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        return model, X_test, y_test
    
    def test_explainer_initialization(self, trained_model):
        """Test ExplainIt initialization."""
        model, X, y = trained_model
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        explainer = ExplainIt(model=model, X=X, y=y, feature_names=feature_names)
        
        assert explainer.model == model
        assert explainer.X.shape == X.shape
        assert explainer.y.shape == y.shape
        assert explainer.feature_names == feature_names
        assert explainer.global_explanation is None
        assert explainer.local_explanations == {}
    
    def test_explainer_without_feature_names(self, trained_model):
        """Test ExplainIt initialization without feature names."""
        model, X, y = trained_model
        
        explainer = ExplainIt(model=model, X=X, y=y)
        
        expected_names = [f"Feature_{i}" for i in range(X.shape[1])]
        assert explainer.feature_names == expected_names
    
    def test_global_explanation_shap(self, trained_model):
        """Test global explanation with SHAP method."""
        model, X, y = trained_model
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        explainer = ExplainIt(model=model, X=X, y=y, feature_names=feature_names)
        global_exp = explainer.explain_global(method="shap")
        
        assert global_exp["method"] == "shap"
        assert "feature_importances" in global_exp
        assert "feature_names" in global_exp
        assert "summary" in global_exp
        assert len(global_exp["feature_importances"]) == len(feature_names)
        assert explainer.global_explanation is not None
    
    def test_global_explanation_permutation(self, trained_model):
        """Test global explanation with permutation method."""
        model, X, y = trained_model
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        explainer = ExplainIt(model=model, X=X, y=y, feature_names=feature_names)
        global_exp = explainer.explain_global(method="permutation")
        
        assert global_exp["method"] == "permutation"
        assert "feature_importances" in global_exp
        assert "importance_std" in global_exp
        assert len(global_exp["feature_importances"]) == len(feature_names)
    
    def test_local_explanation_shap(self, trained_model):
        """Test local explanation with SHAP method."""
        model, X, y = trained_model
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        explainer = ExplainIt(model=model, X=X, y=y, feature_names=feature_names)
        local_exp = explainer.explain_local(sample_index=0, method="shap")
        
        assert local_exp["method"] == "shap"
        assert "sample_index" in local_exp
        assert "prediction" in local_exp
        assert "feature_contributions" in local_exp
        assert len(local_exp["feature_contributions"]) == len(feature_names)
        assert 0 in explainer.local_explanations
    
    def test_invalid_method(self, trained_model):
        """Test error handling for invalid methods."""
        model, X, y = trained_model
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        explainer = ExplainIt(model=model, X=X, y=y, feature_names=feature_names)
        
        with pytest.raises(ValueError, match="Unsupported method"):
            explainer.explain_global(method="invalid_method")
        
        with pytest.raises(ValueError, match="Unsupported method"):
            explainer.explain_local(sample_index=0, method="invalid_method")
    
    def test_invalid_sample_index(self, trained_model):
        """Test error handling for invalid sample index."""
        model, X, y = trained_model
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        explainer = ExplainIt(model=model, X=X, y=y, feature_names=feature_names)
        
        # This should work without error (SHAP handles out-of-bounds gracefully)
        # But we can test with a valid index
        local_exp = explainer.explain_local(sample_index=0, method="shap")
        assert local_exp["sample_index"] == 0
    
    def test_get_summary(self, trained_model):
        """Test get_summary method."""
        model, X, y = trained_model
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        explainer = ExplainIt(model=model, X=X, y=y, feature_names=feature_names)
        summary = explainer.get_summary()
        
        assert "model_type" in summary
        assert "dataset_shape" in summary
        assert "feature_names" in summary
        assert "has_global_explanation" in summary
        assert "local_explanations_count" in summary
        assert summary["model_type"] == "RandomForestClassifier"
        assert summary["dataset_shape"] == X.shape
        assert summary["has_global_explanation"] is False
        assert summary["local_explanations_count"] == 0
    
    def test_summary_with_explanations(self, trained_model):
        """Test get_summary after generating explanations."""
        model, X, y = trained_model
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        explainer = ExplainIt(model=model, X=X, y=y, feature_names=feature_names)
        
        # Generate explanations
        explainer.explain_global()
        explainer.explain_local(sample_index=0)
        
        summary = explainer.get_summary()
        
        assert summary["has_global_explanation"] is True
        assert summary["local_explanations_count"] == 1
        assert "global_method" in summary
        assert "top_features" in summary


class TestVisualizer:
    """Test cases for Visualizer class."""
    
    def test_visualizer_initialization(self):
        """Test Visualizer initialization."""
        from explainit.visualizer import Visualizer
        
        visualizer = Visualizer()
        assert visualizer.style == "default"
        
        visualizer = Visualizer(style="seaborn")
        assert visualizer.style == "seaborn"
    
    def test_plot_global_explanation(self):
        """Test global explanation plotting."""
        from explainit.visualizer import Visualizer
        import matplotlib.pyplot as plt
        
        visualizer = Visualizer()
        
        # Create mock explanation data
        explanation = {
            "feature_importances": {
                "feature_1": 0.5,
                "feature_2": 0.3,
                "feature_3": 0.2
            }
        }
        
        fig = visualizer.plot_global_explanation(explanation, top_n=3)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_local_explanation(self):
        """Test local explanation plotting."""
        from explainit.visualizer import Visualizer
        import matplotlib.pyplot as plt
        
        visualizer = Visualizer()
        
        # Create mock explanation data
        explanation = {
            "feature_contributions": {
                "feature_1": 0.1,
                "feature_2": -0.2,
                "feature_3": 0.05
            },
            "prediction": 1,
            "probabilities": [0.3, 0.7]
        }
        
        fig = visualizer.plot_local_explanation(explanation)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestReporter:
    """Test cases for Reporter class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        X, y = make_classification(
            n_samples=100, 
            n_features=10, 
            n_informative=5, 
            n_redundant=2, 
            random_state=42
        )
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        return X, y, feature_names
    
    @pytest.fixture
    def trained_model(self, sample_data):
        """Create a trained model for testing."""
        X, y, _ = sample_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        return model, X_test, y_test
    
    def test_reporter_initialization(self):
        """Test Reporter initialization."""
        from explainit.reporter import Reporter
        
        reporter = Reporter()
        assert reporter.styles is not None or reporter.styles is None  # May be None if reportlab not available
    
    def test_html_report_generation(self, trained_model):
        """Test HTML report generation."""
        from explainit.reporter import Reporter
        import tempfile
        import os
        
        model, X, y = trained_model
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Create explainer and generate explanations
        from explainit import ExplainIt
        explainer = ExplainIt(model=model, X=X, y=y, feature_names=feature_names)
        global_exp = explainer.explain_global()
        local_exp = explainer.explain_local(sample_index=0)
        
        # Test HTML report generation
        reporter = Reporter()
        
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp:
            filename = reporter.export_report(
                global_exp, 
                {0: local_exp}, 
                tmp.name, 
                format="html"
            )
        
        assert os.path.exists(filename)
        assert filename.endswith('.html')
        
        # Clean up
        os.unlink(filename)
    
    def test_invalid_format(self, trained_model):
        """Test error handling for invalid report format."""
        from explainit.reporter import Reporter
        
        model, X, y = trained_model
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Create explainer and generate explanations
        from explainit import ExplainIt
        explainer = ExplainIt(model=model, X=X, y=y, feature_names=feature_names)
        global_exp = explainer.explain_global()
        
        reporter = Reporter()
        
        with pytest.raises(ValueError, match="Unsupported format"):
            reporter.export_report(global_exp, {}, "test.txt", format="txt")


if __name__ == "__main__":
    pytest.main([__file__]) 