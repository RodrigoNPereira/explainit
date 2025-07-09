"""
Interactive dashboard module for ExplainIt library.

This module provides a Streamlit-based dashboard for interactive
model explanation and visualization.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional
import warnings

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    warnings.warn("Streamlit not available. Dashboard will not work. Install with: pip install streamlit")


class Dashboard:
    """
    Interactive dashboard for ExplainIt explanations.
    
    This class provides a Streamlit-based dashboard for exploring
    model explanations interactively.
    """
    
    def __init__(self, explainer):
        """
        Initialize the dashboard.
        
        Args:
            explainer: ExplainIt explainer instance
        """
        if not STREAMLIT_AVAILABLE:
            raise ImportError("Streamlit is required for the dashboard. Install with: pip install streamlit")
        
        self.explainer = explainer
        self._setup_page()
    
    def _setup_page(self):
        """Setup the Streamlit page configuration."""
        st.set_page_config(
            page_title="ExplainIt Dashboard",
            page_icon="🧠",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🧠 ExplainIt Dashboard")
        st.markdown("Interactive model explanation and visualization")
    
    def run(self):
        """Run the interactive dashboard."""
        # Sidebar
        self._create_sidebar()
        
        # Main content
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Overview", 
            "🌍 Global Analysis", 
            "🎯 Local Analysis", 
            "📄 Reports"
        ])
        
        with tab1:
            self._overview_tab()
        
        with tab2:
            self._global_analysis_tab()
        
        with tab3:
            self._local_analysis_tab()
        
        with tab4:
            self._reports_tab()
    
    def _create_sidebar(self):
        """Create the sidebar with controls."""
        st.sidebar.header("🎛️ Controls")
        
        # Model information
        st.sidebar.subheader("Model Info")
        summary = self.explainer.get_summary()
        
        st.sidebar.metric("Model Type", summary["model_type"])
        st.sidebar.metric("Dataset Shape", f"{summary['dataset_shape'][0]} × {summary['dataset_shape'][1]}")
        st.sidebar.metric("Features", len(summary["feature_names"]))
        
        # Global explanation controls
        st.sidebar.subheader("Global Analysis")
        global_method = st.sidebar.selectbox(
            "Method", 
            ["shap", "permutation"], 
            help="Choose explanation method for global analysis"
        )
        
        if st.sidebar.button("Generate Global Explanation", type="primary"):
            with st.spinner("Generating global explanation..."):
                self.explainer.explain_global(method=global_method)
            st.success("Global explanation generated!")
        
        # Local explanation controls
        st.sidebar.subheader("Local Analysis")
        sample_index = st.sidebar.number_input(
            "Sample Index", 
            min_value=0, 
            max_value=len(self.explainer.X) - 1, 
            value=0,
            help="Choose sample index for local explanation"
        )
        
        local_method = st.sidebar.selectbox(
            "Method", 
            ["shap", "lime"], 
            help="Choose explanation method for local analysis"
        )
        
        if st.sidebar.button("Generate Local Explanation"):
            with st.spinner("Generating local explanation..."):
                self.explainer.explain_local(sample_index=sample_index, method=local_method)
            st.success("Local explanation generated!")
    
    def _overview_tab(self):
        """Create the overview tab."""
        st.header("📊 Model Overview")
        
        # Model summary
        summary = self.explainer.get_summary()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Model Type", summary["model_type"])
        
        with col2:
            st.metric("Dataset Size", f"{summary['dataset_shape'][0]:,} samples")
        
        with col3:
            st.metric("Features", f"{summary['dataset_shape'][1]:,}")
        
        # Feature names
        st.subheader("Feature Names")
        feature_df = pd.DataFrame({
            "Feature": summary["feature_names"],
            "Index": range(len(summary["feature_names"]))
        })
        st.dataframe(feature_df, use_container_width=True)
        
        # Data preview
        st.subheader("Data Preview")
        if hasattr(self.explainer.X, 'head'):
            st.dataframe(self.explainer.X.head(), use_container_width=True)
        else:
            st.dataframe(pd.DataFrame(self.explainer.X[:5], columns=summary["feature_names"]), use_container_width=True)
        
        # Explanation status
        st.subheader("Explanation Status")
        col1, col2 = st.columns(2)
        
        with col1:
            if summary["has_global_explanation"]:
                st.success("✅ Global explanation available")
            else:
                st.warning("⚠️ No global explanation generated")
        
        with col2:
            st.info(f"📊 {summary['local_explanations_count']} local explanations generated")
    
    def _global_analysis_tab(self):
        """Create the global analysis tab."""
        st.header("🌍 Global Model Analysis")
        
        if self.explainer.global_explanation is None:
            st.warning("No global explanation available. Generate one using the sidebar controls.")
            return
        
        # Global explanation info
        global_exp = self.explainer.global_explanation
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Method", global_exp["method"].upper())
            st.metric("Total Features", len(global_exp["feature_names"]))
        
        with col2:
            if "summary" in global_exp:
                st.metric("Top Feature", global_exp["summary"]["top_features"][0][0])
                st.metric("Top Importance", f"{global_exp['summary']['top_features'][0][1]:.4f}")
        
        # Feature importance plot
        st.subheader("Feature Importance")
        
        top_n = st.slider("Number of features to show", 5, 20, 10)
        
        fig = self.explainer.plot_global(top_n=top_n)
        st.pyplot(fig)
        
        # Feature importance table
        st.subheader("Feature Importance Table")
        
        feature_importances = global_exp["feature_importances"]
        importance_df = pd.DataFrame([
            {"Feature": feature, "Importance": importance}
            for feature, importance in feature_importances.items()
        ]).sort_values("Importance", ascending=False)
        
        st.dataframe(importance_df, use_container_width=True)
        
        # Download feature importance data
        csv = importance_df.to_csv(index=False)
        st.download_button(
            label="Download Feature Importance CSV",
            data=csv,
            file_name="feature_importance.csv",
            mime="text/csv"
        )
    
    def _local_analysis_tab(self):
        """Create the local analysis tab."""
        st.header("🎯 Local Prediction Analysis")
        
        if not self.explainer.local_explanations:
            st.warning("No local explanations available. Generate one using the sidebar controls.")
            return
        
        # Sample selection
        sample_indices = list(self.explainer.local_explanations.keys())
        selected_sample = st.selectbox("Select Sample", sample_indices)
        
        local_exp = self.explainer.local_explanations[selected_sample]
        
        # Prediction info
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Sample Index", local_exp["sample_index"])
        
        with col2:
            st.metric("Prediction", local_exp["prediction"])
        
        with col3:
            st.metric("Method", local_exp["method"].upper())
        
        # Probabilities (if available)
        if local_exp.get("probabilities") is not None:
            st.subheader("Prediction Probabilities")
            
            prob_df = pd.DataFrame({
                "Class": [f"Class {i}" for i in range(len(local_exp["probabilities"]))],
                "Probability": local_exp["probabilities"]
            })
            
            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.bar(prob_df["Class"], prob_df["Probability"], color=['lightcoral', 'lightblue'][:len(prob_df)])
            ax.set_ylabel("Probability")
            ax.set_title("Class Probabilities")
            
            # Add value labels
            for bar, prob in zip(bars, prob_df["Probability"]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{prob:.3f}', ha='center', va='bottom')
            
            st.pyplot(fig)
        
        # Feature contributions
        st.subheader("Feature Contributions")
        
        top_n = st.slider("Number of features to show", 5, 20, 10, key="local_top_n")
        
        fig = self.explainer.plot_local(selected_sample)
        st.pyplot(fig)
        
        # Feature contributions table
        feature_contributions = local_exp["feature_contributions"]
        contributions_df = pd.DataFrame([
            {"Feature": feature, "Contribution": contribution}
            for feature, contribution in feature_contributions.items()
        ]).sort_values("Contribution", key=abs, ascending=False)
        
        st.dataframe(contributions_df, use_container_width=True)
        
        # Sample values
        st.subheader("Sample Values")
        if hasattr(local_exp["sample"], "to_dict"):
            sample_dict = local_exp["sample"].to_dict()
        else:
            sample_dict = {self.explainer.feature_names[i]: val for i, val in enumerate(local_exp["sample"])}
        
        sample_df = pd.DataFrame([
            {"Feature": feature, "Value": value}
            for feature, value in sample_dict.items()
        ])
        
        st.dataframe(sample_df, use_container_width=True)
    
    def _reports_tab(self):
        """Create the reports tab."""
        st.header("📄 Generate Reports")
        
        # Report configuration
        st.subheader("Report Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            report_title = st.text_input("Report Title", "Model Explanation Report")
            report_format = st.selectbox("Format", ["html", "pdf"])
        
        with col2:
            include_global = st.checkbox("Include Global Analysis", value=True)
            include_local = st.checkbox("Include Local Analysis", value=True)
            max_local_samples = st.number_input("Max Local Samples", 1, 10, 5)
        
        # Generate report
        if st.button("Generate Report", type="primary"):
            if not self.explainer.global_explanation and include_global:
                st.error("Global explanation required but not available. Generate it first.")
                return
            
            if not self.explainer.local_explanations and include_local:
                st.error("Local explanations required but not available. Generate them first.")
                return
            
            with st.spinner("Generating report..."):
                try:
                    # Prepare data for report
                    global_exp = self.explainer.global_explanation if include_global else None
                    local_exps = {}
                    
                    if include_local and self.explainer.local_explanations:
                        # Limit local explanations
                        sample_indices = list(self.explainer.local_explanations.keys())[:max_local_samples]
                        local_exps = {idx: self.explainer.local_explanations[idx] for idx in sample_indices}
                    
                    # Generate report
                    filename = f"explainit_report_{report_format}"
                    self.explainer.reporter.export_report(
                        global_exp, local_exps, filename, format=report_format, title=report_title
                    )
                    
                    st.success(f"Report generated successfully: {filename}.{report_format}")
                    
                    # Download button
                    with open(f"{filename}.{report_format}", "rb") as f:
                        st.download_button(
                            label=f"Download {report_format.upper()} Report",
                            data=f.read(),
                            file_name=f"{filename}.{report_format}",
                            mime="application/pdf" if report_format == "pdf" else "text/html"
                        )
                
                except Exception as e:
                    st.error(f"Error generating report: {e}")


def run_dashboard(explainer):
    """
    Run the ExplainIt dashboard.
    
    Args:
        explainer: ExplainIt explainer instance
    """
    if not STREAMLIT_AVAILABLE:
        raise ImportError("Streamlit is required for the dashboard. Install with: pip install streamlit")
    
    dashboard = Dashboard(explainer)
    dashboard.run()


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append("..")
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    
    # Load data and train model
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Create explainer
    from explainit import ExplainIt
    explainer = ExplainIt(model=model, X=X_test, y=y_test, feature_names=data.feature_names)
    
    # Run dashboard
    run_dashboard(explainer) 