"""
Report generation module for ExplainIt library.

This module provides functions for creating comprehensive reports
of model explanations in PDF and HTML formats.
"""

import os
import datetime
from typing import Dict, Any, Optional, List
import warnings
import numpy as np

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    warnings.warn("ReportLab not available. PDF reports will not work. Install with: pip install reportlab")


class Reporter:
    """
    Report generation class for creating explanation reports.
    
    This class provides methods for creating comprehensive reports
    of model explanations in various formats.
    """
    
    def __init__(self):
        """Initialize the reporter."""
        self.styles = None
        if REPORTLAB_AVAILABLE:
            self.styles = getSampleStyleSheet()
    
    def export_report(
        self, 
        global_explanation: Dict[str, Any],
        local_explanations: Dict[int, Dict[str, Any]],
        filename: str,
        format: str = "pdf",
        title: str = "Model Explanation Report",
        **kwargs
    ) -> str:
        """
        Export explanation report to file.
        
        Args:
            global_explanation: Global explanation dictionary
            local_explanations: Dictionary of local explanations
            filename: Output filename
            format: Output format ("pdf", "html", "markdown")
            title: Report title
            **kwargs: Additional report arguments
            
        Returns:
            Path to the generated report
        """
        if format.lower() == "pdf":
            return self._export_pdf_report(
                global_explanation, local_explanations, filename, title, **kwargs
            )
        elif format.lower() == "html":
            return self._export_html_report(
                global_explanation, local_explanations, filename, title, **kwargs
            )
        elif format.lower() == "markdown":
            return self._export_markdown_report(
                global_explanation, local_explanations, filename, title, **kwargs
            )
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_pdf_report(
        self,
        global_explanation: Dict[str, Any],
        local_explanations: Dict[int, Dict[str, Any]],
        filename: str,
        title: str,
        **kwargs
    ) -> str:
        """Export report as PDF."""
        if not REPORTLAB_AVAILABLE:
            raise ImportError("ReportLab is required for PDF reports. Install with: pip install reportlab")
        
        # Ensure filename has .pdf extension
        if not filename.endswith('.pdf'):
            filename += '.pdf'
        
        # Create PDF document
        doc = SimpleDocTemplate(filename, pagesize=A4)
        story = []
        
        # Add title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1  # Center alignment
        )
        story.append(Paragraph(title, title_style))
        story.append(Spacer(1, 20))
        
        # Add timestamp
        timestamp_style = ParagraphStyle(
            'Timestamp',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.grey
        )
        timestamp = f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        story.append(Paragraph(timestamp, timestamp_style))
        story.append(Spacer(1, 30))
        
        # Add executive summary
        story.append(Paragraph("Executive Summary", self.styles['Heading2']))
        story.append(Spacer(1, 12))
        
        summary_text = self._generate_executive_summary(global_explanation)
        story.append(Paragraph(summary_text, self.styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Add global explanation section
        story.append(Paragraph("Global Model Explanation", self.styles['Heading2']))
        story.append(Spacer(1, 12))
        
        global_text = self._generate_global_explanation_text(global_explanation)
        story.append(Paragraph(global_text, self.styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Add feature importance table
        if "feature_importances" in global_explanation:
            story.append(Paragraph("Top Feature Importances", self.styles['Heading3']))
            story.append(Spacer(1, 12))
            
            table_data = self._create_feature_importance_table(global_explanation)
            table = Table(table_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(table)
            story.append(Spacer(1, 20))
        
        # Add local explanations section
        if local_explanations:
            story.append(Paragraph("Local Explanations", self.styles['Heading2']))
            story.append(Spacer(1, 12))
            
            for sample_idx, local_exp in list(local_explanations.items())[:5]:  # Limit to first 5
                story.append(Paragraph(f"Sample {sample_idx}", self.styles['Heading3']))
                story.append(Spacer(1, 12))
                
                local_text = self._generate_local_explanation_text(local_exp)
                story.append(Paragraph(local_text, self.styles['Normal']))
                story.append(Spacer(1, 15))
        
        # Add methodology section
        story.append(Paragraph("Methodology", self.styles['Heading2']))
        story.append(Spacer(1, 12))
        
        methodology_text = self._generate_methodology_text(global_explanation)
        story.append(Paragraph(methodology_text, self.styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Build PDF
        doc.build(story)
        print(f"PDF report saved to: {filename}")
        return filename
    
    def _export_html_report(
        self,
        global_explanation: Dict[str, Any],
        local_explanations: Dict[int, Dict[str, Any]],
        filename: str,
        title: str,
        **kwargs
    ) -> str:
        """Export report as HTML."""
        # Ensure filename has .html extension
        if not filename.endswith('.html'):
            filename += '.html'
        
        # Generate HTML content
        html_content = self._generate_html_content(
            global_explanation, local_explanations, title
        )
        
        # Write to file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"HTML report saved to: {filename}")
        return filename
    
    def _generate_executive_summary(self, global_explanation: Dict[str, Any]) -> str:
        """Generate executive summary text."""
        method = global_explanation.get("method", "Unknown")
        top_features = global_explanation.get("summary", {}).get("top_features", [])
        
        summary = f"""
        This report provides a comprehensive analysis of the machine learning model using {method.upper()} methodology. 
        The analysis reveals the most important features driving the model's predictions and provides insights into 
        how individual features contribute to specific predictions.
        """
        
        if top_features:
            top_feature_names = [f"'{name}'" for name, _ in top_features[:3]]
            summary += f" The top three most important features are: {', '.join(top_feature_names)}."
        
        return summary.strip()
    
    def _generate_global_explanation_text(self, global_explanation: Dict[str, Any]) -> str:
        """Generate global explanation text."""
        method = global_explanation.get("method", "Unknown")
        feature_importances = global_explanation.get("feature_importances", {})
        
        text = f"""
        The global explanation using {method.upper()} methodology provides insights into the overall importance 
        of each feature in the model. This analysis helps understand which features have the most significant 
        impact on the model's predictions across the entire dataset.
        """
        
        if feature_importances:
            # Ensure all importances are float scalars
            sorted_features = sorted(
                [
                    (feature, float(np.ravel(importance)[0]))
                    for feature, importance in feature_importances.items()
                ],
                key=lambda x: x[1],
                reverse=True
            )
            text += " The feature importance analysis shows that:"
            for i, (feature, importance) in enumerate(sorted_features[:5]):
                text += f" '{feature}' has an importance score of {importance:.3f},"
            text = text.rstrip(',') + "."
        
        return text.strip()
    
    def _generate_local_explanation_text(self, local_explanation: Dict[str, Any]) -> str:
        """Generate local explanation text."""
        prediction = local_explanation.get("prediction", "N/A")
        method = local_explanation.get("method", "Unknown")
        feature_contributions = local_explanation.get("feature_contributions", {})
        
        text = f"""
        For this specific prediction ({prediction}), the {method.upper()} analysis shows how each feature 
        contributed to the final prediction. Positive contributions indicate features that pushed the 
        prediction higher, while negative contributions indicate features that pushed it lower.
        """
        
        if feature_contributions:
            # Ensure all contributions are float scalars
            sorted_contributions = sorted(
                [
                    (feature, float(np.ravel(contribution)[0]))
                    for feature, contribution in feature_contributions.items()
                ],
                key=lambda x: abs(x[1]),
                reverse=True
            )
            text += " The top contributing features are:"
            for feature, contribution in sorted_contributions[:3]:
                direction = "positively" if contribution > 0 else "negatively"
                text += f" '{feature}' contributed {direction} ({contribution:.3f}),"
            text = text.rstrip(',') + "."
        
        return text.strip()
    
    def _generate_methodology_text(self, global_explanation: Dict[str, Any]) -> str:
        """Generate methodology explanation text."""
        method = global_explanation.get("method", "Unknown")
        
        if method.lower() == "shap":
            return """
            SHAP (SHapley Additive exPlanations) values are used to explain the output of machine learning models. 
            SHAP values provide a unified measure of feature importance that is consistent across different types 
            of models. Each feature's contribution is calculated by considering all possible combinations of features 
            and measuring the change in prediction when that feature is added or removed.
            """
        elif method.lower() == "permutation":
            return """
            Permutation importance measures the decrease in model performance when a feature is randomly shuffled. 
            This method provides a model-agnostic way to measure feature importance by directly measuring the 
            impact of removing a feature's information on the model's predictive performance.
            """
        else:
            return f"""
            This analysis uses the {method.upper()} methodology to explain the model's behavior. 
            The specific details of this method depend on the implementation and the type of model being analyzed.
            """
    
    def _create_feature_importance_table(self, global_explanation: Dict[str, Any]) -> List[List[str]]:
        """Create table data for feature importances."""
        feature_importances = global_explanation.get("feature_importances", {})
        
        # Create table header
        table_data = [["Rank", "Feature", "Importance"]]
        
        # Sort features by importance
        sorted_features = sorted(
            feature_importances.items(), 
            key=lambda x: float(np.ravel(x[1])[0]) if hasattr(x[1], '__iter__') else float(x[1]), 
            reverse=True
        )[:10]  # Top 10 features
        
        # Add feature data
        for i, (feature, importance) in enumerate(sorted_features, 1):
            # Ensure importance is a scalar for formatting
            importance_scalar = float(np.ravel(importance)[0]) if hasattr(importance, '__iter__') else float(importance)
            table_data.append([str(i), feature, f"{importance_scalar:.4f}"])
        
        return table_data
    
    def _generate_html_content(
        self,
        global_explanation: Dict[str, Any],
        local_explanations: Dict[int, Dict[str, Any]],
        title: str
    ) -> str:
        """Generate complete HTML content."""
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{title}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #2c3e50;
                    text-align: center;
                    border-bottom: 3px solid #3498db;
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #34495e;
                    border-left: 4px solid #3498db;
                    padding-left: 15px;
                }}
                h3 {{
                    color: #2c3e50;
                }}
                .timestamp {{
                    text-align: center;
                    color: #7f8c8d;
                    font-style: italic;
                    margin-bottom: 30px;
                }}
                .summary {{
                    background-color: #ecf0f1;
                    padding: 20px;
                    border-radius: 5px;
                    margin: 20px 0;
                }}
                .feature-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                .feature-table th, .feature-table td {{
                    border: 1px solid #ddd;
                    padding: 12px;
                    text-align: left;
                }}
                .feature-table th {{
                    background-color: #3498db;
                    color: white;
                }}
                .feature-table tr:nth-child(even) {{
                    background-color: #f2f2f2;
                }}
                .local-explanation {{
                    background-color: #f8f9fa;
                    padding: 15px;
                    margin: 15px 0;
                    border-radius: 5px;
                    border-left: 4px solid #e74c3c;
                }}
                .methodology {{
                    background-color: #e8f5e8;
                    padding: 20px;
                    border-radius: 5px;
                    margin: 20px 0;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>{title}</h1>
                <div class="timestamp">Generated on: {timestamp}</div>
                
                <div class="summary">
                    <h2>Executive Summary</h2>
                    <p>{self._generate_executive_summary(global_explanation)}</p>
                </div>
                
                <h2>Global Model Explanation</h2>
                <p>{self._generate_global_explanation_text(global_explanation)}</p>
                
                {self._generate_html_feature_table(global_explanation)}
                
                {self._generate_html_local_explanations(local_explanations)}
                
                <div class="methodology">
                    <h2>Methodology</h2>
                    <p>{self._generate_methodology_text(global_explanation)}</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _export_markdown_report(
        self,
        global_explanation: Dict[str, Any],
        local_explanations: Dict[int, Dict[str, Any]],
        filename: str,
        title: str,
        **kwargs
    ) -> str:
        """Export report as Markdown."""
        # Ensure filename has .md extension
        if not filename.endswith('.md'):
            filename += '.md'
        
        # Generate Markdown content
        markdown_content = self._generate_markdown_content(
            global_explanation, local_explanations, title
        )
        
        # Write to file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        print(f"Markdown report saved to: {filename}")
        return filename
    
    def _generate_markdown_content(
        self,
        global_explanation: Dict[str, Any],
        local_explanations: Dict[int, Dict[str, Any]],
        title: str
    ) -> str:
        """Generate complete Markdown content."""
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        markdown = f"""# {title}

*Generated on: {timestamp}*

## Executive Summary

{self._generate_executive_summary(global_explanation)}

## Global Model Explanation

{self._generate_global_explanation_text(global_explanation)}

{self._generate_markdown_feature_table(global_explanation)}

{self._generate_markdown_local_explanations(local_explanations)}

## Methodology

{self._generate_methodology_text(global_explanation)}
"""
        
        return markdown
    
    def _generate_markdown_feature_table(self, global_explanation: Dict[str, Any]) -> str:
        """Generate Markdown table for feature importances."""
        feature_importances = global_explanation.get("feature_importances", {})
        
        if not feature_importances:
            return ""
        
        # Sort features by importance
        sorted_features = sorted(
            [
                (feature, float(np.ravel(importance)[0]))
                for feature, importance in feature_importances.items()
            ],
            key=lambda x: x[1],
            reverse=True
        )[:10]  # Top 10 features
        
        markdown = """
## Top Feature Importances

| Rank | Feature | Importance |
|------|---------|------------|
"""
        
        for i, (feature, importance) in enumerate(sorted_features, 1):
            markdown += f"| {i} | {feature} | {importance:.4f} |\n"
        
        return markdown
    
    def _generate_markdown_local_explanations(self, local_explanations: Dict[int, Dict[str, Any]]) -> str:
        """Generate Markdown for local explanations."""
        if not local_explanations:
            return ""
        
        markdown = "\n## Local Explanations\n\n"
        
        for sample_idx, local_exp in list(local_explanations.items())[:5]:  # Limit to first 5
            markdown += f"### Sample {sample_idx}\n\n"
            markdown += f"{self._generate_local_explanation_text(local_exp)}\n\n"
        
        return markdown
    
    def _generate_html_feature_table(self, global_explanation: Dict[str, Any]) -> str:
        """Generate HTML table for feature importances."""
        feature_importances = global_explanation.get("feature_importances", {})
        
        if not feature_importances:
            return ""
        
        # Sort features by importance
        sorted_features = sorted(
            [
                (feature, float(np.ravel(importance)[0]))
                for feature, importance in feature_importances.items()
            ],
            key=lambda x: x[1],
            reverse=True
        )[:10]  # Top 10 features
        
        table_html = """
        <h3>Top Feature Importances</h3>
        <table class="feature-table">
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Feature</th>
                    <th>Importance</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for i, (feature, importance) in enumerate(sorted_features, 1):
            table_html += f"""
                <tr>
                    <td>{i}</td>
                    <td>{feature}</td>
                    <td>{importance:.4f}</td>
                </tr>
            """
        
        table_html += """
            </tbody>
        </table>
        """
        
        return table_html
    
    def _generate_html_local_explanations(self, local_explanations: Dict[int, Dict[str, Any]]) -> str:
        """Generate HTML for local explanations."""
        if not local_explanations:
            return ""
        
        html = "<h2>Local Explanations</h2>"
        
        for sample_idx, local_exp in list(local_explanations.items())[:5]:  # Limit to first 5
            html += f"""
            <div class="local-explanation">
                <h3>Sample {sample_idx}</h3>
                <p>{self._generate_local_explanation_text(local_exp)}</p>
            </div>
            """
        
        return html 