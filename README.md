# ExplainIt 🧠

A human-friendly explainability (XAI) library that allows users to understand the behavior of machine learning models through **clear visualizations and exportable reports**.

Unlike SHAP, LIME, or ELI5, which expose technical internals, `ExplainIt` focuses on **simplicity and communication with non-technical stakeholders**. The target audience is data scientists, engineers, and analysts who need to explain model behavior clearly.

## ✨ Features

- **🔍 Global Explanations**: Understand overall feature importance across your entire dataset
- **🎯 Local Explanations**: Explain individual predictions with feature contributions
- **📊 Beautiful Visualizations**: Clear, publication-ready plots and charts
- **📄 Exportable Reports**: Generate PDF and HTML reports for stakeholders
- **🚀 Easy to Use**: Simple API that works out of the box
- **🔧 Model Agnostic**: Works with scikit-learn, XGBoost, and more

## 🚀 Quick Start

<!-- ### Installation

```bash
pip install explainit
```

For full functionality (PDF reports, LIME explanations):
```bash
pip install explainit[full]
``` -->

### Basic Usage

```python
from explainit import ExplainIt
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
explainer = ExplainIt(model=model, X=X_test, y=y_test, feature_names=data.feature_names)

# Generate global explanation
global_exp = explainer.explain_global()
print("Top features:", global_exp["summary"]["top_features"][:5])

# Generate local explanation for a specific sample
local_exp = explainer.explain_local(sample_index=0)
print("Prediction:", local_exp["prediction"])

# Create visualizations
fig_global = explainer.plot_global()
fig_local = explainer.plot_local(0)

# Export report
explainer.export_report("model_explanation_report.pdf")
```

## 📖 Documentation

### Core API

#### `ExplainIt(model, X, y=None, feature_names=None)`

Initialize the explainer with your trained model and data.

**Parameters:**
- `model`: Trained machine learning model (sklearn, XGBoost, etc.)
- `X`: Feature data (numpy array or pandas DataFrame)
- `y`: Target data (optional, for validation)
- `feature_names`: List of feature names (optional)

#### `explain_global(method="shap")`

Generate global explanation showing feature importances across the entire dataset.

**Parameters:**
- `method`: Explanation method ("shap", "permutation")

**Returns:** Dictionary containing global explanation data

#### `explain_local(sample_index, method="shap")`

Generate local explanation for a specific sample.

**Parameters:**
- `sample_index`: Index of the sample to explain
- `method`: Explanation method ("shap", "lime")

**Returns:** Dictionary containing local explanation data

#### `plot_global(top_n=10, **kwargs)`

Create visualization of global feature importances.

#### `plot_local(sample_index, **kwargs)`

Create visualization of local feature contributions.

#### `export_report(filename, format="pdf", **kwargs)`

Export comprehensive explanation report.

**Parameters:**
- `filename`: Output filename
- `format`: Output format ("pdf", "html")

## 🎨 Visualization Examples

### Global Feature Importance

```python
# Plot top 10 most important features
fig = explainer.plot_global(top_n=10)
fig.savefig("feature_importance.png", dpi=300, bbox_inches='tight')
```

### Local Feature Contributions

```python
# Plot feature contributions for a specific prediction
fig = explainer.plot_local(sample_index=5)
fig.savefig("local_explanation.png", dpi=300, bbox_inches='tight')
```

## 📄 Report Generation

### PDF Report

```python
# Generate comprehensive PDF report
explainer.export_report("model_analysis.pdf", format="pdf")
```

### HTML Report

```python
# Generate interactive HTML report
explainer.export_report("model_analysis.html", format="html")
```

## 🔧 Supported Models

- **scikit-learn**: All estimators with `predict` method
- **XGBoost**: XGBClassifier, XGBRegressor
- **LightGBM**: LGBMClassifier, LGBMRegressor
- **CatBoost**: CatBoostClassifier, CatBoostRegressor
- **Custom models**: Any model with `predict` method

## 🛠️ Advanced Usage

### Custom Feature Names

```python
feature_names = ["Age", "Income", "Education", "Credit Score"]
explainer = ExplainIt(model, X, y, feature_names=feature_names)
```

### Multiple Explanation Methods

```python
# SHAP explanation
shap_exp = explainer.explain_global(method="shap")

# Permutation importance
perm_exp = explainer.explain_global(method="permutation")

# LIME local explanation
lime_exp = explainer.explain_local(0, method="lime")
```

### Custom Visualizations

```python
from explainit.visualizer import Visualizer

visualizer = Visualizer(style="seaborn")
fig = visualizer.plot_global_explanation(global_exp, top_n=15)
```

## 📊 Example Notebooks

Check out the `examples/` directory for comprehensive Jupyter notebooks:

- `basic_usage.ipynb`: Getting started with ExplainIt
- `classification_example.ipynb`: Classification model explanations
- `regression_example.ipynb`: Regression model explanations
- `advanced_features.ipynb`: Advanced features and customization

## 🧪 Testing

Run the test suite:

```bash
pip install explainit[dev]
pytest tests/
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/RodrigoNPereira/explainit.git
cd explainit
pip install -e .[dev]
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [SHAP](https://github.com/slundberg/shap) for the core explanation algorithms
- [LIME](https://github.com/marcotcr/lime) for local explanations
- [scikit-learn](https://scikit-learn.org/) for the machine learning foundation

## 📞 Support

- 📧 Email: rodrigonpgmae@gmail.com
- 🐛 Issues: [GitHub Issues](https://github.com/RodrigoNPereira/explainit/issues)
<!-- - 📖 Documentation: [Read the Docs](https://explainit.readthedocs.io/) -->

---

**Made with ❤️ for the ML community** 

## 🚦 How to Use the ExplainIt Dashboard

### 1. **About the Dashboard Example**

The file `examples/dashboard_example.py` is a demonstration script. When you run it with:

```bash
python examples/dashboard_example.py
```

it will print information to the terminal and generate a file called `dashboard_script.py`. **It does NOT launch the interactive dashboard in your browser.**

---

### 2. **How to Launch the Interactive Dashboard**

To use the full interactive ExplainIt Dashboard:

1. **Locate the generated file:**
   - After running the example, you will see a file called `dashboard_script.py` in your project directory.

2. **Run the dashboard with Streamlit:**
   ```bash
   streamlit run dashboard_script.py
   ```

3. **What happens next:**
   - This command will start a local web server and open a new tab in your browser (usually at http://localhost:8501).
   - You will see the full ExplainIt Dashboard, including sidebar controls, tabs (Overview, Global Analysis, Local Analysis, Reports), and interactive visualizations.

---

### 3. **Why This Is Necessary**

- **Streamlit** is a special tool for building web apps in Python. It only renders UI elements (like buttons, tabs, and dataframes) when you run a script using the `streamlit run` command.
- If you run a script with `python script.py`, it will only execute as a normal Python script and not as a web app.

---

### 4. **Summary Table**

| Script                | How to Run                        | What You See                                 |
|-----------------------|-----------------------------------|----------------------------------------------|
| `dashboard_example.py`| `python dashboard_example.py`     | Terminal output, generates dashboard_script.py|
| `dashboard_script.py` | `streamlit run dashboard_script.py`| Full interactive dashboard in your browser   |

---

### 5. **Next Steps**

1. **Make sure you have Streamlit installed:**
   ```bash
   pip install streamlit
   ```

2. **Run the dashboard:**
   ```bash
   streamlit run dashboard_script.py
   ```

3. **Enjoy the interactive ExplainIt Dashboard in your browser!**

---

**If you want, you can also copy the contents of `dashboard_script.py` into your own file and run it with Streamlit.**

If you need help, see the examples in the `examples/` directory or ask for support! 