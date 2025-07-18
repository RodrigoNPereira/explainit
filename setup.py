"""
Setup script for ExplainIt library.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "ExplainIt - A human-friendly explainability (XAI) library for machine learning models."

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="explainit",
    version="0.1.0",
    author="RodrigoNPereira",
    author_email="rodrigonpereira@gmail.com",
    description="A human-friendly explainability (XAI) library for machine learning models",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/RodrigoNPereira/explainit",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
            "jupyter>=1.0",
            "ipykernel>=6.0",
        ],
        "full": [
            "reportlab>=3.6.0",
            "lime>=0.2.0",
            "streamlit>=1.0.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="machine-learning explainability xai shap lime visualization reporting",
    project_urls={
        "Bug Reports": "https://github.com/RodrigoNPereira/explainit/issues",
        "Source": "https://github.com/RodrigoNPereira/explainit",
        # "Documentation": "https://explainit.readthedocs.io/",
    },
) 