# Ontology Framework

## ? Overview

This package provides an ontology-based framework for managing datasets, models, and explainability using SHAP values. It includes functionalities for:

- **Converting OWL to JSON** and vice versa
- **Handling ontology structures** with class relationships
- **Logging** with proper best practices
- **Machine learning model integration** using `scikit-learn`
- **SHAP explainability** for models

---

## ? Installation

```sh
pip install -r requirements.txt
```

Ensure you have all necessary dependencies, including:

- `rdflib`
- `shap`
- `sklearn`
- `pandas`
- `numpy`
- `requests`

---

## ? Configuration

This package includes a `Config` class in `config.py` that handles API keys and settings.

Example:

```python
from config import Config

gpt_client = Config.chatgpt_client  # Access OpenAI client
```

Logging is configured separately and does not interfere with package settings.

---

## ? File Structure

```
project_root/
??? ontology/                     # Core ontology functions
?   ??? ontology_io.py            # Handles OWL ? JSON conversion
?   ??? model_processing.py       # ML model integration
?   ??? shap_explainer.py         # SHAP calculations
?
??? config.py                     # Configuration and logging
??? main.py                        # Example usage
??? requirements.txt               # Dependencies
??? README.md                      # Project documentation
```

---

## ? Usage

### 1?? **Convert OWL to JSON**

```python
from ontology.ontology_io import owl_to_json

owl_to_json("baseOntology.owl", "ontology_output.json")
```

### 2?? **Train a Model & Store in Ontology**

```python
from sklearn.tree import DecisionTreeClassifier
from ontology.model_processing import sklearn_model_to_ontology

dt_model = DecisionTreeClassifier()
sklearn_model_to_ontology(dt_model, "model_1", "dataset_1", "task_1", feature_list, X_test, y_test, "ontology.json")
```

### 3?? **Calculate SHAP Values & Store in Ontology**

```python
from ontology.shap_explainer import calculate_shap_values

calculate_shap_values(dt_model, "model_1", X_train, X_test, "ontology.json")
```

---

## ? Contributing

We welcome contributions! To contribute:

1. Fork the repo
2. Create a feature branch (`git checkout -b feature-name`)
3. Commit your changes (`git commit -m "Add new feature"`)
4. Push to your branch (`git push origin feature-name`)
5. Open a pull request

---

## ? License

This project is licensed under the MIT License.

