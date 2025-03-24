import json
import re

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from config import logger
from rdflib import Graph
from rdflib import Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, OWL
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


def sklearn_model_to_ontology(model, model_id, dataset_id, task_id, X_train, X_test, y_test, output_file,
                              preprocessor=None):
    """
    Converts a trained sklearn model into the ontology structure and appends it to an existing JSON file.

    :param model: Trained sklearn model
    :param model_id: Unique model identifier
    :param dataset_id: Identifier of the dataset used for training
    :param task_id: Identifier of the task the model achieves
    :param X_train: Training data features before preprocessing
    :param X_test: Test data features before preprocessing
    :param y_test: True labels for evaluation
    :param output_file: Path to the JSON file to append the ontology entry
    :param preprocessor: Optional preprocessor pipeline to process the input data
    """
    if preprocessor:
        if not hasattr(preprocessor, "transformers_"):
            preprocessor.fit(X_train)

        X_test = X_test[X_train.columns]
        X_train_transformed = preprocessor.transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)
        feature_mappings = get_feature_mappings(preprocessor, X_train)
    else:
        X_train_transformed, X_test_transformed = X_train, X_test
        feature_mappings = {f: [f] for f in X_train.columns.tolist()}

    # Model pipeline
    pipeline = Pipeline(steps=[('model', model)])

    # Predictions
    y_pred = pipeline.predict(X_test_transformed)
    y_proba = pipeline.predict_proba(X_test_transformed) if hasattr(pipeline.named_steps['model'],
                                                                    "predict_proba") else np.zeros(
        (len(X_test_transformed), pipeline.named_steps['model'].n_classes_))

    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred) if hasattr(model, "predict_proba") else None
    precision = precision_score(y_test, y_pred, average=None).tolist() if hasattr(model, "predict_proba") else None
    recall = recall_score(y_test, y_pred, average=None).tolist() if hasattr(model, "predict_proba") else None
    f1 = f1_score(y_test, y_pred, average=None).tolist() if hasattr(model, "predict_proba") else None
    roc_auc = roc_auc_score(y_test, y_proba[:, 1], multi_class='ovr') if hasattr(model, "predict_proba") else None
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()

    algorithm_name = type(model).__name__

    # Load or initialize ontology JSON
    try:
        with open(output_file, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {"node_instances": []}

    node_instances = data["node_instances"]

    # Ensure raw attributes exist in the ontology
    for raw_feature in feature_mappings.keys():
        existing_raw_node = next((node for node in node_instances if node["node_id"] == raw_feature), None)
        if not existing_raw_node:
            raw_node = {"node_id": raw_feature, "node_class": "Attribute", "connections": []}
            node_instances.append(raw_node)

    # Create processed feature nodes and link to raw features
    processed_feature_nodes = []
    model_connections = []

    for raw_feature, transformed_features in feature_mappings.items():
        for transformed_feature in transformed_features:
            existing_transformed_node = next(
                (node for node in node_instances if node["node_id"] == transformed_feature), None)
            if not existing_transformed_node:
                processed_feature_nodes.append({"node_id": transformed_feature, "node_class": "ProcessedAttribute",
                                                "connections": [{"target": raw_feature, "relation": "derivedFrom"}]})
            model_connections.append({"target": transformed_feature, "relation": "used"})

    node_instances.extend(processed_feature_nodes)

    # Check if model is a regression model
    model_weights = None
    if hasattr(model, "coef_"):  # Linear models like LinearRegression, LogisticRegression
        model_weights = {f: w for f, w in zip(X_train.columns, model.coef_.flatten().tolist())}

    elif hasattr(model, "feature_importances_"):  # Tree-based models like DecisionTree, RandomForest
        model_weights = {f: w for f, w in zip(X_train.columns, model.feature_importances_.tolist())}

    # Define model node
    model_node = {"node_id": model_id, "node_class": "Model",
                  "training_information": "Trained using sklearn in Python. A split validation was used.",
                  "algorithm": algorithm_name, "accuracy": accuracy,
                  "precision": {f"Class {i}": p for i, p in enumerate(precision)} if precision else None,
                  "recall": {f"Class {i}": r for i, r in enumerate(recall)} if recall else None,
                  "f1Score": {f"Class {i}": f for i, f in enumerate(f1)} if f1 else None,
                  "confusionMatrix": conf_matrix if hasattr(model, "predict_proba") else None,
                  "connections": [{"target": dataset_id, "relation": "trainedWith"},
                                  {"target": task_id, "relation": "achieves"}] + model_connections}

    if roc_auc is not None:
        model_node["rocAucScore"] = roc_auc

    if model_weights:  # Only include weights if it is a regression model
        model_node["weights"] = model_weights

    node_instances.append(model_node)

    # Save updated ontology
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    logger.info(f"Ontology model appended to {output_file}, weights added if applicable.")
    return model_node


def add_dataset_metadata_from_dataframe(dataset_id, df, domain, location, date, models, output_file):
    """
    Extracts dataset metadata from a pandas DataFrame and appends it to the ontology JSON file.
    Ensures all attributes exist as separate nodes before adding dataset metadata.
    Also adds statistical properties (mean, min, max, std) for numerical attributes.

    :param dataset_id: Unique identifier for the dataset
    :param df: Pandas DataFrame containing the dataset
    :param domain: Domain of the dataset
    :param location: Location where data was recorded
    :param date: Date of data recording
    :param models: List of model node_ids that used this dataset
    :param output_file: Path to the JSON file to append the dataset entry
    """
    attributes = df.columns.tolist()
    amount_of_rows = df.shape[0]
    amount_of_attributes = df.shape[1]

    # Load existing JSON file
    try:
        with open(output_file, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {"node_instances": []}

    # Check existing nodes to prevent duplicates
    existing_node_ids = {node["node_id"] for node in data["node_instances"]}

    # Ensure all attributes exist as separate nodes with statistics
    for attr in attributes:
        if attr not in existing_node_ids:
            # Determine if the attribute is numerical
            if pd.api.types.is_numeric_dtype(df[attr]):
                attr_stats = {"mean": round(float(df[attr].mean()), 6) if not df[attr].isna().all() else None,
                              "min": round(float(df[attr].min()), 6) if not df[attr].isna().all() else None,
                              "max": round(float(df[attr].max()), 6) if not df[attr].isna().all() else None,
                              "std_dev": round(float(df[attr].std()), 6) if not df[attr].isna().all() else None}
            else:
                attr_stats = {}  # No statistics for categorical attributes

            # Create attribute node
            attribute_node = {"node_id": attr, "node_class": "Attribute",
                              "connections": [{"target": dataset_id, "relation": "partOf"}], **attr_stats
                              # Merge stats into the node if available
                              }

            data["node_instances"].append(attribute_node)
            existing_node_ids.add(attr)  # Update existing node set

    # Create dataset entry
    connections = (
            [{"target": attr, "relation": "has"} for attr in attributes] + [{"target": model, "relation": "usedBy"} for
                                                                            model in models])

    dataset_entry = {"node_id": dataset_id, "amountOfRows": amount_of_rows, "amountOfAttributes": amount_of_attributes,
                     "node_class": "Dataset", "domain": domain, "locationOfDataRecording": location,
                     "dateOfRecording": date, "connections": connections}

    # Append dataset entry
    data["node_instances"].append(dataset_entry)

    # Save back to JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    logger.info(f"Dataset metadata appended to {output_file}, and new attributes added if missing.")

    return dataset_entry


def get_shap_explainer(model, X_train_transformed):
    """
    Selects the correct SHAP explainer based on the model type.
    """

    if isinstance(model, (
            DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier, xgb.XGBClassifier,
            lgb.LGBMClassifier,
            cb.CatBoostClassifier)):
        return shap.TreeExplainer(model)
    elif isinstance(model, LogisticRegression):
        return shap.LinearExplainer(model, X_train_transformed)
    elif isinstance(model, MLPClassifier):
        # Use KernelExplainer for scikit-learn MLPClassifier.
        # Reduce background samples using k-means (or shap.sample) to speed up calculations.
        K = 100  # or any other number you prefer
        background_data = shap.kmeans(X_train_transformed, K)  # alternatively: shap.sample(X_train_transformed, K)
        return shap.KernelExplainer(model.predict, background_data)
    else:
        return shap.Explainer(model, X_train_transformed, check_additivity=False)


def calculate_shap_values(model, model_id, X_train, X_test, output_file, preprocessor=None):
    """
    Calculates SHAP values for a given model and updates the ontology JSON.

    :param model: Trained sklearn model
    :param model_id: Unique model identifier
    :param X_train: Training data features (before preprocessing)
    :param X_test: Test data features (before preprocessing)
    :param output_file: Path to the JSON file where ontology is stored
    :param preprocessor: Optional preprocessor pipeline used before model training
    """
    # Apply preprocessing if available
    if preprocessor:
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
        X_train_transformed = pipeline.named_steps['preprocessor'].transform(X_train)
        X_test_transformed = pipeline.named_steps['preprocessor'].transform(X_test)
        feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
    else:
        X_train_transformed = X_train
        X_test_transformed = X_test
        feature_names = X_train.columns.tolist()

    # Ensure the data shapes match
    if X_train_transformed.shape[1] != X_test_transformed.shape[1]:
        raise ValueError(
            f"Feature mismatch! X_train has {X_train_transformed.shape[1]} columns, but X_test has {X_test_transformed.shape[1]}.")

    # Select the appropriate SHAP explainer based on model type
    explainer = get_shap_explainer(model, X_train_transformed)

    # Compute SHAP values with check_additivity disabled
    shap_values = explainer.shap_values(X_test_transformed)

    # If shap_values is a list or tuple (as in many classifiers), choose one class's explanation.
    if isinstance(shap_values, (list, tuple)):
        # If only one array is returned, use it.
        if len(shap_values) == 1:
            shap_array = shap_values[0]
        else:
            # For binary classifiers, use the second element (class 1)
            shap_array = shap_values[1]
    else:
        shap_array = shap_values

    # Create SHAP nodes for the ontology
    shap_nodes = []
    for i, feature in enumerate(feature_names):
        # Use the selected shap_array for slicing
        mean_shap_value = np.mean(np.abs(shap_array[:, i])).item()

        # Create a SHAP node for this feature
        shap_node = {"node_id": f"SHAP_{model_id}_{feature}", "node_class": "SHAPValue", "feature": feature,
                     "mean_shap_value": mean_shap_value, "connections": [{"target": model_id, "relation": "relatedTo"},
                                                                         {"target": feature, "relation": "explains"}]}
        shap_nodes.append(shap_node)

    # Load existing ontology JSON
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print("Error: Ontology JSON file not found or invalid!")
        return

    node_instances = data.get("node_instances", [])
    node_instances.extend(shap_nodes)

    data["node_instances"] = node_instances
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    logger.info(f"SHAP values added as separate nodes and ontology updated in {output_file}")


def sanitize_iri(value):
    """
    Sanitizes an IRI by replacing invalid characters with underscores.

    :param value: The string to sanitize
    :return: A valid IRI-friendly string
    """
    return re.sub(r'[^a-zA-Z0-9_]', '_', value)


def get_feature_mappings(preprocessor, X_train):
    """
    Extracts mappings from original dataset attributes to preprocessed feature names.

    :param preprocessor: A fitted sklearn ColumnTransformer used for preprocessing
    :param X_train: The original training dataset (before preprocessing)
    :return: A dictionary linking raw feature names to transformed feature names
    """
    # Assuming pipeline is already fitted
    feature_names = preprocessor.get_feature_names_out()

    # Now create a mapping
    feature_mapping = {}
    for feature_name in feature_names:
        # For one-hot encoded features
        if '__' in feature_name:
            parts = feature_name.split('__')
            transformer = parts[0]
            feature_info = parts[1]

            # For categorical features with format 'cat__column_value'
            if transformer == 'cat':
                # Attempt to split by the last underscore to separate the category value
                feature_parts = feature_info.rsplit('_', 1)
                if len(feature_parts) == 2:
                    original_col, category = feature_parts
                    if original_col not in feature_mapping:
                        feature_mapping[original_col] = []
                    feature_mapping[original_col].append(feature_name)
            else:
                # For numeric features with format 'num__column'
                original_col = feature_info
                feature_mapping[original_col] = [feature_name]

    return feature_mapping


def convert_json_to_owl(json_file, owl_file_name):
    """
    Converts ontology JSON to an OWL file.

    :param json_file: The JSON file to be converted to OWL Format.
    :param owl_file_name: The name the converted File shall have.
    """
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    g = Graph()
    BASE_URI = "http://example.org/ontology#"
    EX = Namespace(BASE_URI)
    g.bind("ex", EX)

    class_dict = {}
    for node_class in data["node_classes"]:
        class_uri = URIRef(EX[sanitize_iri(node_class["node_class_id"])])
        g.add((class_uri, RDF.type, OWL.Class))
        g.add((class_uri, RDFS.label, Literal(node_class["node_class_id"])))
        class_dict[node_class["node_class_id"]] = class_uri

    for instance in data["node_instances"]:
        instance_uri = URIRef(EX[sanitize_iri(instance["node_id"])])
        class_uri = class_dict.get(instance["node_class"], None)

        if class_uri:
            g.add((instance_uri, RDF.type, class_uri))

        for connection in instance["connections"]:
            target_uri = URIRef(EX[sanitize_iri(connection["target"])])
            relation_uri = URIRef(EX[sanitize_iri(connection["relation"])])
            g.add((instance_uri, relation_uri, target_uri))

    g.serialize(destination=owl_file_name, format="xml")
    logger.info(f"OWL file saved as {owl_file_name}")


def owl_to_json(owl_file, json_file):
    """
    Converts an OWL ontology file to the proprietary JSON format.

    :param owl_file: Path to the OWL file
    :param json_file: Path to save the output JSON file
    """


# Prototype TODO Improve
def upload_ontology(json_file, owl_file):
    # GraphDB Configuration
    GRAPHDB_URL = "http://localhost:7200"  # Change if needed
    REPOSITORY = "partnermeeting"  # Replace with your repo name
    SPARQL_UPDATE_URL = f"{GRAPHDB_URL}/repositories/{REPOSITORY}/statements"

    # OWL File Path
    OWL_FILE_PATH = "output.owl"

    # Load OWL file into rdflib Graph
    g = Graph()
    g.parse(owl_file, format="xml")  # Adjust format if needed

    # Extract Classes
    node_classes = []
    for cls in g.subjects(RDF.type, OWL.Class):
        class_id = sanitize_iri(cls.split("#")[-1])  # Get the local name
        explanation = g.value(cls, RDFS.comment)
        node_classes.append({"node_class_id": class_id, "class_connections": [],
                             "explanation": str(explanation) if explanation else ""})

    # Extract Instances
    node_instances = []
    for instance in g.subjects(RDF.type):  # Get all entities with a type
        instance_id = sanitize_iri(instance.split("#")[-1])
        instance_class = g.value(instance, RDF.type)  # Get its class
        class_id = sanitize_iri(instance_class.split("#")[-1]) if instance_class else "Unknown"

        # Extract connections
        connections = []
        for pred, obj in g.predicate_objects(subject=instance):
            if pred not in {RDF.type, RDFS.label, RDFS.comment}:  # Ignore metadata
                target_id = sanitize_iri(obj.split("#")[-1])
                relation_id = sanitize_iri(pred.split("#")[-1])
                connections.append({"target": target_id, "relation": relation_id})

        # Store instance data
        node_instance = {"node_id": instance_id, "node_class": class_id, "connections": connections}
        node_instances.append(node_instance)

    # Final JSON structure
    ontology_data = {"node_classes": node_classes, "node_instances": node_instances}

    # Save to JSON
    with open(json_file, "w", encoding="utf-8") as file:
        json.dump(ontology_data, file, indent=4, ensure_ascii=False)

    print(f"? OWL file converted to JSON and saved as {json_file}")
