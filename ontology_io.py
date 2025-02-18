import json
import re

import numpy as np
import pandas as pd
from rdflib import Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, OWL
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def sklearn_model_to_ontology(model, model_id, dataset_id, task_id, attributes, X_test, y_test, output_file):
    """
    Converts a trained sklearn model into the ontology structure and appends it to an existing JSON file.

    :param model: Trained sklearn model
    :param model_id: Unique model identifier
    :param dataset_id: Identifier of the dataset used for training
    :param task_id: Identifier of the task the model achieves
    :param attributes: List of attribute node_ids used in training
    :param X_test: Test data features
    :param y_test: True labels for evaluation
    :param output_file: Path to the JSON file to append the ontology entry
    """
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else np.zeros(
        (len(y_pred), model.n_classes_)) if hasattr(model, "predict_proba") else np.zeros_like(y_pred)

    # Evaluation Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average=None).tolist()
    recall = recall_score(y_test, y_pred, average=None).tolist()
    f1 = f1_score(y_test, y_pred, average=None).tolist()
    roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr') if hasattr(model, "predict_proba") else None
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()

    # Algorithm Name
    algorithm_name = type(model).__name__

    # Ontology Structure
    ontology_entry = {"node_id": model_id, "node_class": "Model",
        "training_information": "Trained using sklearn in Python. A split validation was used.",
        "connections": [[dataset_id, task_id] + attributes, ["trainedWith", "achieves"] + ["used"] * len(attributes)],
        "algorithm": algorithm_name, "accuracy": accuracy,
        "precision": {f"Class {i}": p for i, p in enumerate(precision)},
        "recall": {f"Class {i}": r for i, r in enumerate(recall)},
        "f1Score": {f"Class {i}": f for i, f in enumerate(f1)}, "confusionMatrix": conf_matrix, }

    if roc_auc is not None:
        ontology_entry["rocAucScore"] = roc_auc

    # Load existing JSON file
    try:
        with open(output_file, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {"node_instances": []}

    # Append new model entry
    data["node_instances"].append(ontology_entry)

    # Save back to JSON
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Ontology model appended to {output_file}")

    return ontology_entry


def sklearn_model_to_ontology(model, model_id, dataset_id, task_id, attributes, X_test, y_test, output_file):
    """
    Converts a trained sklearn model into the ontology structure and appends it to an existing JSON file.

    :param model: Trained sklearn model
    :param model_id: Unique model identifier
    :param dataset_id: Identifier of the dataset used for training
    :param task_id: Identifier of the task the model achieves
    :param attributes: List of attribute node_ids used in training
    :param X_test: Test data features
    :param y_test: True labels for evaluation
    :param output_file: Path to the JSON file to append the ontology entry
    """
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else np.zeros(
        (len(y_pred), model.n_classes_))

    # Evaluation Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average=None).tolist()
    recall = recall_score(y_test, y_pred, average=None).tolist()
    f1 = f1_score(y_test, y_pred, average=None).tolist()
    roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr') if hasattr(model, "predict_proba") else None
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()

    # Algorithm Name
    algorithm_name = type(model).__name__

    # Ontology Structure
    ontology_entry = {"node_id": model_id, "node_class": "Model",
        "training_information": "Trained using sklearn in Python. A split validation was used.",
        "connections": [[dataset_id, task_id] + attributes, ["trainedWith", "achieves"] + ["used"] * len(attributes)],
        "algorithm": algorithm_name, "accuracy": accuracy,
        "precision": {f"Class {i}": p for i, p in enumerate(precision)},
        "recall": {f"Class {i}": r for i, r in enumerate(recall)},
        "f1Score": {f"Class {i}": f for i, f in enumerate(f1)}, "confusionMatrix": conf_matrix, }

    if roc_auc is not None:
        ontology_entry["rocAucScore"] = roc_auc

    # Load existing JSON file
    try:
        with open(output_file, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {"node_instances": []}

    # Append new model entry
    data["node_instances"].append(ontology_entry)

    # Save back to JSON
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Ontology model appended to {output_file}")

    return ontology_entry


def add_dataset_metadata_from_dataframe(dataset_id, df, domain, location, date, models, output_file):
    """
    Extracts dataset metadata from a pandas DataFrame and appends it to the ontology JSON file.

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

    dataset_entry = {"node_id": dataset_id, "amountOfRows": amount_of_rows, "amountOfAttributes": amount_of_attributes,
        "node_class": "Dataset", "domain": domain, "locationOfDataRecording": location, "dateOfRecording": date,
        "connections": [attributes + models, ["has"] * len(attributes) + ["usedBy"] * len(models)]}

    # Load existing JSON file
    try:
        with open(output_file, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {"node_instances": []}

    # Append new dataset entry
    data["node_instances"].append(dataset_entry)

    # Save back to JSON
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Dataset metadata appended to {output_file}")

    return dataset_entry


def sanitize_iri(value):
    """Sanitize IRI by replacing invalid characters with underscores."""
    return re.sub(r'[^a-zA-Z0-9_]', '_', value)


def convert_json_to_owl(json_file, owl_file):
    # Load JSON file
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Define OWL Graph
    g = Graph()

    # Define Namespace (ensure no trailing special characters)
    BASE_URI = "http://example.org/ontology#"
    EX = Namespace(BASE_URI)
    g.bind("ex", EX)

    # Create Classes
    class_dict = {}
    for node_class in data["node_classes"]:
        sanitized_class = sanitize_iri(node_class["node_class_id"])
        class_uri = URIRef(EX[sanitized_class])
        g.add((class_uri, RDF.type, OWL.Class))
        g.add((class_uri, RDFS.label, Literal(node_class["node_class_id"])))
        class_dict[node_class["node_class_id"]] = class_uri

    # Create Instances & Relationships
    for instance in data["node_instances"]:
        sanitized_instance = sanitize_iri(instance["node_id"])
        instance_uri = URIRef(EX[sanitized_instance])
        class_uri = class_dict.get(instance["node_class"], None)

        if class_uri:
            g.add((instance_uri, RDF.type, class_uri))

        # Add connections as properties
        for targets, predicates in zip(instance["connections"][0], instance["connections"][1]):
            sanitized_target = sanitize_iri(targets)
            sanitized_predicate = sanitize_iri(predicates)

            target_uri = URIRef(EX[sanitized_target])
            property_uri = URIRef(EX[sanitized_predicate])
            g.add((instance_uri, property_uri, target_uri))

    # Save OWL file
    g.serialize(destination=owl_file, format="xml")
    print(f"OWL file saved as {owl_file}")


import requests
from rdflib import Graph


def upload_ontology():
    # GraphDB Configuration
    GRAPHDB_URL = "http://localhost:7200"  # Change if needed
    REPOSITORY = "partnermeeting"  # Replace with your repo name
    SPARQL_UPDATE_URL = f"{GRAPHDB_URL}/repositories/{REPOSITORY}/statements"

    # OWL File Path
    OWL_FILE_PATH = "output.owl"

    # Load OWL file into rdflib Graph
    g = Graph()
    g.parse(OWL_FILE_PATH, format="xml")  # Ensure your file format is RDF/XML

    # Convert OWL data to N-Triples format (compatible with SPARQL)
    rdf_data = g.serialize(format="nt")

    # SPARQL INSERT Query
    SPARQL_UPDATE = f"""
    INSERT DATA {{
        {rdf_data}
    }}
    """

    # Send the update request
    response = requests.post(SPARQL_UPDATE_URL, data={"update": SPARQL_UPDATE},
        headers={"Content-Type": "application/x-www-form-urlencoded"})

    # Check response
    if response.status_code == 204:
        print("? OWL Data successfully imported into GraphDB!")
    else:
        print(f"? Import failed! Status Code: {response.status_code}\nResponse: {response.text}")


if __name__ == '__main__':
    # Load dataset
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    X_train, X_test, y_train, y_test = train_test_split(df, data.target, test_size=0.2, random_state=42)

    # Train model
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # Define metadata for ontology
    model_id = "model_iris_pm"
    dataset_id = "iris_dataset"
    task_id = "classification_task"
    output_file = "./research/ontology/ontologyTest.json"

    # Add dataset to ontology
    add_dataset_metadata_from_dataframe(dataset_id, df, "Iris Dataset", "Unknown Location", "2024", [model_id],
                                        output_file)

    # Add model to ontology
    sklearn_model_to_ontology(model, model_id, dataset_id, task_id, df.columns.tolist(), X_test, y_test, output_file)

    # Example Usage
    convert_json_to_owl("./research/ontology/ontologyTest.json", "output.owl")

    # Run the Upload
    upload_ontology()
