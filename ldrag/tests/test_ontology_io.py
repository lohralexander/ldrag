import json
import os
import unittest
import ldrag.ontology_io
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


class TestOntologyIO(unittest.TestCase):
    def setUp(self):
        # Create a small dataset
        self.X_train, self.y_train = make_classification(n_samples=100, n_features=5, random_state=42)
        self.X_test, self.y_test = make_classification(n_samples=20, n_features=5, random_state=42)

        # Convert to pandas DataFrame
        feature_names = [f"feature_{i}" for i in range(self.X_train.shape[1])]
        self.X_train = pd.DataFrame(self.X_train, columns=feature_names)
        self.X_test = pd.DataFrame(self.X_test, columns=feature_names)

        # Train a RandomForest model
        self.model = RandomForestClassifier(random_state=42)
        self.model.fit(self.X_train, self.y_train)

        # Output file for ontology
        self.output_file = "test_ontology.json"
        if os.path.exists(self.output_file):
            os.remove(self.output_file)

    def test_sklearn_model_to_ontology(self):
        # Call the function
        model_id = "test_model"
        dataset_id = "test_dataset"
        task_id = "classification_task"
        ldrag.ontology_io.sklearn_model_to_ontology(
            model=self.model,
            model_id=model_id,
            dataset_id=dataset_id,
            task_id=task_id,
            X_train=self.X_train,
            X_test=self.X_test,
            y_test=self.y_test,
            output_file=self.output_file
        )

        # Verify the output
        with open(self.output_file, "r") as f:
            data = json.load(f)

        # Check if the model node is added
        model_node = next((node for node in data["node_instances"] if node["node_id"] == model_id), None)
        print("Model node:", model_node)
        self.assertIsNotNone(model_node, "Model node should be added to the ontology.")
        self.assertEqual(model_node["node_class"], "Model", "Node class should be 'Model'.")
        self.assertIn("accuracy", model_node, "Model node should include accuracy.")

    # ToDo: Add more tests for other classifiers
    def test_get_shap_explainer(self):
        explainer = ldrag.ontology_io.get_shap_explainer(self.model, self.X_train)
        print("SHAP explainer:", explainer)
        self.assertIsNotNone(explainer, "SHAP explainer should not be None.")
        self.assertTrue(hasattr(explainer, "shap_values"), "SHAP explainer should have 'shap_values' attribute.")

    def test_add_dataset_metadata_from_dataframe(self):
        # Create a sample DataFrame
        df = pd.DataFrame({
            "feature_1": [1, 2, 3],
            "feature_2": [4, 5, 6],
            "target": [0, 1, 0]
        })
        model_node_id_list=["Preprocessing"]


        ldrag.ontology_io.add_dataset_metadata_from_dataframe("TestData",df,"TestDomain","testlocation" ,"1.1.2025",model_node_id_list,self.output_file)

        with open(self.output_file, "r") as f:
            data = json.load(f)

        # Check if the dataset metadata is added
        dataset_node = next((node for node in data["node_instances"] if node["node_class"] == "Dataset"), None)
        print("Dataset node:", dataset_node)
        self.assertIsNotNone(dataset_node, "Dataset node should be added to the ontology.")
        self.assertIn("feature_1", dataset_node["connections"][0].values(), "Dataset node should include 'feature_1'.")



    def tearDown(self):

        if os.path.exists(self.output_file):
            os.remove(self.output_file)
