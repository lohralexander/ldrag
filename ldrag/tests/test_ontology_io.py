import json
import os
import unittest
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import ldrag.ontology_io


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

        self.df = pd.DataFrame({
            "numerical_feature": [1.0, 2.0, 3.0],
            "categorical_feature": ["A", "B", "A"]
        })

        # Define a ColumnTransformer
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), ["numerical_feature"]),
                ("cat", OneHotEncoder(), ["categorical_feature"])
            ]
        )

        # Fit the preprocessor
        self.preprocessor.fit(self.df)

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
        self.assertIsNotNone(model_node, "Model node should be added to the ontology.")
        self.assertEqual(model_node["node_class"], "Model", "Node class should be 'Model'.")
        self.assertIn("accuracy", model_node, "Model node should include accuracy.")

    def test_get_shap_explainer(self):
        explainer = ldrag.ontology_io.get_shap_explainer(self.model, self.X_train)

        self.assertIsNotNone(explainer, "SHAP explainer should not be None.")
        self.assertTrue(isinstance(explainer, shap.TreeExplainer), "SHAP explainer should be a TreeExplainer.")
        shap_values = explainer.shap_values(self.X_test)
        self.assertEqual(len(shap_values), 2, "SHAP values should have the same length as the number of classes.")

    def test_add_dataset_metadata_from_dataframe(self):
        # Create a sample DataFrame
        df = pd.DataFrame({
            "feature_1": [1, 2, 3],
            "feature_2": [4, 5, 6],
            "target": [0, 1, 0]
        })
        model_node_id_list = ["Preprocessing"]

        ldrag.ontology_io.add_dataset_metadata_from_dataframe("TestData", df, "TestDomain", "testlocation", "1.1.2025",
                                                              model_node_id_list, self.output_file)

        with open(self.output_file, "r") as f:
            data = json.load(f)

        # Check if the dataset metadata is added
        dataset_node = next((node for node in data["node_instances"] if node["node_class"] == "Dataset"), None)
        self.assertIsNotNone(dataset_node, "Dataset node should be added to the ontology.")
        self.assertIn("feature_1", dataset_node["connections"][0].values(), "Dataset node should include 'feature_1'.")

    def test_sanitize_iri(self):

        # Test with an invalid IRI
        invalid_iri = "http://example.com/ontology#Node 1"
        sanitized_iri = ldrag.ontology_io.sanitize_iri(invalid_iri)
        self.assertEqual(sanitized_iri, "http___example_com_ontology_Node_1",
                         "Sanitized IRI should replace spaces with underscores.")

    def test_get_feature_mappings(self):
        feature_mappings = ldrag.ontology_io.get_feature_mappings(self.preprocessor, "")

        # Expected mappings
        expected_mappings = {
            "numerical_feature": ["num__numerical_feature"],
            "categorical_feature": [
                "cat__categorical_feature_A",
                "cat__categorical_feature_B"
            ]
        }

        # Assert the mappings are correct
        self.assertEqual(feature_mappings, expected_mappings, "Feature mappings should match the expected output.")

    def tearDown(self):

        if os.path.exists(self.output_file):
            os.remove(self.output_file)
