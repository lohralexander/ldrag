import unittest
from ldrag.ontology import Ontology, GenericNode
from ldrag.ontology import Node

class TestOntology(unittest.TestCase):
    def setUp(self):
        self.ontology = Ontology()
        self.ontology.deserialize("./mocks/ontology.json")
        self.node2 = Node("Test2", "Model", connections=[])
        self.shap_length_node = self.ontology.get_node("SHAP_nn_2025_03_alx_num__length")


    def test_add_node(self):
        self.node = Node("Test", "Model", connections=[self.node2])
        self.ontology.add_node(self.node)
        self.assertIs(self.node, self.ontology.get_node("Test"), "Node should be added to the ontology.")

    def test_get_connected_nodes(self):
        self.assertIn(self.ontology.get_node("nn_2025_03_alx"), self.ontology.get_connected_nodes(self.shap_length_node).values(), "Connected nodes should include the connected node.")


    def test_get_node_structure(self):
        self.node_structure = self.ontology.get_node_structure(self.shap_length_node)
        self.assertIsInstance(self.node_structure, dict, "Node structure should be a dictionary.")
        self.assertNotIn("connections", self.node_structure, "Node structure should not include connections.")
        self.assertIn("Connected Instances", self.node_structure, "Node structure should 'include Connected Instances'.")
        self.assertGreater(len(self.node_structure), 0, "Node structure should not be empty.")

    def test_get_ontology_structure(self):
        ontology_structure = self.ontology.get_ontology_structure()
        self.assertIsInstance(ontology_structure, list, "Ontology structure should be a dictionary.")
        self.assertGreater(len(ontology_structure), 0, "Ontology structure should not be empty.")
        self.assertIn("Node Class ID", ontology_structure[0].keys(), "Ontology structure should include 'node class id'.")
