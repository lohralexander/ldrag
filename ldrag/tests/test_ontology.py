#Test Unit Case

import unittest
from ldrag.ontology import Ontology, GenericNode
from ldrag.ontology import Node

class TestOntology(unittest.TestCase):
    def setUp(self):
        self.ontology = Ontology()
        self.ontology.deserialize("../../gpt_chatbot_ldrag/ontology/ontology.json")  # Adjust the path to your ontology file
        self.node2 = Node("Test2", "Model", connections=[])


    def test_add_node(self):
        #add more nodes
        self.node = Node("Test", "Model", connections=[self.node2])
        self.ontology.add_node(self.node)
        self.assertIs(self.node, self.ontology.get_node("Test"), "Node should be added to the ontology.")

    def test_get_connected_nodes(self):
        # Test the connections of a node
        self.node = self.ontology.get_node("SHAP_nn_2025_03_alx_num__length")
        self.assertIn(self.ontology.get_node("nn_2025_03_alx"), self.ontology.get_connected_nodes(self.node).values(), "Connected nodes should include the connected node.")


    def test_get_node_structure(self):
        # Test the structure of a node
        self.node = self.ontology.get_node("SHAP_nn_2025_03_alx_num__length")
        self.node_structure = self.ontology.get_node_structure(self.node)
        self.assertIsInstance(self.node_structure, dict, "Node structure should be a dictionary.")
        self.assertNotIn("connections", self.node_structure, "Node structure should not include connections.")
        self.assertIn("Connected Instances", self.node_structure, "Node structure should 'include Connected Instances'.")
        print(self.ontology.get_ontology_structure())
