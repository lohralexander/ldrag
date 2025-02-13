import os
import unittest
import json

from ldrag.ontology import Ontology, Node, GenericNode, GenericClass


class TestOntology(unittest.TestCase):
    """
    Unit tests for the Ontology class and its associated classes.
    """

    def setUp(self):
        """
        Sets up the test environment before each test.
        """
        self.ontology = Ontology()
        self.node_class = GenericClass("TestClass", ["RelatedClass"], "A test class")
        self.node = GenericNode("Node1", self.node_class, [{"target": "Node2", "relation": "connectedTo"}],
                                attribute1="value1")
        self.ontology.add_node_class(self.node_class)
        self.ontology.add_node(self.node)

    def test_add_node(self):
        """
        Tests adding a node to the ontology.
        """
        self.assertIn("Node1", self.ontology._node_dict)

    def test_add_node_class(self):
        """
        Tests adding a node class to the ontology.
        """
        self.assertIn("TestClass", self.ontology._node_class_dict)

    def test_get_node(self):
        """
        Tests retrieving a node by its ID.
        """
        retrieved_node = self.ontology.get_node("Node1")
        self.assertEqual(retrieved_node, self.node)

    def test_check_if_node_exists(self):
        """
        Tests checking if a node exists.
        """
        self.assertTrue(self.ontology.check_if_node_exists("Node1"))
        self.assertFalse(self.ontology.check_if_node_exists("NonExistentNode"))

    def test_check_if_class_exists(self):
        """
        Tests checking if a node class exists.
        """
        self.assertTrue(self.ontology.check_if_class_exists("TestClass"))
        self.assertFalse(self.ontology.check_if_class_exists("NonExistentClass"))

    def test_get_instances_by_class(self):
        """
        Tests retrieving instances by class ID.
        """
        instances = self.ontology.get_instances_by_class("TestClass")
        self.assertIn(self.node, instances)
        self.assertEqual(len(instances), 1)

    def test_node_class(self):
        """
        Tests the Node class functions.
        """
        node = Node("Node2", "TestClass", [{"target": "Node1", "relation": "linked"}])
        self.assertEqual(node.get_node_id(), "Node2")
        self.assertEqual(node.get_node_class_id(), "TestClass")
        self.assertEqual(len(node.get_node_connections()), 1)

    def test_generic_node(self):
        """
        Tests the GenericNode class functions.
        """
        self.assertEqual(self.node.get_node_id(), "Node1")
        self.assertEqual(self.node.get_node_class_id(), "TestClass")
        self.assertEqual(len(self.node.get_class_connections()), 1)
        self.assertEqual(self.node.get_explanation(), "A test class")
        self.assertEqual(self.node.attribute1, "value1")

    def test_generic_class(self):
        """
        Tests the GenericClass class functions.
        """
        self.assertEqual(self.node_class.get_node_class_id(), "TestClass")
        self.assertEqual(len(self.node_class.get_class_connections()), 1)
        self.assertEqual(self.node_class.get_explanation(), "A test class")

    def test_serialization(self):
        """
        Tests serialization of the ontology to JSON.
        """
        ontology_json = json.dumps({
            "node_classes": [{
                "node_class_id": self.node_class.get_node_class_id(),
                "class_connections": self.node_class.get_class_connections(),
                "explanation": self.node_class.get_explanation()
            }],
            "node_instances": [{
                "node_id": self.node.get_node_id(),
                "node_class": self.node.get_node_class_id(),
                "connections": self.node.get_node_connections(),
                "attribute1": self.node.attribute1
            }]
        })
        self.assertTrue("TestClass" in ontology_json)
        self.assertTrue("Node1" in ontology_json)

    def test_deserialization(self):
        """
        Tests deserialization of the ontology from JSON.
        """
        test_json = {
            "node_classes": [{
                "node_class_id": "TestClass",
                "class_connections": ["RelatedClass"],
                "explanation": "A test class"
            }],
            "node_instances": [{
                "node_id": "Node1",
                "node_class": "TestClass",
                "connections": [{"target": "Node2", "relation": "connectedTo"}],
                "attribute1": "value1"
            }]
        }
        ontology = Ontology()
        with open("test_ontology.json", "w") as f:
            json.dump(test_json, f)
        ontology.deserialize("test_ontology.json")
        self.assertTrue(ontology.check_if_node_exists("Node1"))
        self.assertTrue(ontology.check_if_class_exists("TestClass"))
        os.remove("test_ontology.json")


if __name__ == "__main__":
    unittest.main()
