import json
import unittest

import ldrag.ontology as ontology
import ldrag.retriever as retriever


class IntegrationTestRetriever(unittest.TestCase):

    def setUp(self):
        self.ontology = ontology.Ontology()
        self.ontology.deserialize("./mocks/ontology.json")


    def test_retrieve_information(self):
        retrieved_info, node_graph = retriever.information_retriever_with_graph(
            ontology=self.ontology,
            user_query="Zylinder Schraube",
            previous_conversation=[]
        )
        # print(json.dumps(retrieved_info, indent=4))

        expected = "cat__screwType_Zylinder"
        self.assertIn("screwType", json.dumps(retrieved_info), "expected ScrewType to be in retrieved info.")
        self.assertTrue(node_graph, "Node graph should not be empty.")
        self.assertIn(expected, json.dumps(retrieved_info), f"Expected '{expected}' to be in retrieved info.")

    def test_retrieve_information_with_starting_node(self):
        retrieved_info, node_graph = retriever.information_retriever_with_graph(
            ontology=self.ontology,
            user_query="Zylinder Schraube",
            previous_conversation=[],
            starting_node="screwType"
        )
        # print(json.dumps(retrieved_info, indent=4))

        expected = "cat__screwType_Schloss"
        self.assertIn("screwType", json.dumps(retrieved_info), "expected ScrewType to be in retrieved info.")
        self.assertTrue(node_graph, "Node graph should not be empty.")
        self.assertIn(expected, json.dumps(retrieved_info), f"Expected '{expected}' to be in retrieved info.")

    def test_rag_with_conversation_history(self):
        conversation_history = [
            {
                "role": "user",
                "content": "What is the screw type?"
            },
            {
                "role": "assistant",
                "content": "alx_cat__screwType_Schloss is a type of screw."
            }
        ]
        retrieved_info, node_graph = retriever.information_retriever_with_graph(
            ontology=self.ontology,
            user_query="return the mean shap value of said screw type of the decision tree model",
            previous_conversation=conversation_history
        )
        # print(json.dumps(retrieved_info, indent=4))
        self.assertTrue(node_graph, "Node graph should not be empty.")
        self.assertIn("screwType", json.dumps(retrieved_info),
                      "Expected 'screwType' in retrieved information.")
        self.assertEqual(retrieved_info[0]["Annotations"][1][1], 0.005444488822393084,
                         "Expected mean shap value of SHAP_decTree_2025_03_alx_cat__screwType_Schloss to be 0.005444488822393084.")
