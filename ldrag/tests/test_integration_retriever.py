import json
import ldrag.retriever as retriever
import unittest
import ldrag.ontology as ontology

class IntegrationTestRetriever(unittest.TestCase):

    def setUp(self):
        self.ontology = ontology.Ontology()
        self.ontology.deserialize("./mocks/ontology.json")

    #TODO: tests for retriever with parameter
    def test_retrieve_information(self):
        retrieved_info,node_graph=retriever.information_retriever_with_graph(
            ontology=self.ontology,
            user_query="Zylinder Schraube",
            previous_conversation=[]
        )
        #print(json.dumps(retrieved_info, indent=4))

        expected = "cat__screwType_Zylinder"
        self.assertIn("screwType",json.dumps(retrieved_info),"expected ScrewType to be in retrieved info.")
        self.assertTrue(node_graph , "Node graph should not be empty.")
        self.assertIn(expected, json.dumps(retrieved_info), f"Expected '{expected}' to be in retrieved info.")