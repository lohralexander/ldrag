import ldrag.retriever as retriever
import unittest
import ldrag.ontology as ontology



class IntegrationTestRetriever(unittest.TestCase):

    def setUp(self):
        self.ontology = ontology.Ontology()
        self.ontology.deserialize("./mocks/ontology.json")

#TODO: implement
    def test_retrieve_information(self):
        retrieved_info,node_graph=retriever.information_retriever_with_graph(
            ontology=self.ontology,
            user_query="Test",
            previous_conversation=[]
        )
        expected_output = 'SHAP_nn_2025_03_alx_num__length'
        print("Retrieved information:", retrieved_info)
        self.assertEqual(retrieved_info[0].get('Node Instance ID'), expected_output)