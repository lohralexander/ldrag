import unittest
from unittest.mock import patch

from ldrag import retriever
from ldrag.ontology import Ontology


class TestRetriever(unittest.TestCase):
    def setUp(self):
        self.ontology = Ontology()
        self.ontology.deserialize("./mocks/ontology.json")

    def test_retrieve_information(self):
        # Mock return value of gpt_request
        mock_gpt_request = """["Material", "TestCase", "Dataset"]"""
        gpt_history_response = ["SHAP_nn_2025_03_alx_num__length", "none"]

        with patch("ldrag.retriever.gpt_request", return_value=mock_gpt_request), \
                patch("ldrag.retriever.gpt_request_with_history", return_value=gpt_history_response):
            retrieved_info, graph_path = retriever.information_retriever_with_graph(
                ontology=self.ontology,
                user_query="Test",
                previous_conversation=[]
            )
            expected_output = 'SHAP_nn_2025_03_alx_num__length'

            self.assertEqual(retrieved_info[0].get('Node Instance ID'), expected_output,
                             "The retrieved node should match the expected node.")

            # test with start node
            retrieved_info, graph_path = retriever.information_retriever_with_graph(
                ontology=self.ontology,
                user_query="Test",
                previous_conversation=[],
                starting_node="SHAP_nn_2025_03_alx_num__length"
            )

            self.assertEqual(retrieved_info[0].get('Node Instance ID'), expected_output,
                             "The retrieved node should match the expected node when a starting node is provided.")

    def test_execute_query(self):
        expected_nodes = [
            "SHAP_ranFor_2025_03_alx_num__weight",
            "SHAP_ranFor_2025_03_alx_cat__screwType_Sechskant",
        ]

        retrieved_info = retriever.execute_query(
            query="[SHAP_ranFor_2025_03_alx_num__weight,SHAP_ranFor_2025_03_alx_cat__screwType_Sechskant]",
            ontology=self.ontology,
        )
        node_ids = [node.node_id for node in retrieved_info]
        self.assertEqual(node_ids, expected_nodes, "The retrieved nodes should match the expected nodes.")
