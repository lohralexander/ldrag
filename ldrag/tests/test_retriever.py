import unittest
from unittest.mock import patch
from ldrag import retriever, gptconnector
from ldrag.ontology import Ontology


class TestRetriever(unittest.TestCase):
    def setUp(self):
        self.ontology = Ontology()
        self.ontology.deserialize("./mocks/ontology.json")

    # TODO: Not yet working. Possibly rewrite the tested method to be less complex
    def test_retrieve_information(self):
        # Mock return value of gpt_request
        mock_gpt_request = """["Material", "TestCase", "Dataset"]
["Screw", "Material", "Mechanical Component"]
["ProcessedAttribute", "Preprocessing", "Attribute"]
["Robotarm", "Gripper", "Task"]
["GlobalInsight", "Model", "Attribute"]"""

        with patch("ldrag.retriever.gpt_request", return_value=mock_gpt_request):
            retrieved_info, graph_path = retriever.information_retriever_with_graph(
                ontology=self.ontology,
                user_query="Test",
                previous_conversation=[]
            )

            self.assertEqual(retrieved_info, ["Material", "TestCase"])
