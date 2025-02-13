import unittest
import json
import uuid
import re
import os
from unittest.mock import patch, MagicMock

from ontology import Ontology
from retriever import information_retriever_with_graph

class TestInformationRetrieverWithGraph(unittest.TestCase):

    def setUp(self):
        """Erstellt eine Dummy-Ontologie für den Test."""
        self.ontology = MagicMock(spec=Ontology)

        # Dummy Ontology-Struktur
        self.ontology.get_ontology_structure.return_value = {
            "Person": ["Student", "Teacher"],
            "Course": ["Math", "Physics"]
        }

        # Dummy Instanzen für Klassen
        self.ontology.get_instances_by_class.side_effect = lambda cls_list: [
            MagicMock(get_node_id=lambda: f"{cls}_instance") for cls in cls_list
        ]

        # Dummy-Knoten
        node_mock = MagicMock()
        node_mock.get_node_id.return_value = "Person_instance"
        node_mock.get_internal_structure.return_value = {"id": "Person_instance", "name": "John Doe"}
        node_mock.get_node_connections.return_value = (["Course_instance"], ["enrolled_in"])

        self.ontology.get_nodes.side_effect = lambda ids: {id_: node_mock for id_ in ids}
        self.ontology.get_node_structure.side_effect = lambda node: node.get_internal_structure()

    @patch("your_module.gpt_request", return_value='["Person"]')
    @patch("your_module.gpt_request_with_history", return_value=('["Person_instance"]', []))
    @patch("your_module.execute_query", return_value=[MagicMock(get_node_id=lambda: "Person_instance")])
    @patch("your_module.create_rag_instance_graph", return_value="static/graph/rag_test.html")
    def test_information_retriever_with_graph(self, mock_gpt_request, mock_gpt_request_with_history, mock_execute_query, mock_create_graph):
        """Testet die komplette RAG-Methode mit gemockten GPT-Responses und Ontologie."""
        user_query = "Who is John Doe?"
        retrieved_info, graph_path = information_retriever_with_graph(self.ontology, user_query)

        # Überprüfen, ob die Methode sinnvolle Daten zurückgibt
        self.assertIsInstance(retrieved_info, list)
        self.assertGreater(len(retrieved_info), 0)
        self.assertEqual(graph_path, "static/graph/rag_test.html")

        # Sicherstellen, dass GPT-Calls korrekt aufgerufen wurden
        mock_gpt_request.assert_called()
        mock_gpt_request_with_history.assert_called()

        # Sicherstellen, dass eine Graph-Datei erstellt wurde
        self.assertTrue(os.path.exists(graph_path))

    def tearDown(self):
        """Aufräumen nach Tests (löscht Test-Graph-Datei)."""
        if os.path.exists("static/graph/rag_test.html"):
            os.remove("static/graph/rag_test.html")

if __name__ == "__main__":
    unittest.main()
