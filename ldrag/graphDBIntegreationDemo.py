import os
import json
import logging
import uuid
from neo4j import GraphDatabase
from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from ldrag.retriever import information_retriever_with_graph

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class GraphDBOntology:
    """
    Minimal ontology‐like interface for a GraphDB (Neo4j) backend.
    This class mimics required functions from the original Ontology and
    builds an in‑memory dictionary of nodes.
    """
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._node_dict = {}

    def close(self):
        self.driver.close()

    def get_instances_by_class(self, node_class_id_list):
        """
        Retrieves all instances belonging to specific node classes.

        :param node_class_id_list: List of node class IDs.
        :return: List of node instances.
        """
        return [node for node in self._node_dict.values() if node.get_node_class_id() in node_class_id_list]

    def load_nodes(self):
        """
        Loads a subset of nodes and their relationships from Neo4j and stores them
        in a dictionary with the same keys used in the original Ontology.
        """
        query = """
        MATCH (n)-[r]->(m)
        RETURN n, r, m
        LIMIT 50
        """
        with self.driver.session() as session:
            result = session.run(query)
            for record in result:
                node_n = record["n"]
                # Create a minimal node structure that contains:
                # - node_id: use the Neo4j id converted to string
                # - node_class: use one of n's labels
                # - connections: list of dictionaries with target and relation
                node_id = str(node_n.id)
                labels = list(node_n.labels)
                properties = dict(node_n)
                # Define node_class as the first label if available
                node_class = labels[0] if labels else "UnknownClass"
                # Build a simple connection from every outgoing relationship.
                connection = {
                    "target": str(record["m"].id),
                    "relation": record["r"].type
                }
                # Build node structure if not exists
                if node_id not in self._node_dict:
                    self._node_dict[node_id] = SimpleNode(
                        node_id=node_id,
                        node_class=node_class,
                        connections=[connection],
                        properties=properties
                    )
                else:
                    self._node_dict[node_id].connections.append(connection)
                # Also add target node if not present.
                target_id = str(record["m"].id)
                if target_id not in self._node_dict:
                    target_labels = list(record["m"].labels)
                    target_class = target_labels[0] if target_labels else "UnknownClass"
                    self._node_dict[target_id] = SimpleNode(
                        node_id=target_id,
                        node_class=target_class,
                        connections=[],
                        properties=dict(record["m"])
                    )
        logger.info("Loaded nodes from GraphDB.")

    def get_ontology_structure(self):
        """
        Returns a JSON structure summarizing the node classes from the GraphDB.
        """
        classes = {}
        for node in self._node_dict.values():
            classes.setdefault(node.node_class, {"count": 0, "examples": []})
            classes[node.node_class]["count"] += 1
            if len(classes[node.node_class]["examples"]) < 3:
                classes[node.node_class]["examples"].append(node.node_id)
        structure = [{"Node Class": cls, "Details": details} for cls, details in classes.items()]
        return structure

    def get_nodes(self, node_id_list):
        """
        Retrieves nodes for a given list of node IDs.
        """
        ret = {}
        for node_id in node_id_list:
            if node_id in self._node_dict:
                ret[node_id] = self._node_dict[node_id]
        return ret

    def get_node_structure(self, node):
        """
        Retrieves the structure of a node.
        """
        node_structure = {
            "Node Instance ID": node.get_node_id(),
            "Explanation": node.get_explanation(),
            "Connected Instances": ", ".join(
                f"{conn['relation']} {conn['target']}"
                for conn in node.get_node_connections()
                if isinstance(conn, dict) and "target" in conn and "relation" in conn
            )
        }
        # Exclude certain annotations if needed
        ignored_keys = {}
        annotations = [(key, value) for key, value in node.__dict__.items() if key not in ignored_keys]
        node_structure["Annotations"] = annotations
        return node_structure

class SimpleNode:
    """
    Minimal node class to mimic the original Ontology node.
    """
    def __init__(self, node_id, node_class, connections, properties):
        self.node_id = node_id
        self.node_class = node_class
        self.connections = connections
        self.properties = properties

    def get_node_id(self):
        return self.node_id

    def get_node_class_id(self):
        return self.node_class

    def get_node_connections(self):
        return self.connections

    def get_explanation(self):
        # Use a property as explanation if exists
        return self.properties.get("name", f"Node_{self.node_id}")

    def get_internal_structure(self):
        return list(self.__dict__.keys())
def retrieve_relevant_nodes(uri, user, password, user_query):
    """
    Stellt eine Verbindung zu Neo4j her und sucht nach Knoten,
    deren Eigenschaft 'name' den Suchbegriff enthält.
    """
    driver = GraphDatabase.driver(uri, auth=(user, password))
    cypher_query = """
   CALL db.index.fulltext.queryNodes('nameIndex', $user_query)
    YIELD node, score
    RETURN node, score
    LIMIT 10
    """
    retrieved_nodes = []
    with driver.session() as session:
        result = session.run(cypher_query, user_query=user_query)
        for record in result:
            node = record["node"]
            score = record["score"]
            node_id = node.id
            labels = list(node.labels)
            properties = dict(node)
            retrieved_nodes.append({
                "id": str(node_id),
                "labels": labels,
                "properties": properties,
                "score": score
            })
    driver.close()
    logger.info("Knoten wurden aus der Datenbank abgerufen.")
    return retrieved_nodes
def main():
    # Connection details from environment variables or hard-coded for demo.
    neo4j_uri = os.environ.get("NEO4J_URI","neo4j+s://ae66b6dc.databases.neo4j.io")
    neo4j_user = os.environ.get("NEO4J_USER","neo4j")
    neo4j_password = os.environ.get("NEO4J_PASSWORD","kcB3a0jyR0GYFy6KUHWiOb5HJf4qtkp6JYR4IrQUdqA")
    update_embeddings_to_3072(neo4j_uri, neo4j_user, neo4j_password)
    return
    run_rag_demo(neo4j_uri, neo4j_user, neo4j_password)

    # Create our GraphDB ontology interface
    ontology = GraphDBOntology(neo4j_uri, neo4j_user, neo4j_password)
    ontology.load_nodes()

    # For demo, write out the graph visualization used by RAG.
    # In practice, the user_query would come from an end user.
    user_query = "Which model has the highest ROC AUC"
    #retrieved_info, graph_path = information_retriever_with_graph(
    #    ontology=ontology,
    #    user_query=user_query,
    #    sleep_time=0
    #)

    print("Retrieved information:")
    print(json.dumps(retrieve_relevant_nodes(neo4j_uri,neo4j_user,neo4j_password,  user_query), indent=2))
    #print(f"Graph visualization saved at: {graph_path}")

    ontology.close()


def run_rag_demo(uri, user, password, index_name="nodeEmbedding"):
    from neo4j import GraphDatabase
    from neo4j_graphrag.retrievers import VectorRetriever
    from neo4j_graphrag.llm import OpenAILLM
    from neo4j_graphrag.generation import GraphRAG
    from neo4j_graphrag.embeddings import OpenAIEmbeddings

    # Connect to Neo4j database
    driver = GraphDatabase.driver(uri, auth=(user, password))

    # Create Embedder object
    embedder = OpenAIEmbeddings(model="text-embedding-3-small")

    # Initialize the retriever
    retriever = VectorRetriever(driver, index_name, embedder)

    # LLM
    llm = OpenAILLM(model_name="gpt-4o", model_params={"temperature": 0})

    # Initialize the RAG pipeline
    rag = GraphRAG(retriever=retriever, llm=llm)

    # Query the graph
    query_text = "How do I do similarity search in Neo4j?"
    response = rag.search(query_text=query_text, retriever_config={"top_k": 5})
    print(response.answer)




def update_embeddings_to_3072(uri, user, password, text_property_candidates=None, label="VectorNode", embedding_property="embedding3072"):
    """
    Aktualisiert alle Knoten mit dem angegebenen Label und erzeugt ein 3072-dimensionales Embedding
    für ein passendes Textfeld. Das neue Embedding wird als embedding3072 gespeichert.
    """
    from neo4j import GraphDatabase
    from neo4j_graphrag.embeddings import OpenAIEmbeddings
    driver = GraphDatabase.driver(uri, auth=(user, password))
    embedder = OpenAIEmbeddings(model="text-embedding-3-small")
    if text_property_candidates is None:
        text_property_candidates = ["usecase", "feature", "algorithm", "node_id"]
    with driver.session() as session:
        # Alle relevanten Knoten abfragen
        cypher = f"""
        MATCH (n:{label})
        RETURN id(n) as id, n
        """
        result = session.run(cypher)
        for record in result:
            node = record["n"]
            node_id = record["id"]
            # Passendes Textfeld suchen
            text = None
            for prop in text_property_candidates:
                if prop in node and node[prop]:
                    text = str(node[prop])
                    break
            if not text:
                print(f"Kein"
                      f""
                      f""
                      f" Textfeld für Knoten {node_id} gefunden, überspringe.")
                continue
            # Embedding erzeugen
            embedding = embedder.embed_query(text)
            # Embedding speichern
            session.run(
                f"MATCH (n) WHERE node_id(n) = $id SET n.{embedding_property} = $embedding",
                id=node_id, embedding=embedding
            )
            print(f"Knoten {node_id}: embedding3072 aktualisiert.")
    driver.close()

def create_vector_index_in_neo4j(uri, user, password, index_name, label, embedding_property, dimensions, similarity_fn="cosine"):
    """
    Erstellt einen Vektor-Index in Neo4j mit den angegebenen Parametern.
    """
    from neo4j import GraphDatabase
    from neo4j_graphrag.indexes import create_vector_index
    driver = GraphDatabase.driver(uri, auth=(user, password))
    create_vector_index(
        driver,
        index_name,
        label=label,
        embedding_property=embedding_property,
        dimensions=dimensions,
        similarity_fn=similarity_fn,
    )
    driver.close()
    print(f"Vektor-Index '{index_name}' für Label '{label}' und Property '{embedding_property}' mit {dimensions} Dimensionen wurde erstellt.")


def upsert_vector_to_neo4j(uri, user, password, node_ids, embedding_property, embeddings, entity_type_str="NODE"):
    """
    Fügt Vektoren für gegebene Knoten-IDs in Neo4j ein oder aktualisiert sie.
    entity_type_str: "NODE" oder "RELATIONSHIP"
    """
    from neo4j import GraphDatabase
    from neo4j_graphrag.indexes import upsert_vectors
    from neo4j_graphrag.types import EntityType
    driver = GraphDatabase.driver(uri, auth=(user, password))
    entity_type = EntityType[entity_type_str.upper()]
    upsert_vectors(
        driver,
        ids=node_ids,
        embedding_property=embedding_property,
        embeddings=embeddings,
        entity_type=entity_type,
    )
    driver.close()
    print(f"Vektoren für {len(node_ids)} {entity_type_str}(s) wurden in Neo4j upserted.")
if __name__ == '__main__':
    main()