from langchain_community.llms.openai import OpenAIChat
from neo4j import GraphDatabase

from ldrag.gptconnector import logger


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

def retrieve_nodes_with_llm_query(uri, user, password, user_query):
    """
    Nutzt LangChain und ein LLM, um eine Cypher-Query zu generieren und relevante Knoten aus Neo4j zu holen.
    """
    from langchain_neo4j import Neo4jGraph,GraphCypherQAChain

    from langchain_openai import ChatOpenAI
    graph = Neo4jGraph(url=uri, username=user, password=password)

    llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
    chain = GraphCypherQAChain.from_llm(
        llm,
        graph=graph,
        allow_dangerous_requests=True
    )
    antwort = chain.invoke(user_query)
    return antwort

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

if __name__ == "__main__":
    import os
    uri = os.environ.get("NEO4J_URI", "neo4j+s://ae66b6dc.databases.neo4j.io")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "kcB3a0jyR0GYFy6KUHWiOb5HJf4qtkp6JYR4IrQUdqA")
    user_query = "Which model has the highest ROC AUC?"
    print("Antwort von LLM-Cypher:")
    print(retrieve_nodes_with_llm_query(uri, user, password, user_query))

