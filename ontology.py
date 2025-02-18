import json
import os

import networkx as nx
from matplotlib import pyplot as plt
from pyvis.network import Network


def _insert_headline(headline, output_file):
    with open(output_file, 'r') as file:
        html_content = file.read()

    headline_html = f"<h1 style='text-align:center;color:black;font:arial;margin-top:20px;'>{headline}</h1>"
    html_content = html_content.replace('<body>', f'<body>\n{headline_html}', 1)

    with open(output_file, 'w') as file:
        file.write(html_content)


class Ontology:
    """
    Represents an ontology containing nodes and relationships between them.
    """

    def __init__(self):
        """
        Initializes an empty ontology with dictionaries for nodes and node classes.
        """
        self._node_dict = {}
        self._node_class_dict = {}

    def add_node(self, node):
        """
        Adds a node to the ontology.

        :param node: GenericNode instance to be added
        """
        self._node_dict.update({node.get_node_id(): node})

    def add_node_class(self, node_class):
        """
        Adds a node class to the ontology.

        :param node_class: GenericClass instance representing the node class
        """
        self._node_class_dict.update({node_class.get_node_class_id(): node_class})

    def get_node(self, node_id):
        """
        Retrieves a node by its ID.

        :param node_id: ID of the node
        :return: GenericNode instance or None if not found
        """
        return self._node_dict.get(node_id, None)

    def check_if_node_exists(self, node_id):
        """
        Checks if a node exists in the ontology.

        :param node_id: ID of the node
        :return: True if the node exists, False otherwise
        """
        return node_id in self._node_dict

    def check_if_class_exists(self, class_id):
        """
        Checks if a node class exists in the ontology.

        :param class_id: ID of the class
        :return: True if the class exists, False otherwise
        """
        return class_id in self._node_class_dict

    def get_connected_nodes(self, node, depth=1):
        """
        Retrieves nodes connected to a given node up to a specified depth.

        :param node: GenericNode instance representing the starting node
        :param depth: Depth of connections to retrieve
        :return: Dictionary of connected nodes
        """
        connected_nodes = {}
        search_list = [conn["target"] for conn in node.connections]

        while depth > 0:
            depth -= 1
            temporary_search_list = []
            for connection in search_list:
                if connection not in connected_nodes and connection != node.node_id and connection in self._node_dict:
                    connected_nodes[connection] = self._node_dict[connection]
                    for following_connection in self._node_dict[connection].connections:
                        if following_connection["target"] not in connected_nodes:
                            temporary_search_list.append(following_connection["target"])
            search_list = temporary_search_list

        return connected_nodes

    def execute_query(self, node_class, edge, instance):
        """
        Finds nodes connected via a specific edge to a given instance.

        :param node_class: Class of nodes to search
        :param edge: Relationship edge to match
        :param instance: GenericNode instance to check connections for
        :return: List of nodes satisfying the query
        """
        result_list = []
        for node in self._node_dict.values():
            if node.get_node_class_id() == node_class:
                for connection in node.get_node_connections():
                    if connection["relation"] == edge and instance.get_node_id() == connection["target"]:
                        result_list.append(node)
        return result_list

    def deserialize(self, json_file):
        """
        Reads a JSON file and constructs the ontology.

        :param json_file: Path to the JSON file
        """
        with open(json_file, 'r') as file:
            data = json.load(file)

        for node_class in data.get("node_classes", []):
            self.add_node_class(GenericClass(
                node_class_id=node_class.get("node_class_id"),
                class_connections=node_class.get("class_connections", []),
                explanation=node_class.get("explanation", "")
            ))

        for node_instance in data.get("node_instances", []):
            node_id = node_instance.get("node_id")
            node_class = self._node_class_dict.get(node_instance.get("node_class"))
            connections = node_instance.get("connections", [])

            kwargs = {k: v for k, v in node_instance.items() if k not in {"node_id", "node_class", "connections"}}

            self.add_node(GenericNode(node_id=node_id, node_class=node_class, connections=connections, **kwargs))

        return None

    def get_node_structure_by_list(self, node_list):
        """
        Retrieves the structure of multiple nodes.

        :param node_list: List of node instances.
        :return: List of dictionaries containing node structures.
        """
        return [self.get_node_structure(node) for node in node_list]

    def get_node_structure(self, node):
        """
        Retrieves the structure of a single node.

        :param node: Node instance.
        :return: Dictionary containing node details.
        """
        node_structure = {
            "Node Instance ID": node.get_node_id(),
            "Explanation": node.get_explanation(),
            "Connected Instances": ", ".join(
                f"{conn['relation']} {conn['target']} (Node Class {self.get_node(conn['target']).get_node_class_id()})"
                for conn in node.get_node_connections()
            )
        }

        annotations_to_ignore = {"connections", "class_connections", "node_id", "node_class_id", "explanation"}
        annotation_list = [(key, value) for key, value in node.__dict__.items() if key not in annotations_to_ignore]

        node_structure["Annotations"] = annotation_list
        return node_structure

    def get_ontology_node_overview(self):
        """
        Retrieves an overview of all nodes in the ontology.

        :return: List of node IDs.
        """
        return list(self._node_dict.keys())

    def get_instances_by_class(self, node_class_id_list):
        """
        Retrieves all instances belonging to specific node classes.

        :param node_class_id_list: List of node class IDs.
        :return: List of node instances.
        """
        return [node for node in self._node_dict.values() if node.get_node_class_id() in node_class_id_list]

    def create_class_graph(self):
        """
        Generates and displays a graph representation of node classes and their relationships.
        """
        g = nx.DiGraph()

        for node in self._node_dict.values():
            g.add_node(node.get_node_class_id())
            for conn in node.get_class_connections():
                g.add_edge(node.get_node_class_id(), conn["target"], label=conn["relation"])

        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(g)
        nx.draw(g, pos, with_labels=True, node_color='lightblue', node_size=3000, font_size=10, font_weight='bold',
                edge_color='gray', arrowsize=20)
        edge_labels = {(u, v): d["label"] for u, v, d in g.edges(data=True)}
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels)

        plt.title("Graphical Representation of Node Classes and Relationships")
        plt.show()

    def create_instance_graph(self):
        """
        Generates and displays a graph representation of node instances and their relationships.
        """
        g = nx.DiGraph()

        for node_id, node in self._node_dict.items():
            g.add_node(node_id)
            for conn in node.get_node_connections():
                g.add_edge(node_id, conn["target"], label=conn["relation"])

        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(g)
        nx.draw(g, pos, with_labels=True, node_color='lightblue', node_size=3000, font_size=10, font_weight='bold',
                edge_color='gray', arrowsize=20)
        edge_labels = {(u, v): d["label"] for u, v, d in g.edges(data=True)}
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels)

        plt.title("Graphical Representation of Node Instances and Relationships")
        plt.show()

    def create_dynamic_class_graph(self, run_id):
        """
        Generates and saves an interactive HTML visualization of node class relationships.

        :param run_id: Unique identifier for the graph file.
        """
        net = Network(height="100vh", width="100vw", directed=True)

        node_class_set = set()
        for node in self._node_class_dict.values():
            if node.get_node_class_id() not in node_class_set:
                net.add_node(node.get_node_class_id(), title=node.get_internal_structure())
                node_class_set.add(node.get_node_class_id())

        for node in self._node_class_dict.values():
            if node.get_node_class_id() in node_class_set:
                node_class_set.remove(node.get_node_class_id())
                for conn in node.get_class_connections():
                    net.add_edge(node.get_node_class_id(), conn["target"], label=conn["relation"], arrows="to",
                                 length=200)

        output_file = f"graph/{run_id}_class_graph.html"
        net.save_graph(output_file)
        _insert_headline(headline="Ontology Node Class Diagram", output_file=output_file)

    def create_dynamic_instance_graph(self, run_id):
        """
        Generates and saves an interactive HTML visualization of node instances and relationships.

        :param run_id: Unique identifier for the graph file.
        """
        net = Network(height="100vh", width="100vw", directed=True)

        for node in self._node_dict.values():
            net.add_node(node.get_node_id(), title=node.get_internal_structure())

        for node in self._node_dict.values():
            for conn in node.get_node_connections():
                net.add_edge(node.get_node_id(), conn["target"], label=conn["relation"], arrows="to", length=400)

        output_file = f"graph/{run_id}_instance_graph.html"
        directory = os.path.dirname(output_file)
        if not os.path.exists(directory):
            os.makedirs(directory)
        net.save_graph(output_file)
        _insert_headline(headline="Ontology Node Instance Diagram", output_file=output_file)

    def create_rag_instance_graph(self, rag_dict, run_id="rag_graph", question_id=""):
        """
        Generates and saves an interactive HTML visualization of a subgraph based on a given RAG dictionary.

        :param rag_dict: Dictionary containing a subset of ontology nodes.
        :param run_id: Unique identifier for the graph file.
        :param question_id: Identifier for a specific question (optional).
        """
        net = Network(height="100vh", width="100vw", directed=True)

        for node in rag_dict.values():
            net.add_node(node.get_node_id(), title=node.get_internal_structure())

        for node in rag_dict.values():
            for conn in node.get_node_connections():
                if conn["target"] in rag_dict:
                    net.add_edge(node.get_node_id(), conn["target"], label=conn["relation"], arrows="to", length=400)

        output_file = f"graph/{run_id}_{question_id}_rag_instance_graph.html"

        directory = os.path.dirname(output_file)
        if not os.path.exists(directory):
            os.makedirs(directory)
        net.save_graph(output_file)
        _insert_headline(headline=f"ID: {question_id}, RAG Node Instance Diagram", output_file=output_file)

    def get_nodes(self, node_id_list):
        node_dict = {}
        for node_id in node_id_list:
            if self.check_if_node_exists(node_id):
                node_dict.update({node_id: self.get_node(node_id)})
        return node_dict

    def get_ontology_structure(self):
        """
        Retrieves the structure of the ontology by summarizing node classes, explanations, and their connections.

        :return: List of dictionaries containing node class details.
        """
        node_list = []
        used_node_classes = set()

        for node in self._node_dict.values():
            if node.node_class_id not in used_node_classes:
                used_node_classes.add(node.node_class_id)

                # Extracting connected classes properly from new JSON structure
                connected_classes = ", ".join(
                    [f"{conn['relation']} {conn['target']}" for conn in node.get_class_connections()])

                node_structure = {
                    "Node Class ID": node.node_class_id,
                    "Explanation": node.explanation,
                    "Connected Classes": connected_classes
                }

                node_list.append(node_structure)

        return node_list


class Node:
    """
    Represents a node in the ontology.
    """

    def __init__(self, node_id, node_class_id, connections):
        """
        Initializes a Node instance.

        :param node_id: Unique identifier for the node
        :param node_class_id: Identifier for the node's class
        :param connections: List of connections to other nodes
        """
        self.node_id = node_id
        self.node_class_id = node_class_id
        self.connections = connections

    def get_internal_structure(self):
        return list(self.__dict__.keys())

    def get_node_id(self):
        """
        Retrieves the node ID.

        :return: Node ID as a string
        """
        return self.node_id

    def get_node_class_id(self):
        """
        Retrieves the node's class ID.

        :return: Node class ID as a string
        """
        return self.node_class_id

    def get_node_connections(self):
        """
        Retrieves the connections of the node.

        :return: List of node connections
        """
        return self.connections


class GenericNode(Node):
    """
    Represents a specific instance of a node in the ontology with additional attributes.
    """

    def __init__(self, node_id, node_class, connections, **kwargs):
        """
        Initializes a GenericNode instance.

        :param node_id: Unique identifier for the node
        :param node_class: Associated GenericClass instance
        :param connections: List of connections to other nodes
        :param kwargs: Additional attributes for the node
        """
        super().__init__(node_id, node_class.get_node_class_id() if node_class else None, connections)
        self.class_connections = node_class.get_class_connections() if node_class else []
        self.explanation = node_class.get_explanation() if node_class else ""
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_class_connections(self):
        """
        Retrieves class-level connections for this node.

        :return: List of class connections
        """
        return self.class_connections

    def get_explanation(self):
        """
        Retrieves the explanation of this node.

        :return: String explanation of the node
        """
        return self.explanation


class GenericClass:
    """
    Represents a node class in the ontology.
    """

    def __init__(self, node_class_id, class_connections, explanation):
        """
        Initializes a GenericClass instance.

        :param node_class_id: Unique identifier for the class
        :param class_connections: List of connections to other classes
        :param explanation: Description of the class
        """
        self.node_class_id = node_class_id
        self.class_connections = class_connections
        self.explanation = explanation

    def get_node_class_id(self):
        """
        Retrieves the ID of the node class.

        :return: Node class ID as a string
        """
        return self.node_class_id

    def get_class_connections(self):
        """
        Retrieves connections of this class to other classes.

        :return: List of connected classes
        """
        return self.class_connections

    def get_explanation(self):
        """
        Retrieves the explanation of the node class.

        :return: Description of the node class
        """
        return self.explanation

    def get_internal_structure(self):
        """
        Retrieves the internal attributes of the class.

        :return: List of attribute names
        """
        return list(self.__dict__.keys())
