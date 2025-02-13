import json


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

    def get_instances_by_class(self, node_class_id):
        """
        Retrieves all instances of a given class.

        :param node_class_id: ID of the node class
        :return: List of GenericNode instances belonging to the class
        """
        return [node for node in self._node_dict.values() if node.get_node_class_id() == node_class_id]

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
