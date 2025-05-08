import json
import logging
import os
import re
import uuid

from pyvis.network import Network

logger = logging.getLogger(__name__)
from .gptconnector import gpt_request_with_history, gpt_request
from .ontology import Ontology


def information_retriever(ontology: Ontology, user_query: str, previous_conversation=None, sleep_time=0):
    return information_retriever_with_graph(ontology, user_query, previous_conversation, sleep_time)[0]


def information_retriever_with_graph(ontology: Ontology, user_query: str =None, previous_conversation=None, sleep_time=0,starting_node:str=None):
    starting_nodes = starting_node
    ontology_structure = ontology.get_ontology_structure()
    if starting_nodes is not None:
        retrieved_node_dict=ontology.get_nodes([starting_node])
        starting_nodes = [ontology.get_node_structure(node) for node in retrieved_node_dict.values()]

    else:
        logger.info("Starting RAG")
        # Find the most relevant node classes

        system_message = f"The following structure illustrates the class level of the ontology, which will be used to answer the subsequent questions. The node classes have instances that are not listed here. :{json.dumps(ontology_structure)}."
        user_message = f"""
        Return 5 possible interpretations of the most relevant node classes for this user query:
        '{user_query}'.
        Use this syntax for each line: [class1, class2, ...].
        Only respond with JSON arrays, one per line.
        """

        gpt_response = gpt_request(
            user_message=user_message,
            system_message=system_message,
            previous_conversation=previous_conversation,
            sleep_time=sleep_time
        )

        # Extract and parse the lists
        possible_class_lists = re.findall(r'\[(.*?)\]', gpt_response)
        parsed_lists = [re.findall(r'\w+', cls_list) for cls_list in possible_class_lists if cls_list.strip()]

        if parsed_lists:
            found_node_class_list = parsed_lists[0]
            logger.info(f"Found node classes (multi-try): {found_node_class_list}")
        else:
            # Fallback: Retry once with simple query
            logger.warning("No valid class list found. Retrying with fallback query...")
            fallback_message = f"Only give as an answer a list of classes (like [class1, class2]) for this query: {user_query}"
            gpt_response_retry = gpt_request(user_message=fallback_message,
                                             system_message=system_message,
                                             previous_conversation=previous_conversation,
                                             sleep_time=sleep_time)
            found_node_class_list = re.findall(r'\w+', gpt_response_retry)
            logger.info(f"Found node classes (fallback): {found_node_class_list}")

        # Identify possible starting nodes
        instance_ids = [node.get_node_id() for node in ontology.get_instances_by_class(found_node_class_list)]
        user_message = f"Here is a list of instances: {str(instance_ids)}. To which of them refers this user query: {user_query}? Only use the correct one. You can ignore spelling error or cases. Return only JSON Syntax without prefix."
        gpt_response = \
            gpt_request_with_history(user_message=user_message,
                                     previous_conversation=previous_conversation,
                                     sleep_time=sleep_time)[0]
        found_node_instances_list = re.findall(r'\w+', gpt_response)
        retrieved_node_dict=ontology.get_nodes(found_node_instances_list)
        logger.info(f"Found node instances: {found_node_instances_list}")
        starting_nodes = [ontology.get_node_structure(node) for node in retrieved_node_dict.values()]


    logger.info("Beginning iterative ontology search ")
    logger.info(f"Iteration 0. Starting node: {starting_nodes}")
    system_message = f"You are given a starting node, which is part of an ontology. Your job is to traverse the ontology to gather enough information to answer given questions. Every node is connected to other nodes. You can find the connections under  \"'Connections':\" in the form of  \"'Connections': <name of the edge> <name of the connected node>. For example  'Connections': trainedWith data_1. You can request new nodes. To do so write [name of the requested node], for example [data_1]. You can ask for more than one instance this way. For example  [data_1, data_2]. As long as you search for new information, only use this syntax, don't explain yourself. Use the exact name of the instance and don't use the edge. Your job is to gather enough information to answer given questions. To do so, traverse trough the ontology. If you have enough information, write \"BREAK\". Use this class level ontology to orientate yourself: {str(ontology_structure)}. Return only JSON Syntax without prefix."
    previous_conversation = [
        {"role": "assistant", "content": "What is the starting node for the user query?"},
        {"role": "user",
         "content": f"This is the starting node: {starting_nodes}. Write Break if you have enough information to answer the query or request new nodes."}
    ]
    gpt_response, history = gpt_request_with_history(user_message=user_query, system_message=system_message,
                                                     previous_conversation=previous_conversation,
                                                     sleep_time=sleep_time)
    loop_count = 0
    while loop_count < 20 and "BREAK" not in gpt_response:
        logger.info(f"Iteration {loop_count}. Requested nodes: {gpt_response}")
        found_node_instances = execute_query(gpt_response, ontology)
        retrieved_information = []
        if found_node_instances:
            logger.info(
                f"Iteration {loop_count}. Nodes found: {[ontology.get_node_structure(node) for node in found_node_instances]}.")
            for node in found_node_instances:
                retrieved_information.append(ontology.get_node_structure(node))
                retrieved_node_dict.update({f"{node.get_node_id()}": node})
        else:
            retrieved_information = "No instance exists for that ID. You asked for a class or searched for a non existing instance."
            logger.info(f"Iteration {loop_count}. No nodes where found.")
        user_message = f"This is the result to your query: {retrieved_information}. If you need more information, use another query, otherwise write BREAK. Return only JSON Syntax without prefix."

        gpt_response, history = gpt_request_with_history(user_message=user_message,
                                                         previous_conversation=history,
                                                         sleep_time=sleep_time)
        loop_count += 1
    logger.info(f"Iterative search ended with {loop_count - 1} iteration.")

    retrieved_graph_id = uuid.uuid1()
    graph_path = create_rag_instance_graph(retrieved_node_dict, retrieved_graph_id, user_query)
    retrieved_relevant_information = []
    for node in retrieved_node_dict.values():
        retrieved_relevant_information.append(ontology.get_node_structure(node))
    # retrieved_relevant_information = [str(obj) for obj in retrieved_node_dict.values()]
    logger.debug(retrieved_relevant_information)
    return retrieved_relevant_information, graph_path


def execute_query(query, ontology):
    pattern = r"(?<=\[|,|\s)([^,\]\s]+)(?=,|\]|\s)"
    matches = re.findall(pattern, query)
    return list(ontology.get_nodes(matches).values())


def create_rag_instance_graph(rag_dict, question_id, question):
    """
    Creates an interactive graph visualization for Retrieval Augmented Generation (RAG) instances.

    :param rag_dict: Dictionary containing nodes and their connections
    :param question_id: Unique identifier for the question
    :param question: The question being visualized
    :return: Path to the saved HTML file containing the graph
    """
    net = Network(height="100vh", width="100vw", directed=True, notebook=False)

    # Header blue color
    header_blue = "#457b9d"

    # Add nodes with custom color
    for node in rag_dict.values():
        net.add_node(
            node.get_node_id(),
            title=node.get_internal_structure(),
            color=header_blue  # Set node color to header blue
        )

    for node in rag_dict.values():
        for connection in node.get_node_connections():
            # Ensure connection is a dictionary and contains necessary keys
            if isinstance(connection, dict) and "target" in connection and "relation" in connection:
                target = connection["target"]
                relation = connection["relation"]

                # Ensure the target node actually exists in rag_dict before adding an edge
                if target in rag_dict:
                    net.add_edge(node.get_node_id(), target, label=relation, arrows="to", length=400)

    output_file = f"static/graph/rag_{question_id}.html"
    directory = os.path.dirname(output_file)
    if not os.path.exists(directory):
        os.makedirs(directory)
    net.save_graph(output_file)

    # Add custom styling and header
    custom_styles = """
    <style>
        body {
            font-family: 'Barlow Semi Condensed', Arial, sans-serif;
            margin: 0;
            background-color: #f4f4f9;
            color: #333;
        }
        #header {
            font-family: 'Barlow Semi Condensed', Arial, sans-serif;
            display: flex;
            align-items: center;
            justify-content: center;
            background-color: #457b9d;
            color: white;
            padding: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
        }
        #logo {
            height: 50px;
            width: auto;
            margin-right: 15px;
        }
        h1 {
            font-family: 'Barlow Semi Condensed', Arial, sans-serif;
            font-size: 1.8rem;
            margin: 0;
        }
        h2 {
            font-family: 'Barlow Semi Condensed', Arial, sans-serif;
            text-align: center;
            color: #457b9d;
            margin-top: 20px;
            font-weight: 700;
        }
        #graph-container {
            margin: 20px auto;
            max-width: 90%;
            border: 1px solid #ddd;
            border-radius: 8px;
            background-color: white;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        }
    </style>
    """

    custom_header = f"""
    <div id="header">
        <img src="../images/kidz.png" alt="Logo" id="logo">
        <h1>Retrieval Augmented Generation</h1>
    </div>
    <h2>{question}</h2>
    <div id="graph-container">
    """

    # Inject custom styles and header into the graph's HTML
    with open(output_file, 'r') as file:
        html_content = file.read()

    html_content = html_content.replace("<head>", f"<head>\n{custom_styles}", 1)
    html_content = html_content.replace("<body>", f"<body>\n{custom_header}", 1)
    html_content = html_content.replace("</body>", "</div>\n</body>", 1)

    with open(output_file, 'w') as file:
        file.write(html_content)

    return output_file


if __name__ == '__main__':
    owl = Ontology()
    owl.deserialize("../ontology_base.json")
    information_retriever_with_graph(user_query="How many entries does the niryo dataset from september have?",
                                     ontology=owl)
