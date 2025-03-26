import copy

from flask import Flask, request, jsonify, render_template, session

from ldrag.gptconnector import gpt_request_with_history
from ldrag.ontology import Ontology
from ldrag.retriever import information_retriever_with_graph

app = Flask(__name__)
owl = Ontology()
owl.deserialize("./ontology/ontology.json")
app.secret_key = 'BzPopVRViW'

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)


def rag(question):
    logging.debug(f"Conversation History: {session.get('conversation_history', None)}")
    retrieved_information, graph_path = information_retriever_with_graph(ontology=owl,
                                                                         user_query=question,
                                                                         previous_conversation=copy.deepcopy(
                                                                             session.get('conversation_history', [])))
    logging.info(f"Retrieved information: {retrieved_information}")
    logging.debug(f"Conversation History: {session.get('conversation_history', None)}")
    gpt_response, history = gpt_request_with_history(user_message=question,
                                                     previous_conversation=copy.deepcopy(
                                                         session.get('conversation_history', [])),
                                                     retrieved_information=retrieved_information)
    logging.info(f"History: {history}")
    session['conversation_history'] = history
    logging.info(f"Assistend response: {gpt_response}")
    return gpt_response, graph_path


@app.route("/")
def index():
    session['conversation_history'] = []
    return render_template("index.html")


@app.route("/rag", methods=["POST"])
def rag_endpoint():
    data = request.json
    input_text = data.get("input", "")
    logging.info(f"Incoming User request: {input_text}")
    if not input_text:
        return jsonify({"response": "Error: No input provided."}), 400
    response, graph_path = rag(input_text)
    return jsonify({"response": response,
                    "dynamicFileUrl": f"{graph_path}"})


if __name__ == "__main__":
    app.run(debug=True)
