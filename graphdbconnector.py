from rdflib import Graph, URIRef
import json
import tempfile

from rdflib import Graph, URIRef
import json
import tempfile

from rdflib import Graph, URIRef
from SPARQLWrapper import SPARQLWrapper, JSON
import json
import tempfile

from rdflib import Graph, URIRef
from SPARQLWrapper import SPARQLWrapper, JSON
import json
import tempfile

def export_graphdb_to_json(endpoint_url, ontology_graph_uri):
    """
    Exports a GraphDB ontology into the structured JSON format.

    :param endpoint_url: SPARQL endpoint URL of GraphDB.
    :param ontology_graph_uri: The named graph URI in GraphDB to query.
    :return: Path to the temporary JSON file containing the exported ontology.
    """
    # Set up the SPARQL connection
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setReturnFormat(JSON)

    # SPARQL Query to get all triples from the named graph
    sparql.setQuery(f"""
    SELECT ?s ?p ?o WHERE {{
        GRAPH <{ontology_graph_uri}> {{
            ?s ?p ?o
        }}
    }}
    """)

    try:
        results = sparql.query().convert()
        print("SPARQL Query Results:", results)  # Debug output
    except Exception as e:
        print(f"SPARQL Query Failed: {e}")
        return None

    if "results" not in results or "bindings" not in results["results"] or not results["results"]["bindings"]:
        print("No data found in GraphDB for the given named graph.")
        return None  # Avoid writing an empty file

    # Load results into rdflib Graph
    g = Graph()
    for result in results["results"]["bindings"]:
        subject = URIRef(result["s"]["value"])
        predicate = URIRef(result["p"]["value"])
        obj = URIRef(result["o"]["value"]) if result["o"]["type"] == "uri" else result["o"]["value"]
        g.add((subject, predicate, obj))

    # Extract Node Classes
    node_classes = []
    for cls in g.subjects(predicate=URIRef("http://www.w3.org/2000/01/rdf-schema#subClassOf")):
        class_id = str(cls).split("#")[-1]
        class_connections = [
            {"target": str(obj).split("#")[-1], "relation": "subClassOf"}
            for obj in g.objects(subject=cls, predicate=URIRef("http://www.w3.org/2000/01/rdf-schema#subClassOf"))
        ]
        explanation = g.value(subject=cls, predicate=URIRef("http://www.w3.org/2000/01/rdf-schema#comment"))
        node_classes.append({
            "node_class_id": class_id,
            "class_connections": class_connections,
            "explanation": str(explanation) if explanation else ""
        })

    # Extract Node Instances
    node_instances = []
    for instance in g.subjects(predicate=URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")):
        instance_id = str(instance).split("#")[-1]
        class_type = g.value(subject=instance, predicate=URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"))
        class_id = str(class_type).split("#")[-1] if class_type else "Unknown"

        # Extract connections
        connections = [
            {"target": str(obj).split("#")[-1], "relation": str(pred).split("#")[-1]}
            for pred, obj in g.predicate_objects(subject=instance)
            if pred not in [
                URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
                URIRef("http://www.w3.org/2000/01/rdf-schema#label"),
                URIRef("http://www.w3.org/2000/01/rdf-schema#comment")
            ]
        ]

        # Extract other attributes
        attributes = {
            str(pred).split("#")[-1]: str(obj)
            for pred, obj in g.predicate_objects(subject=instance)
            if pred not in [
                URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
                URIRef("http://www.w3.org/2000/01/rdf-schema#label"),
                URIRef("http://www.w3.org/2000/01/rdf-schema#comment")
            ] and isinstance(obj, URIRef) is False
        }

        node_instance = {
            "node_id": instance_id,
            "node_class": class_id,
            "connections": connections,
            **attributes  # Include additional attributes
        }
        node_instances.append(node_instance)

    # Create the final ontology dictionary
    ontology_data = {
        "node_classes": node_classes,
        "node_instances": node_instances
    }

    print("Final JSON Data:", json.dumps(ontology_data, indent=4))  # Debug output

    # Save to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    temp_file_path = temp_file.name

    with open(temp_file_path, "w", encoding="utf-8") as file:
        json.dump(ontology_data, file, indent=4, ensure_ascii=False)

    print(f"Ontology exported to: {temp_file_path}")
    return temp_file_path



if __name__ == '__main__':
    graphdb_sparql_endpoint = "http://localhost:7200/repositories/PrototypTest"
    ontology_graph_uri = "http://example.org/ontology"

    json_file = export_graphdb_to_json(graphdb_sparql_endpoint, ontology_graph_uri)
    print(f"Exported JSON stored at: {json_file}")
