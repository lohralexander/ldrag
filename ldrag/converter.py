import xml.etree.ElementTree as ET
import json
import os
from pprint import pprint


def process_graphml_to_json(graphml_file, json_file, overwrite=False):
    if os.path.exists(json_file) and not overwrite:
        with open(json_file, 'r', encoding='utf-8') as f:
            try:
                json_data = json.load(f)
                print(f"Loaded existing JSON with {len(json_data.get('node_instances', []))} instances.")
            except json.JSONDecodeError:
                print("Invalid JSON. Creating new.")
                json_data = {'node_classes': [], 'node_instances': []}
    else:
        print("Creating new JSON structure")
        json_data = {'node_classes': [], 'node_instances': []}

    tree = ET.parse(graphml_file)
    root = tree.getroot()
    ns = {'g': 'http://graphml.graphdrawing.org/xmlns'}

    keys = {}
    for key_elem in root.findall('g:key', ns):
        keys[key_elem.get('id')] = {
            'name': key_elem.get('attr.name'),
            'for': key_elem.get('for')
        }

    nodes = {}
    new_classes_needed = set()

    for node in root.findall('.//g:node', ns):
        node_id = node.get('id')
        node_data = {'node_id': node_id, 'connections': []}

        for data in node.findall('g:data', ns):
            key = data.get('key')
            if key in keys:
                key_info = keys[key]
                attr_name = key_info['name']
                if attr_name == 'label' and key_info['for'] == 'node':
                    node_data['node_class'] = data.text
                    new_classes_needed.add(data.text)
                elif '::' in attr_name:
                    _, prop = attr_name.split('::', 1)
                    prop = prop.split('--')[0]
                    node_data[prop] = data.text
                else:
                    node_data[attr_name] = data.text

        nodes[node_id] = node_data

    for edge in root.findall('.//g:edge', ns):
        source = edge.get('source')
        target = edge.get('target')
        relation = None
        for data in edge.findall('g:data', ns):
            key = data.get('key')
            if key in keys and keys[key]['for'] == 'edge':
                relation = data.text
        if source in nodes:
            nodes[source]['connections'].append({
                'target': target,
                'relation': relation
            })

    existing_ids = {node['node_id'] for node in json_data.get('node_instances', [])}
    new_nodes = [node for nid, node in nodes.items() if nid not in existing_ids]
    json_data['node_instances'].extend(new_nodes)

    # Handle missing node classes
    existing_class_ids = {cls['node_class_id'] for cls in json_data.get('node_classes', [])}
    missing_classes = new_classes_needed - existing_class_ids

    if missing_classes:
        print(f"Missing node classes to add: {missing_classes}")
        # Try to find their definitions in the current ontology
        with open(json_file, 'r', encoding='utf-8') as f:
            full_ontology = json.load(f)

        ontology_classes = {cls['node_class_id']: cls for cls in full_ontology.get('node_classes', [])}

        for cls_id in missing_classes:
            if cls_id in ontology_classes:
                json_data['node_classes'].append(ontology_classes[cls_id])
                print(f"Added missing class: {cls_id}")
            else:
                print(f"⚠ Warning: Class '{cls_id}' not found in ontology. Added with empty definition.")
                json_data['node_classes'].append({
                    'node_class_id': cls_id,
                    'class_connections': [],
                })

    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=4)

    print(f"Saved updated JSON to: {json_file}")
    return json_data



# Example usage (no CLI for simplicity here)
if __name__ == "__main__":
    process_graphml_to_json('ExporterTest.graphml', 'ontology_43.json', overwrite=False)
