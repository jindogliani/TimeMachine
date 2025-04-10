'''
post-processes the final outcome of Reflexion Prompts from LLMs 

'''

import json
import os
from datetime import datetime
import re

# Sample Input Text from Reflexion Prompts
text_1 = """
Step 4: Output the Scene Graph
Objects: [table, vase, teapot, dinner plates, chair1, chair2, chair3, chair4]  
Relations: [top of, surrounded]
- Spatial: [top of]
- Semantic: [surrounded]
Scene Graph:  
table → top of → [vase, teapot, dinner plates]  
table → surrounded → [chair1, chair2, chair3, chair4]
"""

def parse_text_data(text):
    # Objects Relations words extract patterns
    objects_pattern = r"Objects: \[([^\]]+)\]"
    relations_pattern = r"Relations: \[([^\]]+)\]"
    spatial_pattern = r"- Spatial: \[([^\]]+)\]"
    semantic_pattern = r"- Semantic: \[([^\]]+)\]"
    scene_graph_pattern = r"(.*) → (.*) → \[([^\]]+)\]"
    # Objects Relations words search
    objects_match = re.search(objects_pattern, text)
    relations_match = re.search(relations_pattern, text)
    spatial_match = re.search(spatial_pattern, text)
    semantic_match = re.search(semantic_pattern, text)
    scene_graph_matches = re.findall(scene_graph_pattern, text)

    # words extraction and to the list
    objects = [obj.strip() for obj in objects_match.group(1).split(",")] if objects_match else []
    relations = [rel.strip() for rel in relations_match.group(1).split(",")] if relations_match else []

    spatial_relations = set(spatial_match.group(1).split(", ")) if spatial_match else set()
    semantic_relations = set(semantic_match.group(1).split(", ")) if semantic_match else set()
    relation_types = {rel: "spatial" if rel in spatial_relations else "semantic" for rel in relations}

    # generating temp SG triplet dict
    temp_SG = {}
    for source, relation, targets in scene_graph_matches:
        source = source.strip()
        relation = relation.strip()
        targets = [target.strip() for target in targets.split(",")]
        if source not in temp_SG:
            temp_SG[source] = {}
        temp_SG[source][relation] = targets
    
    return objects, relations, relation_types, temp_SG

# ID generation
def generate_id(prefix, counter):
    return f"{prefix}{str(counter).zfill(4)}"

def create_nodes_and_edges(objects, relations, relation_types, temp_SG):

    objects_node_dict = {}
    relations_node_dict = {}
    edge_dict = {}

    object_counter = 1
    relation_counter = 1
    edge_counter = 1

    # Object nodes generation
    object_id_map = {} #나중에 table을 넣었을 때, ON0001을 뱉어내게 하기 위하여
    for obj in objects:
        obj_id = generate_id("ON", object_counter)
        object_id_map[obj] = obj_id
        objects_node_dict[obj_id] = {
            "id": obj_id,
            "label": obj,
            "in_edge": [],
            "out_edge": []
        }
        object_counter += 1

    # temp_SG 기반으로 Relation Node dict와 Edge dict생성
    for source, relations in temp_SG.items():
        source_id = object_id_map[source]
        for relation, targets in relations.items():
            # Relation 노드 생성
            relation_id = generate_id("RN", relation_counter)
            relations_node_dict[relation_id] = {
                "id": relation_id,
                "label": relation,
                "type": relation_types[relation],
                "in_edge": [],
                "out_edge": []
            }
            relation_counter += 1

            # Source -> Relation Edge 생성
            edge_id = generate_id("E", edge_counter)
            edge_dict[edge_id] = {
                "id": edge_id,
                "link": [source_id, relation_id]
            }
            edge_counter += 1

            # 연결 정보 업데이트
            objects_node_dict[source_id]["out_edge"].append(edge_id)
            relations_node_dict[relation_id]["in_edge"].append(edge_id)

            # Relation -> Target Object Edge 생성
            for target in targets:
                target_id = object_id_map[target]
                edge_id = generate_id("E", edge_counter)
                edge_dict[edge_id] = {
                    "id": edge_id,
                    "link": [relation_id, target_id]
                }
                edge_counter += 1

                # 연결 정보 업데이트
                relations_node_dict[relation_id]["out_edge"].append(edge_id)
                objects_node_dict[target_id]["in_edge"].append(edge_id)

    return objects_node_dict, relations_node_dict, edge_dict

# execution
objects, relations, relation_types, temp_SG = parse_text_data(text_1)
objects_node_dict, relations_node_dict, edge_dict = create_nodes_and_edges(objects, relations, relation_types, temp_SG)


print("Objects Node Dict:")
print(objects_node_dict)
print("\nRelations Node Dict:")
print(relations_node_dict)
print("\nEdge Dict:")
print(edge_dict)

# scene_dict = {
#     "objects": objects,
#     "relations": relations
# }

today = datetime.now().strftime("%m%d")
dir = f"../SceneGraphs/{today}_SG"
os.makedirs(dir, exist_ok=True)

objects_file_path = os.path.join(dir, "1_objects_node.json")
relations_file_path = os.path.join(dir, "2_relations_node.json")
edges_file_path = os.path.join(dir, "3_edge.json")

with open(objects_file_path, 'w', encoding='utf-8') as f:
    json.dump(objects_node_dict, f, indent=4, ensure_ascii=False)

with open(relations_file_path, 'w', encoding='utf-8') as f:
    json.dump(relations_node_dict, f, indent=4, ensure_ascii=False)

with open(edges_file_path, 'w', encoding='utf-8') as f:
    json.dump(edge_dict, f, indent=4, ensure_ascii=False)