objects_node_dict = {
    "ON0001": {
        "id": "ON0001",
        "lable": "table",
        "in_edge": [],
        "out_edge": ["E0001"],
    },
}

relations_node_dict = {
    "RN0001": {
        "id": "RN0001",
        "lable": "top of",
        "type": "spatial",
        "in_edge": ["E0001"],
        "out_edge": ["E0002", "E0003", "E0004", "E0005"],
    },
    "RN0001": {
        "id": "RN0002",
        "lable": "surrounded",
        "type": "semantic",
        "in_edge": [...],
        "out_edge": [...],
    },
}

edge_dict = {
    "E0001": {
        "id": "E0001",
        "link" : ["ON0001", "RN0001"]
    },
    "E0002": {
        "id": "E0002",
        "link" : ["RN0001", "ON0002"] # ON0002 == "vase"
    },
    "E0003": {
        "id": "E0003",
        "link" : ["RN0001", "ON0003"] # ON0003 == "teapot"
    },
    "E0004": {
        "id": "E0004",
        "link" : ["RN0001", "ON0004"] # ON0004 == "dinner plates"
    },
}