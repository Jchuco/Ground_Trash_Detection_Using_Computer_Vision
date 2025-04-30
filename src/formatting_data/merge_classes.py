import json

def guess_superclass(name):
    name_lower = name.lower()
    
    superclass_keywords = {
        'plastic': {'plastic', 'styrofoam'},
        'glass': {'glass', 'bottle', 'jar', 'window', 'mirror'},
        'metal': {'metal', 'aluminium', 'steel', 'can', 'pop tab'},
        'paper': {'paper', 'cardboard', 'carton', 'newspaper', 'magazine', 'carded'},
    }
    
    for superclass, keywords in superclass_keywords.items():
        if any(keyword in name_lower for keyword in keywords):
            return superclass
    
    return 'other'

superclass_to_id = {
    'plastic': 0,
    'paper': 1,
    'metal': 2,
    'glass': 3,
    'other': 4
}

def merge_classes_to_superclasses(input_json_path, output_json_path):
    with open(input_json_path, 'r') as f:
        data = json.load(f)

    id_to_superclass = {}
    for category in data['categories']:
        old_id = category['id']
        name = category['name']
        supercat = guess_superclass(name)
        id_to_superclass[old_id] = superclass_to_id[supercat]


    new_categories = [
        {'id': 0, 'name': 'plastic', 'supercategory': 'plastic'},
        {'id': 1, 'name': 'paper', 'supercategory': 'paper'},
        {'id': 2, 'name': 'metal', 'supercategory': 'metal'},
        {'id': 3, 'name': 'glass', 'supercategory': 'glass'},
        {'id': 4, 'name': 'other', 'supercategory': 'other'},
    ]

    new_annotations = []
    for ann in data['annotations']:
        ann['category_id'] = id_to_superclass[ann['category_id']]
        new_annotations.append(ann)

    new_data = {
        'info': data.get('info', {}),
        'licenses': data.get('licenses', []),
        'images': data['images'],
        'annotations': new_annotations,
        'categories': new_categories
    }

    with open(output_json_path, 'w') as f:
        json.dump(new_data, f, indent=2)

