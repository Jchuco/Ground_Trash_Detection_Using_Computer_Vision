import json

def guess_superclass(name):
    name = name.lower()
    if 'plastic' in name or 'styrofoam' in name :
        return 'plastic'
    elif 'glass' in name or 'bottle' in name or 'jar' in name or 'window' in name or 'mirror' in name:
        return 'glass'
    elif 'metal' in name or 'aluminium' in name or 'steel' in name or 'can' in name or 'pop tab' in name:
        return 'metal'
    elif 'paper' in name or 'cardboard' in name or 'carton' in name or 'newspaper' in name or 'magazine' in name or 'carded' in name:
        return 'paper'
    else:
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

    print("\nMapeamento de classes feito automaticamente:")
    for old_id, super_id in id_to_superclass.items():
        print(f"Categoria original {old_id} → Superclasse {super_id}")

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

    print(f"\nNovo ficheiro JSON criado: {output_json_path}")

merge_classes_to_superclasses(
    input_json_path='./data/annotations.json',
    output_json_path='./data/annotations_5classes.json'
)
