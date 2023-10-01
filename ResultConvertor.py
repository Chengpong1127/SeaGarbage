import pandas as pd
import json

def class_to_label(class_number):
    return 'WASTE_' + str(int(class_number + 1))

def get_single_result(filepath):
    df = pd.read_csv(filepath, sep='\s+')
    image_result = {}
    bounding_boxes = [
        {
            'x1' : round(row['xmin']),
            'y1' : round(row['ymin']),
            'x2' : round(row['xmax']),
            'y2' : round(row['ymax']),
            'label' : class_to_label(row['class']),
            'score' : row['confidence']
        }
        for _, row in df.iterrows()
    ]
    image_result['IMG_PATH'] = ""
    image_result['BOUNDING_BOX'] = bounding_boxes
    return image_result


result = get_single_result('result1.txt')
# to json
with open('result1.json', 'w') as f:
    json.dump(result, f, indent=4)