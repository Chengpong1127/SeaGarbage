
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
import torchvision
import os
from transformers import AutoImageProcessor
import torch

class GarbageDataset(Dataset):
    def __init__(self, model_name, label_filename, image_folder, size):
        self.load_data(label_filename, image_folder)
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.image_size = {
            'height': size[1],
            'width': size[0]
        }
    def load_data(self, label_filename, image_folder):
        self.data = []
        tree = ET.parse(label_filename)
        root = tree.getroot()
        for image in root.findall('.//image'):
            image_id = image.get('id')
            image_name = os.path.join(image_folder, image.get('name'))
            image_height = int(image.get('height'))
            image_width = int(image.get('width'))
            
            
            polygons = image.findall('.//polygon')
            labels = []
            points = []
            for polygon in polygons:
                label = polygon.get('label')
                label = int(label.split('_')[1]) # 获取label中的数字部分
                labels.append(label)
                points_str = polygon.get('points')
                points_list = [tuple(map(int, point.split(','))) for point in points_str.split(';')]
                points.append(points_list)
            self.data.append([image_id, image_name, image_height, image_width, labels, points])
            
    def __len__(self):
        return len(self.data)
    
    def get_label(self, data):
        label = data[4]
        points = data[5]
        label_dict = {
            'class_labels': [],
            'boxes': []
        }
        for i in range(len(label)):
            label_dict['class_labels'].append(label[i])
            label_dict['boxes'].append(self.get_yolos_boxes(*self.get_xmin_ymin_xmax_ymax(points[i]), data[3], data[2]))
        label_dict['class_labels'] = torch.tensor(label_dict['class_labels'])
        label_dict['boxes'] = torch.tensor(label_dict['boxes'])
        return label_dict
    
    def get_xmin_ymin_xmax_ymax(self, points):
        xmin = min([point[0] for point in points])
        ymin = min([point[1] for point in points])
        xmax = max([point[0] for point in points])
        ymax = max([point[1] for point in points])
        return xmin, ymin, xmax, ymax
    
    def get_yolos_boxes(self, xmin, ymin, xmax, ymax, image_width, image_height):
        return [
            (xmin + xmax) / 2 / image_width,
            (ymin + ymax) / 2 / image_height,
            (xmax - xmin) / image_width,
            (ymax - ymin) / image_height
        ]
        
        
    
    def __getitem__(self, idx):
        image = torchvision.io.read_image(self.data[idx][1])
        image = self.image_processor(image, return_tensors="pt", do_resize=True, size=self.image_size)
        return image['pixel_values'], self.get_label(self.data[idx])
    
    
def collate_fn(batch):
    images = []
    labels = []
    for image, label in batch:
        images.append(image)
        labels.append(label)
    return torch.stack(images).squeeze(1), labels