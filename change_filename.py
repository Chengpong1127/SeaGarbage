import xml.etree.ElementTree as ET
import os

# 解析XML文件
tree = ET.parse('dataset/label.xml')  # 替换成你的XML文件路径
root = tree.getroot()

# 重命名图像文件和更新XML中的图像名称
image_counter = 1
for image_elem in root.findall('image'):
    old_image_name = image_elem.get('name')
    image_elem.set('name', f"{image_counter}.jpg")
    
    # 更新图像文件名
    old_image_path = os.path.join(os.path.dirname('dataset/data/'), old_image_name)
    new_image_name = f"{image_counter}.jpg"
    new_image_path = os.path.join(os.path.dirname('dataset/data/'), new_image_name)
    
    os.rename(old_image_path, new_image_path)
    
    image_counter += 1

# 保存修改后的XML文件
tree.write('dataset/new_label.xml')  # 保存到新的XML文件中，你可以替换成你想要的文件名
