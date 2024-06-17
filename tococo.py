import os
from glob import glob
import shutil
from sklearn.model_selection import train_test_split

xml_files = glob("./baobabs_images/xml/*.xml")
image_files = glob("./baobabs_images/jpg/*.jpg")
train_names, temp_names = train_test_split(image_files, test_size=0.3)
val_names, test_names = train_test_split(temp_names, test_size=0.5)

os.makedirs("./baobabs_dataset/train")
os.makedirs("./baobabs_dataset/val")
os.makedirs("./baobabs_dataset/test")
os.makedirs("./baobabs_dataset/train_annot")
os.makedirs("./baobabs_dataset/val_annot")
os.makedirs("./baobabs_dataset/test_annot")

train = "./baobabs_dataset/train"
val = "./baobabs_dataset/val"
test = "./baobabs_dataset/test"
train_annot = "./baobabs_dataset/train_annot"
val_annot = "./baobabs_dataset/val_annot"
test_annot = "./baobabs_dataset/test_annot"

def move_img_files(file_list, destination_path):
    for file in file_list: #extracting only the name of the file and concatenating with extenions
        shutil.copy(file, destination_path)        
    return

def move_xml_files(file_list, destination_path):
    for file in file_list:
        file = file.replace(".jpg", ".xml").replace("jpg", "xml")
        try:
            shutil.copy(file, destination_path) 
        except:
            pass
    return     

move_img_files(train_names,train)
move_img_files(val_names,val)
move_img_files(test_names,test)
move_xml_files(train_names,train_annot)
move_xml_files(val_names,val_annot)
move_xml_files(test_names,test_annot)

import xml.etree.ElementTree as ET
import os
import json

image_id = 20180000000
annotation_id = 0

def addCatItem(name, coco, category_set, category_item_id):
    category_item = dict()
    category_item['supercategory'] = 'none'
    category_item_id += 1
    category_item['id'] = category_item_id
    category_item['name'] = name
    coco['categories'].append(category_item)
    category_set[name] = category_item_id
    return category_item_id

def addImgItem(file_name, size, coco, image_set):
    global image_id
    if file_name is None:
        raise Exception('Could not find filename tag in xml file.')
    if size['width'] is None:
        raise Exception('Could not find width tag in xml file.')
    if size['height'] is None:
        raise Exception('Could not find height tag in xml file.')
    image_id += 1
    image_item = dict()
    image_item['id'] = image_id
    image_item['file_name'] = file_name
    image_item['width'] = size['width']
    image_item['height'] = size['height']
    coco['images'].append(image_item)
    image_set.add(file_name)

    return image_id

def addAnnoItem(object_name, image_id, category_id, bbox, coco):
    global annotation_id
    annotation_item = dict()
    annotation_item['segmentation'] = []
    seg = []
    #bbox[] is x,y,w,h
    #left_top
    seg.append(bbox[0])
    seg.append(bbox[1])
    #left_bottom
    seg.append(bbox[0])
    seg.append(bbox[1] + bbox[3])
    #right_bottom
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1] + bbox[3])
    #right_top
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1])

    annotation_item['segmentation'].append(seg)

    annotation_item['area'] = bbox[2] * bbox[3]
    annotation_item['iscrowd'] = 0
    annotation_item['ignore'] = 0
    annotation_item['image_id'] = image_id
    annotation_item['bbox'] = bbox
    annotation_item['category_id'] = category_id
    annotation_id += 1
    annotation_item['id'] = annotation_id
    coco['annotations'].append(annotation_item)

def parseXmlFiles(xml_path):
    coco = dict()
    coco['images'] = []
    coco['type'] = 'instances'
    coco['annotations'] = []
    coco['categories'] = []

    category_item_id = 0

    category_set = dict()
    image_set = set()
    for f in os.listdir(xml_path):
        if not f.endswith('.xml'):
            continue

        bndbox = dict()
        size = dict()
        current_image_id = None
        current_category_id = None
        file_name = None
        size['width'] = None
        size['height'] = None
        size['depth'] = None

        xml_file = os.path.join(xml_path, f)

        tree = ET.parse(xml_file)
        root = tree.getroot()
        if root.tag != 'annotation':
            raise Exception('pascal voc xml root element should be annotation, rather than {}'.format(root.tag))

        #elem is <folder>, <filename>, <size>, <object>
        for elem in root:
            current_parent = elem.tag
            current_sub = None
            object_name = None

            if elem.tag == 'folder':
                continue

            if elem.tag == 'filename':
                file_name = elem.text
                if file_name in category_set:
                    raise Exception('file_name duplicated')
            
            #add img item only after parse <size> tag
            elif current_image_id is None and file_name is not None and size['width'] is not None:
                if file_name not in image_set:
                    current_image_id = addImgItem(file_name, size, coco, image_set)
                else:
                    raise Exception('duplicated image: {}'.format(file_name))

            #subelem is <width>, <height>, <depth>, <name>, <bndbox>
            for subelem in elem:
                bndbox ['xmin'] = None
                bndbox ['xmax'] = None
                bndbox ['ymin'] = None
                bndbox ['ymax'] = None

                current_sub = subelem.tag
                if current_parent == 'object' and subelem.tag == 'name':
                    object_name = subelem.text
                    # Comentar dependiendo de si se quiere las imágenes 
                    # etiquetadas con sombra(agrandidieri) o sin sombra (no-shadow)
                    
                    #if object_name == 'no-shadow':
                    if object_name == 'agrandidieri':
                        if object_name not in category_set:
                            current_category_id = addCatItem(object_name, coco, category_set, category_item_id)
                            category_item_id = current_category_id
                        else:
                            current_category_id = category_set[object_name]

                elif current_parent == 'size':
                    if size[subelem.tag] is not None:
                        raise Exception('xml structure broken at size tag.')
                    size[subelem.tag] = int(subelem.text)

                for option in subelem:
                    if current_sub == 'bndbox':
                        if bndbox[option.tag] is not None:
                            raise Exception('xml structure corrupted at bndbox tag.')
                        bndbox[option.tag] = int(option.text)

                #only after parse the <object> tag
                if bndbox['xmin'] is not None:
                    if object_name is None:
                        raise Exception('xml structure broken at bndbox tag')
                    if current_image_id is None:
                        raise Exception('xml structure broken at bndbox tag')
                    
                    # Comentar dependiendo de si se quiere las imágenes 
                    # etiquetadas con sombra(agrandidieri) o sin sombra (no-shadow)

                    #if current_category_id is None and (object_name == 'no-shadow'):
                    if current_category_id is None and (object_name == 'agrandidieri'):
                        raise Exception('xml structure broken at bndbox tag')
                    bbox = []
                    #x
                    bbox.append(bndbox['xmin'])
                    #y
                    bbox.append(bndbox['ymin'])
                    #w
                    bbox.append(bndbox['xmax'] - bndbox['xmin'])
                    #h
                    bbox.append(bndbox['ymax'] - bndbox['ymin'])
                    
                    if current_category_id == 1 or current_category_id == 2:
                        addAnnoItem(object_name, current_image_id, current_category_id, bbox, coco)
    return coco



def build_coco_annot(xml_path, output_path):
    coco = parseXmlFiles(xml_path)
    json.dump(coco, open(output_path, 'w'))

build_coco_annot(train_annot, train+"/train.json")
build_coco_annot(val_annot, val+"/val.json")
build_coco_annot(test_annot, test+"/test.json")

shutil.rmtree(train_annot)
shutil.rmtree(val_annot)
shutil.rmtree(test_annot)

