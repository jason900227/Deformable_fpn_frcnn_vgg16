import os
import sys
import cv2
import xml.etree.ElementTree as ET

# config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import opt

# dataset
from data.voc_dataset import VOC_BBOX_LABEL_NAMES
from data.util import VOC_COLOR_LIST

def ground_truth(image_dir, annotation_dir, output_dir):
    """
    Draw ground truth bounding boxes on images using corresponding XML annotations 
    and save the results to an output folder.
    
    :param image_dir: Directory containing the original images
    :param annotation_dir: Directory containing the corresponding XML annotation files
    :param output_dir: Directory to save images with drawn bounding boxes
    """

    # If the target folder does not exist, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg'))]

    for i, image_file in enumerate(image_files):
        if i >= opt.n_visual_imgs:
            break

        # Read pictures
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path)

        # Corresponding XML markup file
        xml_file = os.path.join(annotation_dir, os.path.splitext(image_file)[0] + ".xml")
        if not os.path.exists(xml_file):
            print(f"Markup file missing: {xml_file}")
            continue

        # Parsing XML markup files
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Read the bounding box and draw it on the image
        for obj in root.findall("object"):
            class_name = obj.find("name").text
            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)

            # select color
            color = VOC_COLOR_LIST[VOC_BBOX_LABEL_NAMES.index(class_name)]

            # draw bbox rectangle
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)

            # Label (fixed score=1.0 here）
            label = f"{class_name}: 1.0"
            text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            pl = (xmin, ymin - text_size[1] if ymin - text_size[1] > 0 else ymin + text_size[1])
            cv2.rectangle(image, (pl[0] - 2 // 2, pl[1] - 2 - baseline), (pl[0] + text_size[0], pl[1] + text_size[1]), color, -1)
            cv2.putText(image, label, (pl[0], pl[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, 8)

        # save image
        save_path = os.path.join(output_dir, f"{i}.jpg")
        cv2.imwrite(save_path, image)
        print(f"Image {i} saved: {save_path}")

image_dir = "./data/test/VOCdevkit/VOC2007/JPEGImages"
annotation_dir = "./data/test/VOCdevkit/VOC2007/Annotations"
output_dir = "./raw_data/ground_truth(test_data)"

ground_truth(image_dir, annotation_dir, output_dir)