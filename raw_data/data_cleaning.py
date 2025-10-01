import os
import json
import shutil
from PIL import Image
import xml.etree.ElementTree as ET

def convert_png_to_jpg(source_folder, target_folder):
    """
    Convert all .png images in the source folder to .jpg format 
    and save them in the target folder.
    :param source_folder: Directory containing the original .png images
    :param target_folder: Directory to save the converted .jpg images
    """
    
    # If the target folder does not exist, create it
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    # Get all files in the folder
    for filename in os.listdir(source_folder):
        if filename.endswith(".png"):
            # Build archive full path
            png_path = os.path.join(source_folder, filename)
            jpg_filename = filename.replace(".png", ".jpg")
            jpg_path = os.path.join(target_folder, jpg_filename)
            
            # Open the PNG image and convert it to JPG
            with Image.open(png_path) as img:
                # PNG images may have a transparent channel, convert to RGB
                rgb_img = img.convert('RGB')  
                rgb_img.save(jpg_path, "JPEG")
            print(f"Conversion completed：{filename} -> {jpg_filename}")

def create_voc_annotation(file_name, width, height, depth, syms, boxes, output_dir):
    """
    Convert the annotation data of a single image into VOC format XML and save it.
    :param file_name: Image file name
    :param width: Image width
    :param height: Image height
    :param depth: Number of image channels
    :param syms: List of annotation classes
    :param boxes: List of bounding boxes (format: [[x1, y1, x2, y2], ...])
    :param output_dir: Output directory
    """

    # If syms is empty, skip it.
    if not syms:
        print(f"Info: No annotations for {file_name}. Skipping.")
        return

    # Create the XML root node
    annotation = ET.Element('annotation')

    # Add basic nodes
    folder = ET.SubElement(annotation, 'folder')
    folder.text = 'VOC'
    filename = ET.SubElement(annotation, 'filename')
    filename.text = file_name

    # Image size, number of channels
    size = ET.SubElement(annotation, 'size')
    width_elem = ET.SubElement(size, 'width')
    width_elem.text = str(width)
    height_elem = ET.SubElement(size, 'height')
    height_elem.text = str(height)
    depth_elem = ET.SubElement(size, 'depth')
    depth_elem.text = str(depth)

    for sym, box in zip(syms, boxes):
        if len(box) != 4 or any(v is None for v in box):
            print(f"Warning: Invalid box data for {file_name}, sym={sym}, box={box}")
            continue

        # Make sure the bounding box coordinates are valid numbers and within the image bounds
        x1, y1, x2, y2 = [max(0, min(int(v), width if i % 2 == 0 else height)) for i, v in enumerate(box)]
        if x1 >= x2 or y1 >= y2:
            print(f"Warning: Invalid box coordinates for {file_name}, sym={sym}, box={box}")
            continue

        obj = ET.SubElement(annotation, 'object')
        name = ET.SubElement(obj, 'name')
        name.text = sym.lower().strip()
        pose = ET.SubElement(obj, 'pose')
        pose.text = 'Unspecified'
        truncated = ET.SubElement(obj, 'truncated')
        truncated.text = '0'
        difficult = ET.SubElement(obj, 'difficult')
        difficult.text = '0'

        bndbox = ET.SubElement(obj, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        xmin.text = str(x1)
        ymin = ET.SubElement(bndbox, 'ymin')
        ymin.text = str(y1)
        xmax = ET.SubElement(bndbox, 'xmax')
        xmax.text = str(x2)
        ymax = ET.SubElement(bndbox, 'ymax')
        ymax.text = str(y2)

    # Save as XML file (with declaration & UTF-8 encoding)
    output_path = os.path.join(output_dir, file_name.replace('.jpg', '.xml'))
    tree = ET.ElementTree(annotation)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)

def json_to_xml(json_file, image_dir, output_dir, default_width=1024, default_height=1024, default_depth=3):
    """
    Convert JSON format annotations to VOC format.
    :param json_file: Path to the input JSON file
    :param image_dir: Directory where the images are located
    :param output_dir: Output directory for VOC format annotations
    :param default_width: Default image width (used if image size cannot be obtained)
    :param default_height: Default image height (used if image size cannot be obtained)
    :param default_depth: Default number of image channels (used if image channels cannot be obtained)
    """

    # If the target folder does not exist, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(json_file, 'r') as f:
        annotations = json.load(f)

    for ann in annotations:
        file_name = ann['file_name']
        syms = ann.get('syms', [])
        boxes = ann.get('boxes', [])

        # JSON annotation uses .png filenames, but the actual images have been converted to .jpg
        if file_name.endswith('.png'):
            jpg_file_name = file_name.replace('.png', '.jpg')
            image_path = os.path.join(image_dir, jpg_file_name)

        # Get image size and number of channels
        try:
            if os.path.exists(image_path):
                img = Image.open(image_path)
                width, height = img.size
                
                # Use getbands() to get the number of channels
                depth = len(img.getbands())
            else:
                raise FileNotFoundError(f"Image file not found: {image_path}")
        except Exception as e:
            # Print the name of the image file whose size cannot be obtained
            print(f"Error: Cannot get size or depth for {file_name}. Using default values. Reason: {str(e)}")
            width, height, depth = default_width, default_height, default_depth

        create_voc_annotation(jpg_file_name, width, height, depth, syms, boxes, output_dir)

def create_txt_from_xml(xml_dir, output_dir, txt_file_name):
    """
    Create a TXT file that lists the filenames (without extensions) of all XML files 
    in the specified directory.
    :param xml_dir: Directory containing the XML files
    :param output_dir: Directory to save the TXT file
    :param txt_file_name: Name of the generated TXT file
    """

    # If the target folder does not exist, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get all XML files in the folder
    xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]

    # Extract the content of the <filename> node in the XML and write it to a TXT file
    with open(os.path.join(output_dir, txt_file_name), 'w') as f:
        for xml_file in xml_files:
            xml_path = os.path.join(xml_dir, xml_file)
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                filename = root.find('filename').text
                
                # Remove the .jpg extension
                f.write(filename.replace('.jpg', '') + '\n')
            except Exception as e:
                print(f"Error processing file {xml_file}: {str(e)}")

def copy_images_by_txt(image_dir, txt_file_path, output_dir):
    """
    Extract images from the image directory based on the filenames listed in a TXT file 
    and save them into a new directory.
    :param image_dir: Directory of the original images
    :param txt_file_path: Path to the TXT file containing image filenames (without extensions)
    :param output_dir: Directory to save the extracted images
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read the contents of a txt file
    with open(txt_file_path, 'r') as f:
        image_names = [line.strip() for line in f.readlines()]

    # Extract the corresponding picture from the picture folder
    for image_name in image_names:
        image_path = os.path.join(image_dir, f"{image_name}.jpg")
        if os.path.exists(image_path):
            shutil.copy(image_path, os.path.join(output_dir, f"{image_name}.jpg"))
        else:
            print(f"Warning: Image {image_name}.jpg not found in {image_dir}")


train_img_png_dir = './raw_data/train_img(png)'
train_img_jpg_dir = './raw_data/train_img(jpg)'
train_json_path = './raw_data/train_label.json'

test_img_png_dir = './raw_data/test_img(png)'
test_img_jpg_dir = './raw_data/test_img(jpg)'
test_json_path = './raw_data/test_label.json'

train_xml_without_normal_dir = './data/train/VOCdevkit/VOC2007/Annotations'
test_xml_without_normal_dir = './data/test/VOCdevkit/VOC2007/Annotations'

train_txt_without_normal_dir = './data/train/VOCdevkit/VOC2007/ImageSets/Main'
test_txt_without_normal_dir = './data/test/VOCdevkit/VOC2007/ImageSets/Main'

train_img_without_normal_dir = './data/train/VOCdevkit/VOC2007/JPEGImages'
test_img_without_normal_dir = './data/test/VOCdevkit/VOC2007/JPEGImages'

convert_png_to_jpg(train_img_png_dir, train_img_jpg_dir)
convert_png_to_jpg(test_img_png_dir, test_img_jpg_dir)

json_to_xml(train_json_path, train_img_jpg_dir, train_xml_without_normal_dir)
json_to_xml(test_json_path, test_img_jpg_dir, test_xml_without_normal_dir)

create_txt_from_xml(train_xml_without_normal_dir, train_txt_without_normal_dir, 'train.txt')
create_txt_from_xml(test_xml_without_normal_dir, test_txt_without_normal_dir, 'test.txt')

train_txt_without_normal_file = os.path.join(train_txt_without_normal_dir, 'train.txt')
test_txt_without_normal_file = os.path.join(test_txt_without_normal_dir, 'test.txt')
copy_images_by_txt(train_img_jpg_dir, train_txt_without_normal_file, train_img_without_normal_dir)
copy_images_by_txt(test_img_jpg_dir, test_txt_without_normal_file, test_img_without_normal_dir)