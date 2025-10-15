import os
import numpy as np
import xml.etree.ElementTree as ET

from data.util import read_image

class VOCBboxDataset:
    def __init__(self, data_dir, split, use_difficult=False):
        id_list_file = os.path.join(data_dir, 'ImageSets/Main/{0}.txt'.format(split))

        self.ids = [id_.strip() for id_ in open(id_list_file)]
        self.data_dir = data_dir
        self.use_difficult = use_difficult
        self.label_names = VOC_BBOX_LABEL_NAMES

    def __len__(self):
        return len(self.ids)
    
    def get_sample(self, i):
        """Returns the i-th example.

        Returns a color image and bounding boxes. The image is in CHW format.
        The returned image is RGB.
        Args:
            i (int): The index of the example.
        Returns:
            tuple of an image and bounding boxes
        """

        id_ = self.ids[i]

        # parse annotation
        anno = ET.parse(os.path.join(self.data_dir, 'Annotations', id_ + '.xml'))
        bbox = []
        label = []
        difficult = []
        for obj in anno.findall('object'):
            # when in not using difficult split, and the object is
            # difficult, skipt it.
            if not self.use_difficult and int(obj.find('difficult').text) == 1:
                continue

            difficult.append(int(obj.find('difficult').text))
            bndbox_anno = obj.find('bndbox')

            # subtract 1 to make pixel indexes 0-based
            bbox.append([int(bndbox_anno.find(tag).text) - 1 for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
            name = obj.find('name').text.lower().strip()
            try:
                label.append(VOC_BBOX_LABEL_NAMES.index(name))
            except ValueError:
                print(f"Unknown label: {name}")
                continue
            
        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)

        # When `use_difficult==False`, all elements in `difficult` are False.
        difficult = np.array(difficult, dtype=bool).astype(np.uint8)

        # Load a image
        img_file = os.path.join(self.data_dir, 'JPEGImages', id_ + '.jpg')
        img = read_image(img_file, color=True)

        return img, bbox, label, difficult
    
    __getitem__= get_sample

VOC_BBOX_LABEL_NAMES = (
    'consolidation',
    'pneumothorax',
    'mass',
    'nodule',
    'effusion',
    'fibrosis',
    'emphysema',
    'fracture',
    'calcification',
    'atelectasis'
)