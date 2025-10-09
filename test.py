import os
import torch

# config
from utils.config import opt

# dataset
from torch.utils.data import DataLoader
from data.dataset import Dataset
from data.voc_dataset import VOC_BBOX_LABEL_NAMES
from data.util import VOC_COLOR_LIST

# model 
from model import FPNFasterRCNNVGG16

# utils
from utils import array_tool as at
from utils.eval_tool import voc_ap
import numpy as np
import cv2

def test(**kwargs):
    # set up cuda
    device = torch.device('cuda:0' if  torch.cuda.is_available() else 'cpu')

    # parse model parameters from config 
    opt.f_parse_args(kwargs)

    # load testing dataset
    print('Load dataset')
    mode = 'vis' if opt.visualize else 'test'
    test_data = Dataset(opt, mode=mode)
    test_dataloader = DataLoader(test_data,
                             batch_size=1,
                             shuffle=False, 
                             num_workers=opt.test_num_workers)
    
    # model construction
    print('load the trained Deformable FPN Faster RCNN Model')
    net = FPNFasterRCNNVGG16(n_fg_class=10).to(device) 

    # load the trained weight
    PATH = f'{opt.save_model_dir}/deformable_fpn_frcnn_vgg16.pth'
    net.load_state_dict(torch.load(PATH))

    # evaluation
    print('Start evaluation')
    net.eval()

    visual_dir = opt.save_visuals_dir
    if not os.path.exists(visual_dir):
        os.makedirs(visual_dir)

    if opt.visualize:
        for i, input_data in enumerate(test_dataloader):
            if i == opt.n_visual_imgs:
                break
            # get original image size
            ori_img, trans_img, scale, ori_size = input_data

            # convert from tensor to numpy array
            scale = at.scalar(scale)
            original_size = [ori_size[0][0].item(), ori_size[1][0].item()]
            
            # prediction
            pred_bboxes, pred_labels, pred_scores = net(trans_img, None, None, scale,original_size)

            # visualize
            # original image
            ori_img = ori_img.squeeze(0).permute(1,2,0).cpu().numpy()
            ori_img = ori_img.astype(np.uint8)
            ori_img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2BGR)

            # draw bboxes
            for bbox, label, score in zip(pred_bboxes, pred_labels, pred_scores):
                if score < 0.5:
                    continue
                class_name = VOC_BBOX_LABEL_NAMES[label]

                # bounding box
                bbox = bbox.astype(np.int32)

                # get bbox coordinate
                ymin, xmin, ymax, xmax = bbox

                # select color
                color = VOC_COLOR_LIST[VOC_BBOX_LABEL_NAMES.index(class_name)]

                # draw bbox rectangle
                cv2.rectangle(ori_img, (xmin, ymin), (xmax, ymax), color, 1)

                # label
                label = class_name + ': ' + str(round(score, 2))
                text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                pl = (xmin, ymin - text_size[1])
                cv2.rectangle(ori_img, (pl[0] - 2//2, pl[1] - 2 - baseline), (pl[0] + text_size[0], pl[1] + text_size[1]), color, -1)
                cv2.putText(ori_img, label, (pl[0], pl[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, 8)
                
            # save image
            cv2.imwrite(f'{visual_dir}/{i}.jpg', ori_img)
            print(f'Image {i} saved')
    else:
        map_result = voc_ap(net, test_dataloader)
        mAP = map_result['mAP']
        print("mAP=={}", mAP)

if __name__ == '__main__':
    test()