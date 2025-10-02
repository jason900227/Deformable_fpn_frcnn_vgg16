from pprint import pprint

class Config:
    # dataset params
    voc_train_data_dir = './data/train/VOCdevkit/VOC2007/'
    voc_test_data_dir = './data/test/VOCdevkit/VOC2007/'
    min_size = 600      # image resize
    max_size = 1000     # image resize
    train_num_workers = 4
    test_num_workers = 4

    # optimizer params
    weight_decay = 0.0005
    lr = 1e-3
    lr_decay = 0.1

    # model params
    model_name = 'deformable_fpn_frcnn_vgg16' 

    # training params
    nms_thresh = 0.3        # iou threshold in nms
    score_thresh = 0.05     # score threshold in nms
    rpn_sigma = 3.          # rpn sigma for l1_smooth_loss
    roi_sigma = 1.          # roi sigma for l1_smooth_loss
    epoch = 40             # total training epoch
    epoch_decay = 32       # epoch to decay lr

    # testing params
    n_visual_imgs = 20      # number of images to visualize
    visualize = True

    # save
    save_dir = './exp'

    def f_parse_args(self, kwargs):
        # parse user input argument
        params = {k: getattr(self, k) for k, _ in Config.__dict__.items() if not k == 'f_parse_args'}
        for key, value in kwargs.items():
            if key not in params:
                raise ValueError('UnKnown Option: "--%s"' % key)
            setattr(self, key, value)

        print('=================== User config ===============')
        pprint(params)
        print('===============================================')

opt = Config()
