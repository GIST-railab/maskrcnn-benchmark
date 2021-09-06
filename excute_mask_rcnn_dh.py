# execute mask_rcnn
# save txt file
# output : " frame_id, -1, x1, y1, w, h, scores, -1, -1, -1 "<- deep sort want this format

import torch
import numpy as np
import os
from PIL import Image

from maskrcnn_benchmark.config import cfg
from demo.predictor import COCODemo
import cv2

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

GPU_NUM = 0
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)

test_split = 'test.txt'
test_vids = []
with open(test_split, 'r') as f:
    for line in f:
        vid = line.split(' ')[0]
        if 'positive' in vid:
            vid_name = vid.replace('positive/', '')
            vid_name = vid_name.replace('.npz', '')
            test_vids.append(vid_name)


#choose the pretrained model.
config_file = "configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"
# update the config options with the config file
cfg.merge_from_file(config_file)
#Load the model
coco_demo = COCODemo(cfg, min_image_size=800, confidence_threshold=0.7,)

vid_root = '/HDD/accident_anticipation/Data/CCD/Crash'

test_data = {}

for i in range(len(test_vids)):
    vid_name = test_vids[i]
    vid_path = os.path.join(vid_root, vid_name+'.mp4')
    cap = cv2.VideoCapture(vid_path)
    cnt = 0
    frames = []
    while(cap.isOpened()):
        _, img = cap.read()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        frames.append(img_pil)
        cnt+=1
        if cnt > 49:
            break
    cap.release()

    test_data[vid_name] = frames

save_root = '/HDD/accident_anticipation/Data/CCD/BBox/test'
vid_list = list(test_data.keys())
for vid_name in vid_list[92:]:
    frames = test_data[vid_name]
    data = None

    for i in range(len(frames)):
        image = frames[i]
        image = np.array(image)[:,:,[2,1,0]]

        bbox_predictions = coco_demo.compute_prediction(image)
        top_predictions = coco_demo.select_top_predictions(bbox_predictions)

        boxes = top_predictions.bbox
        labels = top_predictions.get_field("labels").tolist()
        scores = top_predictions.get_field("scores").tolist()

        boxes_arr = np.array(boxes) #[x1,y1,x2,y2]

        #start frame is 0
        frame_id = i

        for k in range(len(scores)):
            label_id = labels[k]
            # Only person(1), bicycle(2), car(3), motorcycle(4), bus(6), truck(8)
            if label_id == 1 or label_id==2 or label_id==3 or label_id==4 or label_id==6 or label_id==8:
                if data is not None: #if i==0 and k==0:
                    data_temp = np.array([frame_id, -1, boxes_arr[k][0], boxes_arr[k][1], boxes_arr[k][2] - boxes[k][0],
                                          boxes_arr[k][3] - boxes_arr[k][1], scores[k], -1, -1, -1])
                    data = np.vstack([data, data_temp])

                else:
                    data = np.array([frame_id, -1, boxes_arr[k][0], boxes_arr[k][1], boxes_arr[k][2] - boxes[k][0],
                                     boxes_arr[k][3] - boxes_arr[k][1], scores[k], -1, -1, -1])

    try:
        save_folder = os.path.join(save_root, vid_name)
        os.makedirs(save_folder, exist_ok = True)
        save_path = os.path.join(save_folder, vid_name+'.txt')
        np.savetxt(save_path, data, fmt='%.3f', delimiter=',')
    except:
        print(vid_name, 'passed')
        pass