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

#choose the pretrained model.
config_file = "configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"
# update the config options with the config file
cfg.merge_from_file(config_file)
#Load the model
coco_demo = COCODemo(cfg, min_image_size=800, confidence_threshold=0.7,)

vid_path = '/HDD/accident_anticipation/Data/CCD/Crash/000001.mp4'

cap = cv2.VideoCapture(vid_path)
cnt = 0
frames = []
while(cap.isOpened()):
    ret, img = cap.read()
    frames.append(img)
    cnt+=1
    if cnt > 49:
        break
cap.release()
print(cnt)


dataset_path = "E:/Intern/frames/1/"
#dataset_path = "E:/DoTA/dataset/not_ego_involved/0_train_2/frames/"
for f_num in range(5):
    dataset_path = "E:/Intern/frames/%d/"%(f_num+1)
    folder_list = os.listdir(dataset_path)
    for foldername in folder_list:
        #path = dataset_path + foldername +"/images"
        path = dataset_path + foldername
        print(">> Target data folder: ", foldername)
        print(path)
        file_list = os.listdir(path)
        file_list = [file for file in file_list if file.endswith(".jpg")]
        print("Number of frames:", len(file_list))

        trigger = 0

        for i in range(len(file_list)):
            image_path = path+'/'+file_list[i]

            image = Image.open(image_path)
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
                if label_id == 1 or label_id==2 or label_id==3 or label_id==4 or label_id==6 or label_id==8 :
                    if trigger == 0 : #if i==0 and k==0:
                        data = np.array([frame_id, -1, boxes_arr[k][0], boxes_arr[k][1], boxes_arr[k][2]-boxes[k][0], boxes_arr[k][3]-boxes_arr[k][1],scores[k], -1, -1, -1])
                        trigger = 1
                    else :
                        data_temp = np.array([frame_id, -1, boxes_arr[k][0], boxes_arr[k][1], boxes_arr[k][2]-boxes[k][0], boxes_arr[k][3]-boxes_arr[k][1],scores[k], -1, -1, -1])
                        data = np.vstack([data,data_temp])

        os.makedirs("E:/Intern/mask_rcnn_result_kor/%d/"%(f_num+1), exist_ok = True)
        save_path = "E:/Intern/mask_rcnn_result_kor/%d/"%(f_num+1) + foldername + ".txt"
        #save_path = dataset_path + foldername + "/" + foldername+".txt"
        np.savetxt(save_path, data, fmt='%.3f', delimiter=',')
        #print(" Save path: ", save_path)