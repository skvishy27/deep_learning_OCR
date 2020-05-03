import cv2
import os
import time
import sys
import argparse
import sys
import shutil
from argparse import Namespace
from collections import OrderedDict
import numpy as np
from skimage import io

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import craft
import imgproc
import craft_utils
import demo


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, 1280, interpolation=cv2.INTER_LINEAR,
                                                                          mag_ratio=1.5)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    return boxes, polys, ret_score_text


def detect_text(img, text_threshold=0.7, link_threshold=0.4, low_text=0.4):
    bboxes, polys, score_text = test_net(net, img, text_threshold, link_threshold, low_text, False, False,
                                         refine_net)
    return bboxes, polys, score_text

def show_bounding_boxes(img, bboxes):
    img = np.array(img)
    for i, box in enumerate(bboxes):
        poly = np.array(box).astype(np.int32).reshape((-1))

        poly = poly.reshape(-1, 2)
        cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)

    return img

def bounding_box(points):
    points = points.astype(np.int16)
    x_coordinates, y_coordinates = zip(*points)
    return [(min(x_coordinates), min(y_coordinates)), (max(x_coordinates), max(y_coordinates))]

if __name__ == '__main__':

    arg = argparse.ArgumentParser()
    arg.add_argument("-i" ,"--input", required=True, 
                    help='path to input images')
    arg.add_argument("-o" ,"--output", required=True, 
                    help='path to input images')    
    args = vars(arg.parse_args())

    image = cv2.imread(args['input'])
    # print(image.shape)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # # print(gray.shape)
    # blur = cv2.GaussianBlur(gray, (5,5), 0)
    # # thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)[1]
    # # thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # ret,thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # thresh = np.expand_dims(thresh, axis=2)
    # cv2.imwrite('fram.png',thresh)
    # cv2.imshow('thresh_image',thresh)
    # cv2.waitKey(0)
    # print(thresh.shape)
    global net, refine_net

    net = craft.CRAFT()
    net.load_state_dict(copyStateDict(torch.load('weights/craft_mlt_25k.pth', map_location='cpu')))
    net.eval()
    refine_net = None

    bboxes, polys, heatmap = detect_text(image)
    img_boxed = show_bounding_boxes(image,bboxes)

    filename = os.path.basename(args['input'])
    output_dir = args['output']+f'/{filename}'
    # print(output_dir)
    # cv2.imwrite(output_dir,img_boxed)

    # res_file = args['output'] + f"/res_{filename.split('.')[0]}.txt"
    # with open(res_file, 'w') as f:
    #     for i, box in enumerate(polys):
    #         poly = np.array(box).astype(np.int32).reshape((-1))
    #         strResult = ','.join([str(p) for p in poly]) + '\r\n'
    #         f.write(strResult)
    # cv2.imshow('fig', img_boxed)
    # cv2.waitKey(0)
    # cv2.imshow('fig', heatmap)

    # detection
    # bboxes, score_text = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda)

    for i, bbs in enumerate(bboxes):
        # print('a')
        crop = bounding_box(bbs)
        # print('b')
        cropped = image[crop[0][1]:crop[1][1],crop[0][0]:crop[1][0]]
        # print('c')
        
        cv2.imwrite(args['output'] + '/res_' + filename.split('.')[0] + '_cropped_' + str(i) + '.png', cropped)


    opt = Namespace(FeatureExtraction='ResNet', PAD=False, Prediction='Attn', SequenceModeling='BiLSTM', 
                    Transformation='TPS', batch_max_length=25, batch_size=192, 
                    character='0123456789abcdefghijklmnopqrstuvwxyz', 
                    hidden_size=256, image_folder=args['output'], imgH=32, imgW=100, input_channel=1, 
                    num_fiducial=20, num_gpu=0, output_channel=512, rgb=False, 
                    saved_model='weights/TPS-ResNet-BiLSTM-Attn.pth', sensitive=False, workers=4)
    opt.num_gpu = torch.cuda.device_count()
    extract_text = demo.demo(opt)
    print(extract_text)

    for filename in os.listdir(args['output']):
        file_path = os.path.join(args['output'], filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))