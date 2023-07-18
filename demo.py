# ----------------------------------------------------------------------------------------------
# CoFormer Official Code
# Copyright (c) Junhyeong Cho. All Rights Reserved 
# Licensed under the Apache License 2.0 [see LICENSE for details]
# ----------------------------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved [see LICENSE for details]
# ----------------------------------------------------------------------------------------------

"""
Run an inference on a custom image 
"""
import argparse
import random
import numpy as np
import torch
import datasets
import util.misc as utils
import cv2
import skimage
import skimage.transform
import nltk
import re
import json
from util import box_ops
from PIL import Image
from torch.utils.data import DataLoader
from datasets import build_dataset
from models import build_model
from pathlib import Path
from nltk.corpus import wordnet as wn
from segment_anything import sam_model_registry, SamPredictor
from mobile_encoder.setup_mobile_sam import setup_model
from detectron2.utils.visualizer import ColorMode, Visualizer

def noun2synset(noun):
    return wn.synset_from_pos_and_offset(noun[0], int(noun[1:])).name() if re.match(r'n[0-9]*', noun) else "'{}'".format(noun)

def visualize_bbox(sam, image_path=None, num_roles=None, noun_labels=None, pred_bbox=None, pred_bbox_conf=None, output_dir=None):
    image = cv2.imread(image_path)
    original = image.copy()
    image_name = image_path.split('/')[-1].split('.')[0]
    h, w = image.shape[0], image.shape[1]
    red_color = (232, 126, 253)
    green_color = (130, 234, 198)
    blue_color = (227,188, 134)
    orange_color = (98, 129, 240)
    brown_color = (79, 99, 216)
    purple_color = (197, 152, 173)
    colors = [red_color, green_color, blue_color, orange_color, brown_color, purple_color]
    white_color = (255, 255, 255)
    line_width = 4
    boxes = []

    # the value of pred_bbox_conf is logit, not probability. 
    for i in range(num_roles):  
        if pred_bbox_conf[i] >= 0:
            # bbox
            pred_left_top = (int(pred_bbox[i][0].item()), int(pred_bbox[i][1].item()))
            pred_right_bottom = (int(pred_bbox[i][2].item()), int(pred_bbox[i][3].item()))
            lt_0 = max(pred_left_top[0], line_width)
            lt_1 = max(pred_left_top[1], line_width)
            rb_0 = min(pred_right_bottom[0], w-line_width)
            rb_1 = min(pred_right_bottom[1], h-line_width)
            lt = (lt_0, lt_1)
            rb = (rb_0, rb_1)
            cv2.rectangle(img=image, pt1=lt, pt2=rb, color=colors[i], thickness=line_width, lineType=-1)   
            boxes.append([lt_0, lt_1, rb_0, rb_1])
            # label
            label = noun_labels[i].split('.')[0]
            lw = max(round((w + h) / 2 * 0.002), 2)
            tf = max(lw - 1, 1)
            label_w, label_h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]
            outside = lt[1] - label_h >= 3
            cv2.rectangle(image, (lt[0], lt[1] - 2 if outside else lt[1] + label_h + 2), (lt[0]+ label_w, lt[1] - 2-label_h if outside else lt[1] + 2), colors[i], -1)
            cv2.putText(image, label, (lt[0], lt[1] - 2 if outside else lt[1] + label_h + 2), 0, lw / 3, (255, 255, 255), thickness=tf, lineType=cv2.LINE_AA)
        else:
            boxes.append([-1,-1,-1,-1])

    boxes = torch.Tensor(boxes).cuda()
    masks = bbx_to_sem(sam, boxes, original)
    v = Visualizer(image, metadata=None, scale=1.0, instance_mode=ColorMode.IMAGE)
    for i in range(num_roles):
        if pred_bbox_conf[i] >= 0:
            mask = masks[i][0]+0
            color = tuple(map(lambda item: item / 255, colors[i]))
            image = v.draw_binary_mask(mask.cpu().numpy(),color=color,edge_color=(1.0, 1.0, 240.0 / 255), alpha=0.8)
    image = image.get_image()
    # save image
    cv2.imwrite("{}/{}_{}.jpg".format(output_dir, image_name, sam), image)

    return 

def bbx_to_sem(sam, bbx, img):
    if sam == 'sam':
        sam_checkpoint = 'ckpt/sam_vit_h_4b8939.pth'
        model_type = 'vit_h'
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    elif sam == 'mobilesam':
        checkpoint = torch.load('ckpt/mobile_sam.pt')
        sam = setup_model()
        sam.load_state_dict(checkpoint,strict=True)
    device = 'cuda'
    sam.to(device=device)
    sam.eval()
    predictor = SamPredictor(sam)
    predictor.set_image(img)
    transformed_boxes = predictor.transform.apply_boxes_torch(bbx, img.shape[:2])
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )   
    return masks

def process_image(image):
    mean = np.array([[[0.485, 0.456, 0.406]]])
    std = np.array([[[0.229, 0.224, 0.225]]])
    image = (image.astype(np.float32) - mean) / std
    min_side, max_side= 512, 700
    rows_orig, cols_orig, cns_orig = image.shape
    smallest_side = min(rows_orig, cols_orig)
    scale = min_side / smallest_side
    largest_side = max(rows_orig, cols_orig)
    if largest_side * scale > max_side:
        scale = max_side / largest_side

    # resize the image with the computed scale
    image = skimage.transform.resize(image, (int(round(rows_orig * scale)), int(round((cols_orig * scale)))))
    rows, cols, cns = image.shape
    new_image = np.zeros((rows, cols, cns)).astype(np.float32)
    new_image[:rows, :cols, :] = image.astype(np.float32)
    image = torch.from_numpy(new_image)

    shift_1 = int((700 - cols) * 0.5)
    shift_0 = int((700 - rows) * 0.5)

    max_height = 700
    max_width = 700
    padded_imgs = torch.zeros(1, max_height, max_width, 3)
    padded_imgs[0, shift_0:shift_0+image.shape[0], shift_1:shift_1+image.shape[1], :] = image
    padded_imgs = padded_imgs.permute(0, 3, 1, 2)
    
    height = torch.tensor(int(image.shape[0])).float()
    width = torch.tensor(int(image.shape[1])).float()
    shift_0 = torch.tensor(shift_0).float()
    shift_1 = torch.tensor(shift_1).float()
    scale = torch.tensor(scale).float()
    mw = torch.tensor(max_width).float()
    mh = torch.tensor(max_height).float()

    return (utils.nested_tensor_from_tensor_list(padded_imgs),
            {'width': width,
            'height': height,
            'shift_0': shift_0,
            'shift_1': shift_1,
            'scale': scale,
            'max_width': mw,
            'max_height': mh})


def inference(model, device, sam, image_path=None, inference=False, idx_to_verb=None, idx_to_role=None, 
              vidx_ridx=None, idx_to_class=None, output_dir=None):
    model.eval()
    image_name = image_path.split('/')[-1].split('.')[0]
    
    # load image & process
    image = Image.open(image_path)
    image = image.convert('RGB')
    image = np.array(image)
    image = image.astype(np.float32) / 255.0
    image, info = process_image(image)
    image = image.to(device)
    info = {k: v.to(device) if type(v) is not str else v for k, v in info.items()}

    output = model(image, inference=inference)
    # print(output)
    pred_verb = output['pred_verb'][0]
    pred_noun = output['pred_noun_3'][0]
    pred_bbox = output['pred_bbox'][0]
    pred_bbox_conf = output['pred_bbox_conf'][0]

    top1_verb = torch.topk(pred_verb, k=1, dim=0)[1].item()
    roles = vidx_ridx[top1_verb]
    num_roles = len(roles)
    verb_label = idx_to_verb[top1_verb]
    role_labels = []
    noun_labels = []
    for i in range(num_roles):
        top1_noun = torch.topk(pred_noun[i], k=1, dim=0)[1].item()
        role_labels.append(idx_to_role[roles[i]])
        noun_labels.append(noun2synset(idx_to_class[top1_noun]))
    
    # convert bbox
    mw, mh = info['max_width'], info['max_height']
    w, h = info['width'], info['height']
    shift_0, shift_1, scale  = info['shift_0'], info['shift_1'], info['scale']
    pb_xyxy = box_ops.swig_box_cxcywh_to_xyxy(pred_bbox.clone(), mw, mh, device=device)
    for i in range(num_roles):
        pb_xyxy[i][0] = max(pb_xyxy[i][0] - shift_1, 0)
        pb_xyxy[i][1] = max(pb_xyxy[i][1] - shift_0, 0)
        pb_xyxy[i][2] = max(pb_xyxy[i][2] - shift_1, 0)
        pb_xyxy[i][3] = max(pb_xyxy[i][3] - shift_0, 0)
        # locate predicted boxes within image (processing w/ image width & height)
        pb_xyxy[i][0] = min(pb_xyxy[i][0], w)
        pb_xyxy[i][1] = min(pb_xyxy[i][1], h)
        pb_xyxy[i][2] = min(pb_xyxy[i][2], w)
        pb_xyxy[i][3] = min(pb_xyxy[i][3], h)
    pb_xyxy /= scale

    # outputs
    json_file_path = 'SWiG/SWiG_jsons/imsitu_space.json'
    with open(json_file_path, 'r') as f:
        abstract = json.load(f)['verbs'][verb_label]['abstract']
        f.close()
    with open("{}/{}_{}.txt".format(output_dir, image_name, sam), "w") as f:
        text_line = "verb: {} \n".format(verb_label)
        f.write(text_line)
        for i in range(num_roles):
            text_line = "role: {}, noun: {} \n".format(role_labels[i], noun_labels[i])
            # print('!!!!!!!!!!!!!!!!!!!!!!!!!', role_labels[i].upper())
            abstract = abstract.replace(role_labels[i].upper(), noun_labels[i].split('.')[0])
            f.write(text_line)
        f.write(abstract)
        f.close()
    visualize_bbox(sam=sam, image_path=image_path, num_roles=num_roles, noun_labels=noun_labels, pred_bbox=pb_xyxy, pred_bbox_conf=pred_bbox_conf, output_dir=output_dir)


def get_args_parser():
    parser = argparse.ArgumentParser('Set CoFormer', add_help=False)

    # Backbone parameters
    parser.add_argument('--backbone', default='swin_T_224_1k', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--position_embedding', default='learned', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # Transformer parameters
    parser.add_argument('--num_glance_enc_layers', default=3, type=int,
                        help="Number of encoding layers in Glance Transformer")
    parser.add_argument('--num_gaze_s1_dec_layers', default=3, type=int,
                        help="Number of decoding layers in Gaze-Step1 Transformer")
    parser.add_argument('--num_gaze_s1_enc_layers', default=3, type=int,
                        help="Number of encoding layers in Gaze-Step1 Transformer")
    parser.add_argument('--num_gaze_s2_dec_layers', default=3, type=int,
                        help="Number of decoding layers in Gaze-Step2 Transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=512, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.15, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")

    # Dataset parameters
    parser.add_argument('--dataset_file', default='swig')
    parser.add_argument('--swig_path', type=str, default="SWiG")
    parser.add_argument('--image_path', default='inference/image.jpg',  
                        help='path where the test image is')

    # Etc...
    parser.add_argument('--inference', default=True)
    parser.add_argument('--output_dir', default='inference',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for inference')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--saved_model', default='ckpt/OpenSU_Swin.pth',
                        help='path where saved model is')
    parser.add_argument('--sam', default='sam', choices=['sam','mobilesam'])
    #Deformabel DETR
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--return_interm_indices', default=[3])
    parser.add_argument('--backbone_freeze_keywords', )
    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print(args)

    if not args.inference:
        assert False, f"Please set inference to True"

    # fix the seed
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # num noun classes in train dataset
    dataset_train = build_dataset(image_set='train', args=args)
    args.num_noun_classes = dataset_train.num_nouns()

    # build model
    device = torch.device(args.device)
    model, _ = build_model(args)
    model.to(device)
    checkpoint = torch.load(args.saved_model, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    inference(model, device, sam=args.sam, image_path=args.image_path, inference=args.inference, 
              idx_to_verb=args.idx_to_verb, idx_to_role=args.idx_to_role, vidx_ridx=args.vidx_ridx, 
              idx_to_class=args.idx_to_class, output_dir=args.output_dir)

    return


if __name__ == '__main__':
    nltk.download('wordnet')
    parser = argparse.ArgumentParser('CoFormer inference script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)