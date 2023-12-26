import torch
from datasets.coco import make_coco_transforms
from PIL import Image
from models import build_model
from argparse import Namespace
import cv2
from torchvision.ops import nms
from util.box_ops import box_cxcywh_to_xyxy
import numpy as np
import json
import os
import shutil
import sys

args = {'model': 'deformable_detr', 'lr': 0.0002, 'max_prop': 30, 'lr_backbone_names': ['backbone.0'], 'lr_backbone': 2e-05, 'lr_linear_proj_names': ['reference_points', 'sampling_offsets'], 'lr_linear_proj_mult': 0.1, 'batch_size': 4, 'weight_decay': 0.0001, 'epochs': 50, 'lr_drop': 40, 'lr_drop_epochs': None, 'clip_max_norm': 0.1, 'sgd': False, 'filter_pct': -1, 'with_box_refine': False, 'two_stage': False, 'strategy': 'topk', 'obj_embedding_head': 'intermediate', 'frozen_weights': None, 'backbone': 'resnet50', 'dilation': False, 'position_embedding': 'sine', 'position_embedding_scale': 6.283185307179586, 'num_feature_levels': 4, 'enc_layers': 6, 'dec_layers': 6, 'dim_feedforward': 1024, 'hidden_dim': 256, 'dropout': 0.1, 'nheads': 8, 'num_queries': 300, 'dec_n_points': 4, 'enc_n_points': 4, 'pretrain': '', 'load_backbone': 'swav', 'masks': False, 'aux_loss': True, 'set_cost_class': 2, 'set_cost_bbox': 5, 'set_cost_giou': 2, 'object_embedding_loss_coeff': 1, 'mask_loss_coef': 1, 'dice_loss_coef': 1, 'cls_loss_coef': 2, 'bbox_loss_coef': 5, 'giou_loss_coef': 2, 'focal_alpha': 0.25, 'dataset_file': 'coco', 'dataset': 'imagenet', 'data_root': 'data', 'coco_panoptic_path': None, 'remove_difficult': False, 'output_dir': '', 'cache_path': 'cache/ilsvrc/ss_box_cache', 'device': 'cuda', 'seed': 42, 'resume': '', 'eval_every': 1, 'start_epoch': 0, 'eval': False, 'viz': False, 'num_workers': 2, 'cache_mode': False, 'object_embedding_loss': False}
args = Namespace(**args)
model, criterion, postprocessors = build_model(args)
model.cuda()
checkpoint = torch.hub.load_state_dict_from_url("https://github.com/amirbar/DETReg/releases/download/1.0.0/checkpoint_imagenet.pth", progress=True, map_location=torch.device('cuda'))
load_msg = model.load_state_dict(checkpoint['model'], strict=False)
transforms = make_coco_transforms('val')


def find_objects(im_np):

    div_M = 2
    div_N = 3

    M = im_np.shape[0]//div_M
    N = im_np.shape[1]//div_N
    final_boxes = []
    final_scores = []

    for m in range(div_M):
        for n in range(div_N):
            top_left = (m*M, n*N)
            tile = im_np[top_left[0]:top_left[0]+M, top_left[1]:top_left[1]+N]
            imfa = Image.fromarray(tile)
            im_t, _ = transforms(imfa, None)
            im_t = im_t.unsqueeze(0)
            res = model(im_t.cuda())
            scores = torch.sigmoid(res['pred_logits'][..., 1])
            pred_boxes = res['pred_boxes']
            img_w, img_h = imfa.size
            max_area = 0.2*img_h*img_h
            pred_boxes_ = box_cxcywh_to_xyxy(pred_boxes) * torch.Tensor([img_w, img_h, img_w, img_h]).cuda()
            I = scores.argsort(descending=True)  # sort by model confidence
            pred_boxes_ = pred_boxes_[0, I[0, :5]]  # pick top 5 proposals
            scores_ = scores[0, I[0, :5]]
            filt_boxes, filt_scores = delete_area(pred_boxes_, scores_, max_area)
            index = non_m_s(filt_boxes, filt_scores, 0.35)
            filt_boxes_in_entire_image = filt_boxes[index]
            filt_boxes_in_entire_image[:, [0, 2]] += top_left[1]
            filt_boxes_in_entire_image[:, [1, 3]] += top_left[0]
            filt_scores_in_entire_image = filt_scores[index]
            final_boxes += filt_boxes_in_entire_image.tolist()
            final_scores += filt_scores_in_entire_image.tolist()
    return final_boxes, final_scores


def enhanced_image_as_np_array(img_url):
    im = Image.open(img_url)
    im = image_contrast(im)
    im_np = np.asarray(im)
    return im_np


def image_contrast(img):
    img = np.asarray(img)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(20, 20))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    enhanced_img = Image.fromarray(enhanced_img)
    return enhanced_img


def delete_area(pred_boxes_, scores_, max_area):
    areas = (pred_boxes_[:, 2] - pred_boxes_[:, 0]) * (pred_boxes_[:, 3] - pred_boxes_[:, 1])
    min_area = 0  # Establecer el área mínima que deseas filtrar
    mask = (areas >= min_area) & (areas <= max_area)
    filtered_tensor = pred_boxes_[mask]
    filtered_scores = scores_[mask]
    return filtered_tensor, filtered_scores


def non_m_s(pred_boxes_, scores_, IoU):
    bboxes = nms(pred_boxes_, scores_, IoU)
    index = bboxes.detach().cpu().numpy()
    return index


def pred_boxes(list_filename, boxes_filename):
    with open(list_filename, "r") as fr:
        for img_url in fr:
            im_np = enhanced_image_as_np_array(img_url.strip())
            boxes, scores = find_objects(im_np)
            for idx, b in enumerate(boxes):
                with open(boxes_filename, "a") as fw:
                    fw.write(f"{img_url.strip()}; {str(boxes[idx])}; {str(scores[idx])}\n")


def crop_image(boxes_filename, recorte_path):
    with open(boxes_filename) as f:
        lines = f.readlines()
    os.mkdir(recorte_path)
    for line in lines:
        img_path = line.split(';')[0]
        img_name = img_path.split('/')[-1].split('.')[0]
        coord_str = line.split(';')[1]

        coord_list = json.loads(coord_str)
        img = cv2.imread(img_path)
        nombre_img = recorte_path + '/' + img_name + '_' + str(int(max(0, coord_list[1]))) + '_' + str(int(min(coord_list[3], img.shape[0] - 1))) + '_' + str(int(max(0, coord_list[0]))) + '_' + str(int(min(coord_list[2], img.shape[1] - 1))) + '.jpg'

        cropped_image = img[int(max(0, coord_list[1])):int(min(coord_list[3], img.shape[0] - 1)), int(max(0, coord_list[0])):int(min(coord_list[2], img.shape[1] - 1))]
        cv2.imwrite(nombre_img, cropped_image)


def main(image_dir):
    img_files = []
    list_filename = "img_filelist.txt"
    boxes_filename = "boxes_list.txt"
    if image_dir[-1] == "/":
        image_dir = image_dir[:-1]
    crops_dir = f"{image_dir}/crops"

    for file in os.listdir(image_dir):
        if file.endswith((".jpg", ".jpeg")):
            img_files.append(image_dir + "/" + file)

    with open(list_filename, "w") as file_txt:
        for file in img_files:
            file_txt.write(file + "\n")

    if os.path.exists(crops_dir):
        shutil.rmtree(crops_dir)

    pred_boxes(list_filename, boxes_filename)
    crop_image(boxes_filename, crops_dir)


if __name__ == "__main__":
    image_dir = sys.argv[1]
    main(image_dir)
