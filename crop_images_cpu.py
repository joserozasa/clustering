from PIL import Image
import cv2
import numpy as np
import json
import os
import shutil
import sys


def find_objects(image):
    gray = cv2.cvtColor(image[:, :, ::-1], cv2.COLOR_BGR2GRAY)

    params = cv2.SimpleBlobDetector_Params()
    detector = cv2.SimpleBlobDetector_create(params)

    keypoints = detector.detect(gray)
    margin = 0

    boxes = []
    threshold = 0.35

    for kp in keypoints:
        xc = kp.pt[0]
        yc = kp.pt[1]
        size = kp.size
        top = xc - (size + margin)
        left = yc - (size + margin)
        bottom = xc + (size + margin)
        right = yc + (size + margin)
        box = [top, left, bottom, right]
        boxes.append(box)

    non_m_s(boxes, threshold)
    return boxes


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
    min_area = 0
    mask = (areas >= min_area) & (areas <= max_area)
    filtered_tensor = pred_boxes_[mask]
    filtered_scores = scores_[mask]
    return filtered_tensor, filtered_scores


def iou(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    x2 = x2 + w2
    y2 = y2 + h2

    intersection = (max(0, min(x1 + w1, x2) - max(x1, bbox2[0])) *
                    max(0, min(y1 + h1, y2) - max(y1, bbox2[1])))
    union = (w1 * h1) + (w2 * h2) - intersection

    iou = intersection / union

    return iou


def non_m_s(bboxes, threshold):
    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            bbox1 = bboxes[i]
            bbox2 = bboxes[j]

            iou_value = iou(bbox1, bbox2)

            if iou_value > threshold:
                bboxes.pop(j)
                break

def nms(bboxes, threshold):
    bboxes = sorted(bboxes, key=lambda x: x[4], reverse=True)

    # Lista para almacenar cuadros delimitadores después de aplicar NMS
    nms_bboxes = []

    while len(bboxes) > 0:
        # Agregar el cuadro delimitador con la mayor confianza a la lista de resultados
        max_bbox = bboxes.pop(0)
        nms_bboxes.append(max_bbox)

        # Calcular el área de intersección de la caja de límites máxima con las demás cajas de límites
        areas = []
        for bbox in bboxes:
            intersection_xmin = max(max_bbox[0], bbox[0])
            intersection_ymin = max(max_bbox[1], bbox[1])
            intersection_xmax = min(max_bbox[2], bbox[2])
            intersection_ymax = min(max_bbox[3], bbox[3])
            intersection_area = max(0, intersection_xmax - intersection_xmin) * max(0, intersection_ymax - intersection_ymin)
            areas.append(intersection_area)

        # Eliminar cajas de límites con una intersección de área superior a un umbral de solapamiento
        keep = []
        for i, bbox in enumerate(bboxes):
            overlap = areas[i] / ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) + (max_bbox[2] - max_bbox[0]) * (max_bbox[3] - max_bbox[1]) - areas[i])
            if overlap < threshold:
                keep.append(i)
 
        bboxes = [bboxes[i] for i in keep]

    return nms_bboxes


def pred_boxes(list_filename, boxes_filename):
    with open(list_filename, "r") as fr:
        for img_url in fr:
            im_np = enhanced_image_as_np_array(img_url.strip())
            boxes = find_objects(im_np)
            for idx, b in enumerate(boxes):
                with open(boxes_filename, "a") as fw:
                    fw.write(f"{img_url.strip()}; {str(boxes[idx])}\n")


def crop_image(boxes_filename, recorte_path):
    with open(boxes_filename, 'r') as f:
        lines = f.readlines()
    os.mkdir(recorte_path, 0o777)
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
