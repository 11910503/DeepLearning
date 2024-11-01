import argparse
import os
import platform
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
from PIL import Image
import numpy as np

import torch
from models.common import DetectMultiBackend
from yoloutils.augmentations import (Albumentations, augment_hsv, classify_albumentations, classify_transforms, copy_paste,
                                     letterbox, mixup, random_perspective)
from yoloutils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from yoloutils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                               increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from yoloutils.plots import Annotator, colors, save_one_box
from yoloutils.torch_utils import select_device, smart_inference_mode
import torch.nn.functional as F
class predicter:
    def __init__(self,weight):
        device = select_device('')
        model = DetectMultiBackend(weight, device=device, dnn=False, data='data/coco128.yaml', fp16=False)
        stride, names, pt = model.stride, model.names, model.pt
        self.model=model
        self.stride=stride
        self.names=names
        self.pt=pt
        self.bs=1
        self.imgsz = check_img_size((640, 640), s=self.stride)  # check image size
        self.model.warmup(imgsz=(1 if self.pt or self.model.triton else self.bs, 3, *self.imgsz))  # warmup
        print("model ready")
    def run(self,pic,save_path):

        dataset = LoadImages(pic, img_size=self.imgsz, stride=self.stride, auto=self.pt, vid_stride=1)

        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        for path, im, im0s, vid_cap, s in dataset:
            print(dt)
            with dt[0]:
                print(1)
                im = torch.from_numpy(im).to(self.model.device)
                im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                print(2)
                pred = self.model(im, augment=False, visualize=False)

            # NMS
            with dt[2]:
                print(3)
                pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)

            # Second-stage classifier (optional)
            # pred = yoloutils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1

                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                annotator = Annotator(im0, line_width=3, example=str(self.names))

                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        line = (cls, *xywh, conf)
                        c = int(cls)  # integer class
                        label = f'{self.names[c]}'

                        annotator.box_label(xyxy, label, color=colors(c, True))

                        print(('%g ' * len(line)).rstrip() % line + '\n')

                # Stream results
                im0=annotator.result()
                cv2.imwrite(save_path, im0)
                return im0
    def runtest(self,pic):
        im0s=pic.copy()
        im = letterbox(im0s, self.imgsz, stride=self.stride, auto=True)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(self.model.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        pred = self.model(im, augment=False, visualize=False)
        pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)

        for i, det in enumerate(pred):  # per image
            im0 = im0s.copy()
            annotator = Annotator(im0, line_width=3, example=str(self.names))

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = f'{self.names[c]}'

                    annotator.box_label(xyxy, label, color=colors(c, True))

            # Stream results
            im0 = annotator.result()
            return im0
    def runtest2(self,pic):
        im0s=pic.copy()
        im = letterbox(im0s, self.imgsz, stride=self.stride, auto=True)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(self.model.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        pred = self.model(im, augment=False, visualize=False)
        pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)
        return  pred
    def runlabel(self,pic):
        im0s = pic.copy()
        im = letterbox(im0s, self.imgsz, stride=self.stride, auto=True)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(self.model.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        pred = self.model(im, augment=False, visualize=False)
        pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)
        ans=[]
        for i, det in enumerate(pred):  # per image

            gn = torch.tensor(pic.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], pic.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    line = (cls, *xywh)
                    ans.append(line)
        return ans

class predicter_cls:
    def __init__(self,weight):
        device = select_device('')
        model = DetectMultiBackend(weight, device=device, dnn=False, data='data/coco128.yaml', fp16=False)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size((224,224), s=stride)  # check image size
        self.model = model
        self.stride = stride
        self.names = names
        self.pt = pt
        self.bs = 1
        self.imgsz = check_img_size((224, 224), s=self.stride)
        self.transforms = classify_transforms(self.imgsz[0])
          # check image size
        self.model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    def run(self,pic):
        im0s = pic.copy()
        im=self.transforms(im0s)
        # im = letterbox(im0s, self.imgsz, stride=self.stride, auto=True)[0]  # padded resize
        # im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        # im = np.ascontiguousarray(im)
        im = torch.Tensor(im).to(self.model.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        results = self.model(im)

        pred = F.softmax(results, dim=1)
        return pred




