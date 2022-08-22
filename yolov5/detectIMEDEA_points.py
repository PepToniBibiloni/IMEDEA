import argparse
from copy import deepcopy
import os
import sys
from pathlib import Path
from turtle import left
import statistics
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to leftPATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2, increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

focal_length = 22
distance_between_Camera = 15.0

def load_stereo_coefficients(path):
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    K1 = cv_file.getNode("K1").mat()
    D1 = cv_file.getNode("D1").mat()
    K2 = cv_file.getNode("K2").mat()
    D2 = cv_file.getNode("D2").mat()
    R = cv_file.getNode("R").mat()
    T = cv_file.getNode("T").mat()
    E = cv_file.getNode("E").mat()
    F = cv_file.getNode("F").mat()
    R1 = cv_file.getNode("R1").mat()
    R2 = cv_file.getNode("R2").mat()
    P1 = cv_file.getNode("P1").mat()
    P2 = cv_file.getNode("P2").mat()
    Q = cv_file.getNode("Q").mat()
    cv_file.release()
    return [K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q]

def matching(left_img, right_img): 
    global kp_1,kp_2,matches
    sift = cv2.xfeatures2d.SIFT_create()
    kp_1, desc_1 = sift.detectAndCompute(left_img, None)
    kp_2, desc_2 = sift.detectAndCompute(right_img, None)
    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc_1, desc_2, k=2)

def is_in(bb_cord, x, y): 
	bb_cord = list(map(float, bb_cord))
	x = float(x)
	y = float(y)
	if(x>=bb_cord[0] and x<=bb_cord[2] and y>=bb_cord[1] and y<=bb_cord[3]):
		return True
	else:
		return False

def detectar(left_img,right_img,bb_cord):
    points = []
    for m, n in matches:
        if m.distance < 0.7*n.distance:
            if(is_in(bb_cord, kp_1[m.queryIdx].pt[0], kp_1[m.queryIdx].pt[1])):
                points.append(m)
    #result = cv2.drawMatches(left_img, kp_1, right_img, kp_2, points, None) 
    #cv2.imshow("img",result)
    #cv2.waitKey(0)
    ret = [(kp_1[m.queryIdx].pt[0] - kp_2[m.trainIdx].pt[0]) for m in points]
    return ret


@torch.no_grad()
def run(
        weights=ROOT / 'yolov5s.pt',  # model.pt leftPath(s)
        left_source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        right_source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml leftPath
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        calib_file=''
):
    leftSource = str(left_source)
    rightSource = str(right_source)
    save_img = not nosave and not leftSource.endswith('.txt') and not rightSource.endswith('.txt') # save inference images
    left_is_file = Path(leftSource).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    right_is_file = Path(rightSource).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    left_is_url = leftSource.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    right_is_url = leftSource.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    left_webcam = leftSource.isnumeric() or leftSource.endswith('.txt') or (left_is_url and not left_is_file)
    right_webcam = rightSource.isnumeric() or rightSource.endswith('.txt') or (right_is_url and not right_is_file)
    if left_is_url and left_is_file:
        leftSource = check_file(leftSource)  # download
    if right_is_url and right_is_file:
        rightSource = check_file(rightSource)  # download
    if left_webcam != right_webcam:
        sys.exit()

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if left_webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        datasetl = LoadStreams(leftSource, img_size=imgsz, stride=stride, auto=pt)
        datasetr = LoadStreams(rightSource, img_size=imgsz, stride=stride, auto=pt)
        bs = len(datasetl)  # batch_size
    else:
        datasetl = LoadImages(leftSource, img_size=imgsz, stride=stride, auto=pt)
        datasetr = LoadImages(rightSource, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs
    
    # Calibration
    K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q = load_stereo_coefficients(calib_file)  # Get cams params
    leftMapX, leftMapY = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (1280, 720), cv2.CV_32F)
    rightMapX, rightMapY = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (1280, 720), cv2.CV_32F)
	
    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    
    for leftIMG,rightIMG in zip(datasetl,datasetr):
        leftPath, leftIm, leftIm0s, left_vid_cap, s = leftIMG
        rightPath, rightIm, rightIm0s, right_vid_cap, s = rightIMG
        
        t1 = time_sync()
        leftIm = torch.from_numpy(leftIm).to(device)
        leftIm = leftIm.half() if model.fp16 else leftIm.float()  # uint8 to fp16/32
        leftIm /= 255  # 0 - 255 to 0.0 - 1.0
        if len(leftIm.shape) == 3:
            leftIm = leftIm[None]  # expand for batch dim

        t2 = time_sync()
        dt[0] += t2 - t1
        #leftIm0s = cv2.remap(leftIm0s, leftMapX, leftMapY, cv2.INTER_LINEAR)
        #rightIm0s = cv2.remap(rightIm0s, rightMapX, rightMapY, cv2.INTER_LINEAR)
        # Matching
        matching(leftIm0s,rightIm0s)
        # Inference
        visualize = increment_path(save_dir / Path(leftPath).stem, mkdir=True) if visualize else False
        pred = model(leftIm, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2
        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if left_webcam:  # batch_size >= 1
                p, im0, frame = leftPath[i], leftIm0s[i].copy(), datasetl.count
                s += f'{i}: '
            else:
                p, im0, frame = leftPath, leftIm0s.copy(), getattr(datasetl, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path= str(save_dir / 'labels' / p.stem) + ('' if datasetl.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % leftIm.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(leftIm.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                
                # Write results
                for *xyxy, conf, cls in reversed(det):    
                    diff_arr =  detectar(leftIm0s,rightIm0s,xyxy)
                    if diff_arr!=[]:
                        med = statistics.median(diff_arr)+1
                        distance = (focal_length*distance_between_Camera)/med
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(f'{txt_path}.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')
                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (f'{names[c]}{distance:.1f}' if hide_conf else f'{names[c]} {conf:.2f} {distance:.1}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
            # Stream results
            im0 = annotator.result() 
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(0)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if datasetl.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if left_vid_cap:  # video
                            fps = left_vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(left_vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(left_vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--left-source', type=str, required=True, help='Left source path')
    parser.add_argument('--right-source', type=str, required=True, help='Right source path')
    parser.add_argument('--calib-file', type=str, required=True, help='Calib file path')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
