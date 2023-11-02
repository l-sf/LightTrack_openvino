#
# Created by lsf on 2022/7/19.
#

import cv2
import numpy as np
import argparse
from LightTack import LightTrack
from utils import cxy_wh_2_rect
import glob


def parse_args():
    """
    args for fc testing.
    """
    parser = argparse.ArgumentParser(description='LightTrack infer')
    parser.add_argument('--mode', default=2, type=int, help='0:视频文件, 1:摄像头, 2:数据集')
    parser.add_argument('--z_feature_path', default='../models/z_feature.xml', type=str, help='z_model_path')
    parser.add_argument('--x_feature_path', default='../models/x_feature.xml', type=str, help='x_model_path')
    parser.add_argument('--neck_head_path', default='../models/head.xml', type=str, help='head_model_path')
    parser.add_argument('--video', default='../images/bag.avi', type=str, help='video file path')
    parser.add_argument('--image_path', type=str, default='../images/Woman/img/*.jpg', help='dataset image path')
    parser.add_argument('--gt_path', type=str, default='../images/Woman/groundtruth_rect.txt', help='dataset gt path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    tracker = LightTrack(args.z_feature_path, args.x_feature_path, args.neck_head_path)
    display_name = 'LightTrack'

    if args.mode == 0:
        cap = cv2.VideoCapture(args.video)
        success, frame = cap.read()
        if success is not True:
            print("Read failed.")
            exit(-1)
        cv2.imshow(display_name, frame)

        cv2.putText(frame, 'Select target ROI and press ENTER', (20, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                    (255, 0, 0), 1)
        lx, ly, w, h = cv2.selectROI(display_name, frame, fromCenter=False)
        target_pos = np.array([lx + w / 2, ly + h / 2])
        target_sz = np.array([w, h])
        tracker.init(frame, target_pos, target_sz)  # init tracker

        while True:
            ret, frame = cap.read()
            if ret is not True:
                print("Read failed.")
                return
            frame_disp = frame.copy()
            target_pos, target_sz = tracker.track(frame_disp)  # track
            location = cxy_wh_2_rect(target_pos, target_sz)
            x1, y1, x2, y2 = int(location[0]), int(location[1]), int(location[0] + location[2]), \
                             int(location[1] + location[3])
            cv2.rectangle(frame_disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow(display_name, frame_disp)
            cv2.waitKey(1)

    elif args.mode == 1:
        cap = cv2.VideoCapture(0)

        success, frame = cap.read()
        if success is not True:
            print("Read failed.")
            exit(-1)
        cv2.imshow(display_name, frame)

        frame_disp = frame.copy()
        cv2.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    1, (255, 0, 0), 1)
        lx, ly, w, h = cv2.selectROI(display_name, frame_disp, fromCenter=False)
        target_pos = np.array([lx + w / 2, ly + h / 2])
        target_sz = np.array([w, h])
        tracker.init(frame_disp, target_pos, target_sz)  # init tracker

        while True:
            ret, frame = cap.read()
            if ret is not True:
                print("Read failed.")
                return
            frame_disp = frame.copy()
            target_pos, target_sz = tracker.track(frame_disp)  # track
            location = cxy_wh_2_rect(target_pos, target_sz)
            x1, y1, x2, y2 = int(location[0]), int(location[1]), int(location[0] + location[2]), \
                             int(location[1] + location[3])
            cv2.rectangle(frame_disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow(display_name, frame_disp)
            cv2.waitKey(1)

    else:
        image_files = sorted(glob.glob(args.image_path))
        gt = np.loadtxt(args.gt_path, delimiter=',')

        image_file = image_files[0]
        frame = cv2.imread(image_file)
        init_bbox = gt[0]
        lx, ly, w, h = init_bbox
        # lx, ly, w, h = cv2.selectROI(display_name, frame, fromCenter=False)
        target_pos = np.array([lx + w / 2, ly + h / 2])
        target_sz = np.array([w, h])
        tracker.init(frame, target_pos, target_sz)  # init tracker

        for i in range(1, 597):
            frame_disp_name = image_files[i]
            frame_disp = cv2.imread(frame_disp_name)
            target_pos, target_sz = tracker.track(frame_disp)  # track
            location = cxy_wh_2_rect(target_pos, target_sz)
            x1, y1, x2, y2 = int(location[0]), int(location[1]), int(location[0] + location[2]), \
                             int(location[1] + location[3])
            cv2.rectangle(frame_disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow(display_name, frame_disp)
            cv2.waitKey(1)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
