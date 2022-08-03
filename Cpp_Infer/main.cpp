//
// Created by lsf on 2022/7/21.
//

#include <iostream>
#include <boost/format.hpp>
#include "Tracker.hpp"

using namespace std;


int main(int argc, char* argv[]){
//    string video_path = "../../images/bag.avi";
//    string image_path = "../../images/Woman/img/%04d.jpg";
    string gt_path = "../../images/Woman/groundtruth_rect.txt";

    string z_path = "../../models/z_feature.xml";
    string x_path = "../../models/x_feature.xml";
    string head_path = "../../models/head.xml";

    // 读参数
    if (argc != 3)
    {
        fprintf(stderr, "usage: %s [mode] [path]. \n For video, mode=0, path=/xxx/xxx/*.mp4; \n For webcam mode=1, path is cam id; \n For image dataset, mode=2, path=xxx/xxx/%04d.jpg; \n", argv[0]);
        return -1;
    }
    // MODE=0: 视频文件   MODE=1: 摄像头   MODE=2: 数据集
    int MODE = atoi(argv[1]);
    string path = argv[2];

    LightTrack tracker = LightTrack(z_path, x_path, head_path);
    cv::VideoCapture cap;
    string display_name = "LightTack";

    if (MODE == 0) {
        cv::Mat frame;
        cap.open(path);
        cap >> frame;
        cv::imshow(display_name, frame);
        cv::putText(frame, "Select target ROI and press ENTER", cv::Point2i(20, 30),
                    cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255,0,0), 1);
        cv::Rect init_bbox = cv::selectROI(display_name, frame);
        tracker.init(frame, init_bbox);
        cv::Rect bbox;
        cv::Mat img;
        while (true) {
            bool ret = cap.read(img);
            if (!ret) {
                cout << "----------Read failed!!!----------" << endl;
                return 0;
            }

            bbox = tracker.track(img);

            cv::rectangle(img, bbox, cv::Scalar(0,255,0), 2);
            cv::imshow(display_name, img);
            cv::waitKey(1);
        }
    } else if (MODE == 1) {
        cv::Mat frame;
        cap.open(path);
        cap >> frame;
        cv::imshow(display_name, frame);
        cv::putText(frame, "Select target ROI and press ENTER", cv::Point2i(20, 30),
                    cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255,0,0), 1);
        cv::Rect init_bbox = cv::selectROI(display_name, frame);
        tracker.init(frame, init_bbox);
        cv::Rect bbox;
        cv::Mat img;
        while (true) {
            bool ret = cap.read(img);
            if (!ret) {
                cout << "----------Read failed!!!----------" << endl;
                return 0;
            }
            
            bbox = tracker.track(img);
            
            cv::rectangle(img, bbox, cv::Scalar(0,255,0), 2);
            cv::imshow(display_name, img);
            cv::waitKey(1);
        }
    } else if (MODE == 2) {
        boost::format fmt(path.data());  //数据集图片
        cv::Mat frame;
        frame = cv::imread((fmt % 1).str(),1);
        cv::imshow(display_name, frame);
        // cv::putText(frame, "Select target ROI and press ENTER", cv::Point2i(20, 30),
        //             cv::FONT_HERSHEY_COMPLEX_SMALL, 1.5, cv::Scalar(255,0,0), 1);
        // cv::Rect init_bbox = cv::selectROI(display_name, frame);
        cv::Rect init_bbox = tracker.read_gt(gt_path);
        tracker.init(frame, init_bbox);
        cv::Rect bbox;
        cv::Mat img;
        for (int i = 1; i < 597; ++i) {
            img = cv::imread((fmt % i).str(),1);

            bbox = tracker.track(img);
            
            cv::rectangle(img, bbox, cv::Scalar(0,255,0), 2);
            cv::imshow(display_name, img);
            cv::waitKey(1);
        }
        return 0;

    } else {
        printf("MODE错误，0：视频文件；1：摄像头；2：数据集");
        return -1;
    }

}
