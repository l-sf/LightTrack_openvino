//
// Created by lsf on 2022/7/21.
//

#ifndef LIGHTTRACK_LightTrack_HPP
#define LIGHTTRACK_LightTrack_HPP

#include <cstdlib>
#include <iostream>
#include <string>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <inference_engine.hpp>
#include <openvino/openvino.hpp>


class LightTrack {
public:
    LightTrack(const std::string &z_path, const std::string &x_path, const std::string &head_path);

    /**
     * @brief 初始化跟踪器
     *
     * @param z_img 跟踪的第一帧图像
     * @param init_bbox 第一帧图像上的目标框
     */
    void init(cv::Mat &z_img, cv::Rect &init_bbox);

    /**
     * @brief 跟踪器跟踪
     *
     * @param x_img 当前跟踪的图像
     * @return cv::Rect 返回当前跟踪图像上目标框
     */
    cv::Rect track(cv::Mat &x_img);

    /**
    * @brief 从文件读真值
    *
    * @param gt_file 真值文件的路径
    * @return cv::Rect 返回当前跟踪图像上真实目标框
    */
    cv::Rect read_gt(std::string &gt_file);

private:
    int exemplar_size_ = 128; // 模板图像块大小
    int instance_size_ = 256; // 搜索图像块大小
    int total_stride_ = 16; // 最后特征图的总stride
    float penalty_k_ = 0.062; // 惩罚系数
    float window_influence_ = 0.15; // hanning窗权重系数
    float lr_ = 0.765; // 目标框更新的学习率
    int score_size_ = 16; // 最后特征图的大小
    cv::Rect target_bbox_; // 目标框

    ov::InferRequest z_infer_request_;
    ov::InferRequest x_infer_request_;
    ov::InferRequest head_infer_request_;
    ov::Tensor zf_;
    ov::Tensor xf_;
    ov::Tensor cls_score_;
    ov::Tensor bbox_pred_;

    Eigen::Matrix<float, 16, 16> hanning_win_;
    Eigen::Matrix<float, 16, 16> grid_to_search_x_;
    Eigen::Matrix<float, 16, 16> grid_to_search_y_;
    Eigen::Matrix<float, 16, 16> pred_score_;
    Eigen::Matrix<float, 16, 16> pred_x1_;
    Eigen::Matrix<float, 16, 16> pred_x2_;
    Eigen::Matrix<float, 16, 16> pred_y1_;
    Eigen::Matrix<float, 16, 16> pred_y2_;
    Eigen::Matrix<float, 16, 16> s_c_;
    Eigen::Matrix<float, 16, 16> r_c_;
    Eigen::Matrix<float, 16, 16> penalty_;
    Eigen::Matrix<float, 16, 16> pscore_;

    /**
     * @brief 跟踪器更新
     *
     * @param x_crop 当前跟踪的图像的剪切图像块
     * @param target_sz 上一帧目标框按照模板图像缩放比例缩放之后的大小
     * @param scale_z 模板图像缩放比例
     * @return
     */
    void update(cv::Mat &x_crop, cv::Size_<float> target_sz, float scale_z);

    /**
     * @brief 以目标框为中心剪裁图像，SiamFC type cropping
     *
     * @param img 原图像
     * @param model_sz 需要剪裁图像的大小，模板图像是128，搜索图像是256
     * @param original_sz 不缩放时的目标框加上pad的大小
     * @param avg_chans 图像各通道的均值
     * @return cv::Mat 返回剪裁后的图像
     */
    cv::Mat get_subwindow_tracking(cv::Mat &img, int model_sz, float original_sz, const cv::Scalar &avg_chans);

    /**
     * @brief 产生hanning窗
     */
    void gen_window();

    /**
     * @brief 产生搜索网格
     */
    void gen_grids();
};


#endif //LIGHTTRACK_LightTrack_HPP
