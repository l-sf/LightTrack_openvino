//
// Created by lsf on 2022/7/21.
//

#include "Tracker.hpp"

Eigen::VectorXf calc_hanning(int m, int n) {
    Eigen::VectorXf w, w1, w2, w3;
    w1.setLinSpaced(m, 0, m - 1);
    w2 = w1 * (2 * M_PI / (n - 1));
    w3 = 0.5 * w2.array().cos();
    w = 0.5 - w3.array();
    return w;
    //w = 0.5*(1 - cos(2 * PI*w1/(n+1)));
}

Eigen::MatrixXf change(const Eigen::MatrixXf &r) {
    return r.cwiseMax(r.cwiseInverse());
}

Eigen::MatrixXf sz(const Eigen::MatrixXf &w, const Eigen::MatrixXf &h) {
    Eigen::MatrixXf pad = (w + h) * 0.5;
    Eigen::MatrixXf sz2 = (w + pad).cwiseProduct(h + pad);
    return sz2.cwiseSqrt();
}

float sz(const float w, const float h) {
    float pad = (w + h) * 0.5;
    float sz2 = (w + pad) * (h + pad);
    return std::sqrt(sz2);
}

Eigen::MatrixXf mxexp(Eigen::MatrixXf mx) {
    for (int i = 0; i < mx.rows(); ++i) {
        for (int j = 0; j < mx.cols(); ++j) {
            mx(i, j) = std::exp(mx(i, j));
        }
    }
    return mx;
}

LightTrack::LightTrack(const std::string &z_path, const std::string &x_path, const std::string &head_path) {
    // 1.创建OpenVINO Runtime Core对象
    ov::Core core;
    // 2.载入并编译模型
    ov::CompiledModel z_compile_model = core.compile_model(z_path, "CPU");
    ov::CompiledModel x_compile_model = core.compile_model(x_path, "CPU");
    ov::CompiledModel head_compile_model = core.compile_model(head_path, "CPU");
    // 3.创建推理请求
    z_infer_request_ = z_compile_model.create_infer_request();
    x_infer_request_ = x_compile_model.create_infer_request();
    head_infer_request_ = head_compile_model.create_infer_request();
}

void LightTrack::init(cv::Mat &z_img, cv::Rect &init_bbox) {
    target_bbox_ = init_bbox;
    cv::Scalar mean = cv::mean(z_img);
    float wc_z = target_bbox_.width + 0.5 * (target_bbox_.width + target_bbox_.height);
    float hc_z = target_bbox_.height + 0.5 * (target_bbox_.width + target_bbox_.height);
    float s_z = std::round(std::sqrt(wc_z * hc_z));  // (s_z)^2=(w+2p)x(h+2p), 模板图像上 不缩放时的 框加上pad 的大小
    cv::Mat z_crop = get_subwindow_tracking(z_img, exemplar_size_, s_z, mean);
    // openvino推理部分
    ov::element::Type input_type = ov::element::u8;
    ov::Shape input_shape = {1, 128, 128, 3};
    // 使用ov::Tensor包装图像数据，无需分配新内存
    ov::Tensor input_tensor = ov::Tensor(input_type, input_shape, z_crop.data);
    z_infer_request_.set_input_tensor(input_tensor);
    z_infer_request_.infer();
    // 得到模板图像的特征张量
    zf_ = z_infer_request_.get_output_tensor();
    // 产生hanning窗，以及搜索图像上的grids
    gen_window();
    gen_grids();
}

cv::Rect LightTrack::track(cv::Mat &x_img) {
    cv::Scalar mean = cv::mean(x_img);
    float wc_z = target_bbox_.width + 0.5 * (target_bbox_.width + target_bbox_.height);
    float hc_z = target_bbox_.height + 0.5 * (target_bbox_.width + target_bbox_.height);
    float s_z = std::round(std::sqrt(wc_z * hc_z));  // (s_z)^2=(w+2p)x(h+2p), 模板图像上 不缩放时的 框加上pad 的大小
    float scale_z = float(exemplar_size_) / float(s_z);
    float d_search = float(instance_size_ - exemplar_size_) / 2.f;
    float pad = d_search / scale_z;
    float s_x = s_z + 2 * pad;
    cv::Mat x_crop = get_subwindow_tracking(x_img, instance_size_, s_x, mean);

    auto start = std::chrono::steady_clock::now();
    update(x_crop, cv::Size(target_bbox_.width * scale_z, target_bbox_.height * scale_z), scale_z);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    double time = 1000 * elapsed.count();
    printf("preprocess+inference+postprocess time: %f ms\n", time);

    target_bbox_.x = std::max(0, std::min(x_img.cols, target_bbox_.x));
    target_bbox_.y = std::max(0, std::min(x_img.rows, target_bbox_.y));
    target_bbox_.width = std::max(10, std::min(x_img.cols, target_bbox_.width));
    target_bbox_.height = std::max(10, std::min(x_img.rows, target_bbox_.height));

    return target_bbox_;
}


void LightTrack::update(cv::Mat &x_crop, cv::Size_<float> target_sz, float scale_z) {
    // openvino推理部分
    ov::element::Type input_type = ov::element::u8;
    ov::Shape x_input_shape = {1, 256, 256, 3};
    // 使用ov::Tensor包装图像数据，无需分配新内存
    ov::Tensor x_input_tensor = ov::Tensor(input_type, x_input_shape, x_crop.data);
    x_infer_request_.set_input_tensor(x_input_tensor);
    x_infer_request_.infer();
    // 得到搜索图像的特征张量
    xf_ = x_infer_request_.get_output_tensor();

    head_infer_request_.set_input_tensor(0, zf_);
    head_infer_request_.set_input_tensor(1, xf_);
    head_infer_request_.infer();
    cls_score_ = head_infer_request_.get_output_tensor(0);
    bbox_pred_ = head_infer_request_.get_output_tensor(1);

    const auto *cls_ptr = cls_score_.data<const float>();
    const auto *bbox_ptr = bbox_pred_.data<const float>();

    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 16; ++j) {
            for (int k = 0; k < 16; ++k) {
                if (i == 0)
                    pred_x1_(j, k) = static_cast<float>(bbox_ptr[i * 256 + j * 16 + k]);
                else if (i == 1)
                    pred_y1_(j, k) = static_cast<float>(bbox_ptr[i * 256 + j * 16 + k]);
                else if (i == 2)
                    pred_x2_(j, k) = static_cast<float>(bbox_ptr[i * 256 + j * 16 + k]);
                else if (i == 3)
                    pred_y2_(j, k) = static_cast<float>(bbox_ptr[i * 256 + j * 16 + k]);
                else
                    pred_score_(j, k) = static_cast<float>(cls_ptr[j * 16 + k]);
            }
        }
    }

    pred_x1_ = grid_to_search_x_ - pred_x1_;
    pred_y1_ = grid_to_search_y_ - pred_y1_;
    pred_x2_ = grid_to_search_x_ + pred_x2_;
    pred_y2_ = grid_to_search_y_ + pred_y2_;

    // size penalty
    s_c_ = change(
            sz(pred_x2_ - pred_x1_, pred_y2_ - pred_y1_) / sz(target_sz.width, target_sz.height));  // scale penalty
    r_c_ = change((target_sz.width / target_sz.height) /
                  ((pred_x2_ - pred_x1_).array() / (pred_y2_ - pred_y1_).array()).array());  // ratio penalty

    penalty_ = mxexp(-(r_c_.cwiseProduct(s_c_) - Eigen::MatrixXf::Ones(16, 16)) * penalty_k_);
    pscore_ = penalty_.cwiseProduct(pred_score_);

    // window penalty
    pscore_ = pscore_ * (1 - window_influence_) + hanning_win_ * window_influence_;

    // get max
    Eigen::MatrixXd::Index maxRow, maxCol;
    pscore_.maxCoeff(&maxRow, &maxCol);

    // to real size
    float x1 = pred_x1_(maxRow, maxCol);
    float y1 = pred_y1_(maxRow, maxCol);
    float x2 = pred_x2_(maxRow, maxCol);
    float y2 = pred_y2_(maxRow, maxCol);

    float pred_xs = (x1 + x2) / 2.f;
    float pred_ys = (y1 + y2) / 2.f;
    float pred_w = x2 - x1;
    float pred_h = y2 - y1;

    float diff_xs = pred_xs - float(instance_size_) / 2.f;
    float diff_ys = pred_ys - float(instance_size_) / 2.f;

    diff_xs = diff_xs / scale_z;
    diff_ys = diff_ys / scale_z;
    pred_w = pred_w / scale_z;
    pred_h = pred_h / scale_z;

    target_sz.width = std::round(float(target_sz.width) / scale_z);
    target_sz.height = std::round(float(target_sz.height) / scale_z);

    // size learning rate
    float lr = penalty_(maxRow, maxCol) * pred_score_(maxRow, maxCol) * lr_;

    // size rate
    float res_xs = target_bbox_.x + target_bbox_.width / 2.f + diff_xs;
    float res_ys = target_bbox_.y + target_bbox_.height / 2.f + diff_ys;
    float res_w = pred_w * lr + (1 - lr) * target_sz.width;
    float res_h = pred_h * lr + (1 - lr) * target_sz.height;

    target_bbox_.width = std::round(target_sz.width * (1 - lr) + lr * res_w);
    target_bbox_.height = std::round(target_sz.height * (1 - lr) + lr * res_h);
    target_bbox_.x = std::round(res_xs - target_bbox_.width / 2.f);
    target_bbox_.y = std::round(res_ys - target_bbox_.height / 2.f);

}

void LightTrack::gen_window() {
    cv::Mat window(16, 16, CV_32FC1);
    cv::createHanningWindow(window, cv::Size(16, 16), CV_32F);
    cv::cv2eigen(window, hanning_win_);
}

void LightTrack::gen_grids() {
    Eigen::Matrix<float, 1, 16> grid_x;
    Eigen::Matrix<float, 16, 1> grid_y;

    for (int i = 0; i < 16; ++i) {
        grid_x[i] = float(i) * 16;
        grid_y[i] = float(i) * 16;
    }
    grid_to_search_x_
            << grid_x, grid_x, grid_x, grid_x, grid_x, grid_x, grid_x, grid_x, grid_x, grid_x, grid_x, grid_x, grid_x, grid_x, grid_x, grid_x;
    grid_to_search_y_
            << grid_y, grid_y, grid_y, grid_y, grid_y, grid_y, grid_y, grid_y, grid_y, grid_y, grid_y, grid_y, grid_y, grid_y, grid_y, grid_y;
}


cv::Mat LightTrack::get_subwindow_tracking(cv::Mat &img, int model_sz, float original_sz, const cv::Scalar &avg_chans) {
    cv::Mat img_patch_ori;  // 填充不缩放
    cv::Mat img_patch;  // 填充+缩放之后输出的最终图像块
    cv::Rect crop;
    float c = (original_sz + 1) / 2.f;
    // 计算出剪裁边框的左上角和右下角
    int context_xmin = std::round(target_bbox_.x + target_bbox_.width / 2.f - c);
    int context_xmax = std::round(context_xmin + original_sz - 1);
    int context_ymin = std::round(target_bbox_.y + target_bbox_.height / 2.f - c);
    int context_ymax = std::round(context_ymin + original_sz - 1);
    // 边界部分要填充的像素
    int left_pad = std::max(0, -context_xmin);
    int top_pad = std::max(0, -context_ymin);
    int right_pad = std::max(0, context_xmax - img.cols + 1);
    int bottom_pad = std::max(0, context_ymax - img.rows + 1);
    // 填充之后的坐标
    crop.x = context_xmin + left_pad;
    crop.y = context_ymin + top_pad;
    crop.width = context_xmax + left_pad - crop.x;
    crop.height = context_ymax + top_pad - crop.y;
    // 填充像素
    if (left_pad > 0 || top_pad > 0 || right_pad > 0 || bottom_pad > 0) {
        cv::Mat pad_img = cv::Mat(img.rows + top_pad + bottom_pad, img.cols + left_pad + right_pad, CV_8UC3, avg_chans);
        for (int i = 0; i < img.rows; ++i) {
            memcpy(pad_img.data + (i + top_pad) * pad_img.cols * 3 + left_pad * 3, img.data + i * img.cols * 3,
                   img.cols * 3);
        }
        pad_img(crop).copyTo(img_patch_ori);
    } else
        img(crop).copyTo(img_patch_ori);
    if (img_patch_ori.rows != model_sz || img_patch_ori.cols != model_sz)
        cv::resize(img_patch_ori, img_patch, cv::Size(model_sz, model_sz));
    else
        img_patch = img_patch_ori;

    return img_patch;
}


cv::Rect LightTrack::read_gt(std::string &gt_file) {
    std::string line;
    cv::Rect gt_box;
    std::ifstream fin(gt_file);  //真实值文件
    getline(fin, line);
    if (!fin)
        std::cout << " Do not read groundtruth!!! " << std::endl;

    std::stringstream line_ss = std::stringstream(line);
    if (line.find(",") != std::string::npos) {
        std::vector<std::string> data;
        std::string tmp;
        while (getline(line_ss, tmp, ','))
            data.push_back(tmp);
        gt_box.x = stoi(data[0]);
        gt_box.y = stoi(data[1]);
        gt_box.width = stoi(data[2]);
        gt_box.height = stoi(data[3]);
    } else
        line_ss >> gt_box.x >> gt_box.y >> gt_box.width >> gt_box.height;

    return gt_box;
}


