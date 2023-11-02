
#include "Tracker.hpp"
#include <string>
#include <cmath>
#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <inference_engine.hpp>
#include <openvino/openvino.hpp>


namespace LightTrack {

    using namespace std;

    using RMEigen = Eigen::Matrix<float, 16, 16, Eigen::RowMajor>;

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


    class TrackerImpl : public Tracker {
    public:
        ~TrackerImpl() = default;

        void startup(const std::string &z_path, const std::string &x_path, const std::string &head_path) {
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

        void init(cv::Mat &z_img, cv::Rect &init_bbox) override {
            target_bbox_ = init_bbox;
            cv::Scalar mean = cv::mean(z_img);
            float wc_z = target_bbox_.width + 0.5 * (target_bbox_.width + target_bbox_.height);
            float hc_z = target_bbox_.height + 0.5 * (target_bbox_.width + target_bbox_.height);
            float s_z = std::round(std::sqrt(wc_z * hc_z));  // (s_z)^2=(w+2p)x(h+2p), 模板图像上 不缩放时的 框加上pad 的大小
            cv::Mat z_patch(template_size_, template_size_, CV_8UC3);
            cropSubImg(z_img, z_patch, template_size_, s_z);

            // openvino推理部分
            ov::element::Type input_type = ov::element::u8;
            ov::Shape input_shape = {1, 128, 128, 3};
            // 使用ov::Tensor包装图像数据，无需分配新内存
            ov::Tensor input_tensor = ov::Tensor(input_type, input_shape, z_patch.data);
            z_infer_request_.set_input_tensor(input_tensor);
            z_infer_request_.infer();
            // 得到模板图像的特征张量
            zf_ = z_infer_request_.get_output_tensor();
            // 产生hanning窗，以及搜索图像上的grids
            gen_window();
            gen_grids();
        }

        cv::Rect track(cv::Mat &x_img) override {
            cv::Scalar mean = cv::mean(x_img);
            float wc_z = target_bbox_.width + 0.5 * (target_bbox_.width + target_bbox_.height);
            float hc_z = target_bbox_.height + 0.5 * (target_bbox_.width + target_bbox_.height);
            float s_z = std::round(std::sqrt(wc_z * hc_z));  // (s_z)^2=(w+2p)x(h+2p), 模板图像上 不缩放时的 框加上pad 的大小
            float scale_z = float(template_size_) / float(s_z);
            float d_search = float(search_size_ - template_size_) / 2.f;
            float pad = d_search / scale_z;
            float s_x = s_z + 2 * pad;
            cv::Mat x_patch(search_size_, search_size_, CV_8UC3);
            cropSubImg(x_img, x_patch, search_size_, s_x);

            // openvino推理部分
            ov::element::Type input_type = ov::element::u8;
            ov::Shape x_input_shape = {1, 256, 256, 3};
            // 使用ov::Tensor包装图像数据，无需分配新内存
            ov::Tensor x_input_tensor = ov::Tensor(input_type, x_input_shape, x_patch.data);
            x_infer_request_.set_input_tensor(x_input_tensor);
            x_infer_request_.infer();
            // 得到搜索图像的特征张量
            xf_ = x_infer_request_.get_output_tensor();

            head_infer_request_.set_input_tensor(0, zf_);
            head_infer_request_.set_input_tensor(1, xf_);
            head_infer_request_.infer();
            cls_score_ = head_infer_request_.get_output_tensor(0);
            bbox_pred_ = head_infer_request_.get_output_tensor(1);

            auto *pred_score = cls_score_.data<float>();
            auto *pred_box = bbox_pred_.data<float>();

            float *pred_x1 = pred_box;
            float *pred_y1 = pred_box + feat_sz_ * feat_sz_;
            float *pred_x2 = pred_box + 2 * feat_sz_ * feat_sz_;
            float *pred_y2 = pred_box + 3 * feat_sz_ * feat_sz_;
            pred_score_ = Eigen::Map<RMEigen>(pred_score, 16, 16);
            pred_x1_ = Eigen::Map<RMEigen>(pred_x1, 16, 16);
            pred_y1_ = Eigen::Map<RMEigen>(pred_y1, 16, 16);
            pred_x2_ = Eigen::Map<RMEigen>(pred_x2, 16, 16);
            pred_y2_ = Eigen::Map<RMEigen>(pred_y2, 16, 16);

            pred_x1_ = grid_to_search_x_ - pred_x1_;
            pred_y1_ = grid_to_search_y_ - pred_y1_;
            pred_x2_ = grid_to_search_x_ + pred_x2_;
            pred_y2_ = grid_to_search_y_ + pred_y2_;

            cv::Size_<float> target_sz(target_bbox_.width * scale_z, target_bbox_.height * scale_z);
            // size penalty
            s_c_ = change(
                    sz(pred_x2_ - pred_x1_, pred_y2_ - pred_y1_) /
                    sz(target_sz.width, target_sz.height));  // scale penalty
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

            float diff_xs = pred_xs - float(search_size_) / 2.f;
            float diff_ys = pred_ys - float(search_size_) / 2.f;

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

            // clip box
            target_bbox_.x = std::max(0, std::min(x_img.cols - 5, target_bbox_.x));
            target_bbox_.y = std::max(0, std::min(x_img.rows - 5, target_bbox_.y));
            target_bbox_.width = std::max(5, std::min(x_img.cols, target_bbox_.width));
            target_bbox_.height = std::max(5, std::min(x_img.rows, target_bbox_.height));

            return target_bbox_;
        }

    private:

        void cropSubImg(cv::Mat &img, cv::Mat &dst, int model_sz, float original_sz) const {
            cv::Mat img_patch_roi;  // 填充不缩放
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
            // 填充之后的坐标(即要裁切的ROI)
            if (left_pad > 0 || top_pad > 0 || right_pad > 0 || bottom_pad > 0) {
                // 填充像素
                cv::Mat pad_img;
                cv::copyMakeBorder(img, pad_img, top_pad, bottom_pad,
                                   left_pad, right_pad, cv::BORDER_CONSTANT);
                img_patch_roi = pad_img(crop);
            } else {
                img_patch_roi = img(crop);
            }
            // 缩放
            if (img_patch_roi.rows != model_sz || img_patch_roi.cols != model_sz)
                cv::resize(img_patch_roi, dst, cv::Size(model_sz, model_sz));
            else
                dst = img_patch_roi;
        }

        int template_size_ = 128;
        int search_size_ = 256;
        int feat_sz_ = 16;
        float penalty_k_ = 0.062; // 惩罚系数
        float window_influence_ = 0.15; // hanning窗权重系数
        float lr_ = 0.765; // 目标框更新的学习率
        cv::Rect target_bbox_; // 目标框

        ov::InferRequest z_infer_request_;
        ov::InferRequest x_infer_request_;
        ov::InferRequest head_infer_request_;
        ov::Tensor zf_;
        ov::Tensor xf_;
        ov::Tensor cls_score_;
        ov::Tensor bbox_pred_;

        RMEigen hanning_win_;
        RMEigen grid_to_search_x_;
        RMEigen grid_to_search_y_;
        RMEigen pred_score_;
        RMEigen pred_x1_;
        RMEigen pred_x2_;
        RMEigen pred_y1_;
        RMEigen pred_y2_;
        RMEigen s_c_;
        RMEigen r_c_;
        RMEigen penalty_;
        RMEigen pscore_;

        void gen_window() {
            cv::Mat window(16, 16, CV_32FC1);
            cv::createHanningWindow(window, cv::Size(16, 16), CV_32F);
            cv::cv2eigen(window, hanning_win_);
        }

        void gen_grids() {
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
    };

    shared_ptr<Tracker> create_tracker(const std::string &z_path, const std::string &x_path, const std::string &head_path){
        shared_ptr<TrackerImpl> instance(new TrackerImpl{});
        instance->startup(z_path, x_path, head_path);
        return instance;
    }

}
