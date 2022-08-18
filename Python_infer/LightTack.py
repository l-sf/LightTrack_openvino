#
# Created by lsf on 2022/7/19.
#
import cv2
import numpy as np
from openvino.runtime import Core
from utils import get_subwindow_tracking
import time


def change(r):
    return np.maximum(r, 1. / r)


def sz(w, h):
    pad = (w + h) * 0.5
    sz2 = (w + pad) * (h + pad)
    return np.sqrt(sz2)


def sz_wh(wh):
    pad = (wh[0] + wh[1]) * 0.5
    sz2 = (wh[0] + pad) * (wh[1] + pad)
    return np.sqrt(sz2)


class LightTrack(object):
    def __init__(self, z_feature_path, x_feature_path, neck_head_path):
        core = Core()
        # 加载模型
        z_feature_model = core.read_model(z_feature_path)
        x_feature_model = core.read_model(x_feature_path)
        neck_head_model = core.read_model(neck_head_path)
        # 编译模型： 模板分支、搜索分支、分类回归头
        self.compiled_z_feature_model = core.compile_model(z_feature_model, device_name="CPU")
        self.compiled_x_feature_model = core.compile_model(x_feature_model, device_name="CPU")
        self.compiled_head_model = core.compile_model(neck_head_model, device_name="CPU")
        # 创建推理请求
        self.z_infer_request = self.compiled_z_feature_model.create_infer_request()
        self.x_infer_request = self.compiled_x_feature_model.create_infer_request()
        self.head_infer_request = self.compiled_head_model.create_infer_request()

        # 一些跟踪中的参数
        self.exemplar_size = 128
        self.instance_size = 256
        self.total_stride = 16
        self.penalty_k = 0.062
        self.window_influence = 0.38
        self.lr = 0.765
        self.score_size = int(round(self.instance_size / self.total_stride))  # 16

        self.target_pos, self.target_size = None, None
        self.zf, self.xf = None, None
        self.window = None
        self.grid_to_search_x, self.grid_to_search_y = None, None
        self.debug_num = 0

    def init(self, z_img, target_pos, target_size):
        self.target_pos = target_pos
        self.target_size = target_size
        avg_chans = np.mean(z_img, axis=(0, 1))  # .astype('float32')
        wc_z = target_size[0] + 0.5 * sum(target_size)
        hc_z = target_size[1] + 0.5 * sum(target_size)
        s_z = round(np.sqrt(wc_z * hc_z))  # (s_z)^2=(w+2p)x(h+2p), 模板图像上 不缩放时的 框加上pad 的大小
        z_crop = get_subwindow_tracking(z_img, target_pos, self.exemplar_size, s_z, avg_chans)
        z_in_tensor = np.expand_dims(z_crop, axis=0)

        self.z_infer_request.infer({0: z_in_tensor})
        self.zf = self.z_infer_request.get_output_tensor().data

        self.window = np.outer(np.hanning(self.score_size), np.hanning(self.score_size))  # [16,16]
        self.grids()

    def update(self, x_in_tensor, target_sz, scale_z):
        self.x_infer_request.infer({0: x_in_tensor})
        self.xf = self.x_infer_request.get_output_tensor().data
        self.head_infer_request.infer({0: self.zf, 1: self.xf})

        cls_score = self.head_infer_request.get_tensor(self.compiled_head_model.outputs[0]).data
        bbox_pred = self.head_infer_request.get_tensor(self.compiled_head_model.outputs[1]).data

        pred_x1 = self.grid_to_search_x - bbox_pred[0, ...]
        pred_y1 = self.grid_to_search_y - bbox_pred[1, ...]
        pred_x2 = self.grid_to_search_x + bbox_pred[2, ...]
        pred_y2 = self.grid_to_search_y + bbox_pred[3, ...]

        # size penalty
        s_c = change(sz(pred_x2 - pred_x1, pred_y2 - pred_y1) / (sz_wh(target_sz)))  # scale penalty
        r_c = change((target_sz[0] / target_sz[1]) / ((pred_x2 - pred_x1) / (pred_y2 - pred_y1)))  # ratio penalty

        penalty = np.exp(-(r_c * s_c - 1) * self.penalty_k)
        pscore = penalty * cls_score

        # window penalty
        pscore = pscore * (1 - self.window_influence) + self.window * self.window_influence

        # get max
        r_max, c_max = np.unravel_index(pscore.argmax(), pscore.shape)

        # to real size
        pred_x1 = pred_x1[r_max, c_max]
        pred_y1 = pred_y1[r_max, c_max]
        pred_x2 = pred_x2[r_max, c_max]
        pred_y2 = pred_y2[r_max, c_max]

        pred_xs = (pred_x1 + pred_x2) / 2
        pred_ys = (pred_y1 + pred_y2) / 2
        pred_w = pred_x2 - pred_x1
        pred_h = pred_y2 - pred_y1

        diff_xs = pred_xs - self.instance_size // 2
        diff_ys = pred_ys - self.instance_size // 2

        diff_xs, diff_ys, pred_w, pred_h = diff_xs / scale_z, diff_ys / scale_z, pred_w / scale_z, pred_h / scale_z

        target_sz = target_sz / scale_z

        # size learning rate
        lr = penalty[r_max, c_max] * cls_score[r_max, c_max] * self.lr

        # size rate
        res_xs = self.target_pos[0] + diff_xs
        res_ys = self.target_pos[1] + diff_ys
        res_w = pred_w * lr + (1 - lr) * target_sz[0]
        res_h = pred_h * lr + (1 - lr) * target_sz[1]

        self.target_pos = np.array([res_xs, res_ys])
        self.target_size = target_sz * (1 - lr) + lr * np.array([res_w, res_h])

    def track(self, x_img):
        avg_chans = np.mean(x_img, axis=(0, 1))
        wc_z = self.target_size[0] + 0.5 * sum(self.target_size)
        hc_z = self.target_size[1] + 0.5 * sum(self.target_size)
        s_z = round(np.sqrt(wc_z * hc_z))
        scale_z = self.exemplar_size / s_z
        d_search = (self.instance_size - self.exemplar_size) / 2
        pad = d_search / scale_z
        s_x = s_z + 2 * pad
        x_crop = get_subwindow_tracking(x_img, self.target_pos, self.instance_size, s_x,
                                        avg_chans)

        x_in_tensor = np.expand_dims(x_crop, axis=0)

        start = time.time()
        self.update(x_in_tensor, self.target_size * scale_z, scale_z)
        end = time.time()
        print('preprocess+inference+postprocess time = ', 1000 * (end - start), 'ms')

        self.target_pos[0] = max(0, min(x_img.shape[1], self.target_pos[0]))
        self.target_pos[1] = max(0, min(x_img.shape[0], self.target_pos[1]))
        self.target_size[0] = max(10, min(x_img.shape[1], self.target_size[0]))
        self.target_size[1] = max(10, min(x_img.shape[0], self.target_size[1]))

        self.debug_num += 1

        return self.target_pos, self.target_size

    def grids(self):
        """
        each element of feature map on input search image
        return: H*W*2 (position for each element)
        """
        sz = self.score_size

        sz_x = sz // 2
        sz_y = sz // 2

        x, y = np.meshgrid(np.arange(0, sz) - np.floor(float(sz_x)),
                           np.arange(0, sz) - np.floor(float(sz_y)))
        self.grid_to_search_x = x * self.total_stride + self.instance_size // 2
        self.grid_to_search_y = y * self.total_stride + self.instance_size // 2
