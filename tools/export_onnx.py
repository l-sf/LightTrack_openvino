#
# Created by lsf on 2022/7/19.
#

import argparse
import torch
import onnx
import onnxsim
import lib.models.models as models
from lib.utils.utils import load_pretrain


def main(args):
    net = models.LightTrackM_Subnet(args.path_name, args.stride)
    net = load_pretrain(net, args.resume)

    dummy_input = (
        # torch.randn(1, 3, 128, 128),
        # torch.randn(1, 3, 256, 256),
        torch.randn(1, 96, 8, 8),
        torch.randn(1, 96, 16, 16),
    )

    torch.onnx.export(
        net,
        dummy_input,
        args.output_path,
        verbose=True,
        opset_version=11,
        input_names=["zf", "xf"],
        output_names=["cls", "reg"],
        # dynamic_axes={
        #     "in_feature": {2: "h", 3: "w"},
        #     "out_feature": {2: "h", 3: "w"},
        # }
    )
    print('----------finished exporting onnx-----------')
    print('----------start simplifying onnx-----------')
    model_sim, flag = onnxsim.simplify(args.output_path)
    if flag:
        onnx.save(model_sim, args.output_path)
        print("---------simplify onnx successfully---------")
    else:
        print("---------simplify onnx failed-----------")


if __name__ == '__main__':
    # Parse command line arguments.
    parser = argparse.ArgumentParser(description='Test LightTrack')
    parser.add_argument('--output_path', default='./head.onnx', help='onnx output path')
    parser.add_argument('--arch', default='LightTrackM_Subnet', type=str, help='backbone architecture')
    parser.add_argument('--resume', default='../snapshot/LightTrackM/LightTrackM.pth', type=str,
                        help='pretrained model')
    parser.add_argument('--dataset', default='VOT2019', help='dataset test')
    parser.add_argument('--epoch_test', default=False, type=bool, help='multi-gpu epoch test flag')
    parser.add_argument('--video', default=None, type=str, help='test a video in benchmark')
    parser.add_argument('--stride', default=16, type=int, help='network stride')
    parser.add_argument('--even', default=0, type=int)
    parser.add_argument('--path_name', type=str, default='back_04502514044521042540+cls_211000022+reg_100000111_ops_32')
    args = parser.parse_args()
    print(args)
    main(args)
