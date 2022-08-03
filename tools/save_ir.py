#
# Created by lsf on 2022/7/19.
#

from openvino.preprocess import PrePostProcessor, ColorFormat
from openvino.runtime import Core, Layout, Type
from openvino.offline_transformations import serialize


z_feature_path = "/home/c2214/Object_Tracking/LightTrack/models/new_ir_models/z_feature.xml"
x_feature_path = "/home/c2214/Object_Tracking/LightTrack/models/new_ir_models/x_feature.xml"

# Step1: 创建OpenVINO运行时
core = Core()

# Step2: 读取模型、加载模型
z_feature_model = core.read_model(z_feature_path)
x_feature_model = core.read_model(x_feature_path)

# Step3: 通过OpenVINO预处理器将预处理功能集成到模型中
# 模板分支的预处理
z_ppp = PrePostProcessor(z_feature_model)
z_ppp.input().tensor() \
    .set_element_type(Type.u8) \
    .set_color_format(ColorFormat.BGR) \
    .set_layout(Layout('NHWC'))
z_ppp.input().model().set_layout(Layout('NCHW'))
z_ppp.output().tensor().set_element_type(Type.f32)
z_ppp.input().preprocess() \
    .convert_element_type(Type.f32) \
    .convert_color(ColorFormat.RGB) \
    .mean([123.675, 116.28, 103.53]) \
    .scale([58.395, 57.12, 57.375])
z_feature_model = z_ppp.build()
# 搜索分支的预处理
x_ppp = PrePostProcessor(x_feature_model)
x_ppp.input().tensor() \
    .set_color_format(ColorFormat.BGR) \
    .set_element_type(Type.u8) \
    .set_layout(Layout('NHWC'))
x_ppp.input().model().set_layout(Layout('NCHW'))
x_ppp.output().tensor().set_element_type(Type.f32)
x_ppp.input().preprocess() \
    .convert_element_type(Type.f32) \
    .convert_color(ColorFormat.RGB) \
    .mean([123.675, 116.28, 103.53]) \
    .scale([58.395, 57.12, 57.375])
x_feature_model = x_ppp.build()

# Step4: 使用预处理保存模型
serialize(z_feature_model, 'z_feature.xml', 'z_feature.bin')
serialize(x_feature_model, 'x_feature.xml', 'x_feature.bin')

