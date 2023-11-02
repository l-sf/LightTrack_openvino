# 导出模型、添加预处理和部分后处理 tutorials



## 一、导出模型

由于原代码是训练代码，如果直接导出模型，会把完整的带有模板分支和搜索分支的模型导出，这样并不是one-shot的推理。

因此我们要导出3个模型，分别是模板分支、搜索分支、neck_head。（虽然backbone是共享参数，但是openvino建议我们最好固定模型shape，因此我们还是分成两个模型，不使用动态shape）

源码中的 [forward函数](https://github.com/researchmm/LightTrack/blob/main/lib/models/super_model_DP.py#L133) ，修改为

```python
def forward(self, zf, xf):
    # Batch Normalization before Corr
    zf, xf = self.neck(zf, xf)
    # Point-wise Correlation
    feat_dict = self.feature_fusor(zf, xf)
    # supernet head
    oup = self.head(feat_dict)
    # 添加部分后处理，方便C++端部署
    # 分类分支增加 压缩维度+sigmoid
    cls_score = nn.functional.sigmoid(torch.squeeze(oup['cls']))
    # 回归分支增加 压缩维度
    bbox_pred = torch.squeeze(oup['reg'])
    return cls_score, bbox_pred
```



#### 1 导出模板分支

```python
net = models.LightTrackM_Subnet('back_04502514044521042540+cls_211000022+reg_100000111_ops_32', 16)
net = load_pretrain(net, '../snapshot/LightTrackM/LightTrackM.pth')
dummy_input = (
    torch.randn(1, 3, 128, 128),
    )
torch.onnx.export(
    net.features,
    dummy_input,
    args.output_path,
    verbose=True,
    opset_version=11,
    input_names=["z"],
    output_names=["zf"],
    )
print('----------finished exporting onnx-----------')
print('----------start simplifying onnx-----------')
model_sim, flag = onnxsim.simplify('./z_feature.onnx')
if flag:
    onnx.save(model_sim, './z_feature.onnx')
    print("---------simplify onnx successfully---------")
else:
    print("---------simplify onnx failed-----------")
```



#### 2 导出搜索分支

```python
net = models.LightTrackM_Subnet('back_04502514044521042540+cls_211000022+reg_100000111_ops_32', 16)
net = load_pretrain(net, '../snapshot/LightTrackM/LightTrackM.pth')
dummy_input = (
    torch.randn(1, 3, 256, 256),
    )
torch.onnx.export(
    net.features,
    dummy_input,
    args.output_path,
    verbose=True,
    opset_version=11,
    input_names=["x"],
    output_names=["xf"],
    )
print('----------finished exporting onnx-----------')
print('----------start simplifying onnx-----------')
model_sim, flag = onnxsim.simplify('./x_feature.onnx')
if flag:
    onnx.save(model_sim, './x_feature.onnx')
    print("---------simplify onnx successfully---------")
else:
    print("---------simplify onnx failed-----------")
```



#### 3 导出neck_head

```python
net = models.LightTrackM_Subnet('back_04502514044521042540+cls_211000022+reg_100000111_ops_32', 16)
net = load_pretrain(net, '../snapshot/LightTrackM/LightTrackM.pth')
dummy_input = (
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
    )
print('----------finished exporting onnx-----------')
print('----------start simplifying onnx-----------')
model_sim, flag = onnxsim.simplify('./neck_head.onnx')
if flag:
    onnx.save(model_sim, './neck_head.onnx')
    print("---------simplify onnx successfully---------")
else:
    print("---------simplify onnx failed-----------")
```

完整代码在  [tools/export_onnx.py](../tools/export_onnx.py)



## 二、添加预处理

OpenVINO 支持将预处理融入到模型当中，提高速度，方便部署。

#### 1 转换模型

我们首先把上一步得到的3个 ONNX 模型转换成 OpenVINO 模型：

```bash
mo --input_model z_feature.onnx --data_type FP16 --output_dir ./new_ir_models
mo --input_model x_feature.onnx --data_type FP16 --output_dir ./new_ir_models
mo --input_model neck_head.onnx --data_type FP16 --output_dir ./new_ir_models
```



#### 2 添加预处理

模板分支和搜索分支进行相同的预处理。

输入格式  `NHWC`  转为  `NCHW` ；

数据格式  `unsigned char(int8)`  转为  `float32` ；

颜色通道  `BGR`  转为  `RGB` ；

三个通道除以均值  $[123.675, 116.28, 103.53]$ ；

减标准差  $[58.395, 57.12, 57.375]$ 。

```python
# 通过OpenVINO预处理器将预处理功能集成到模型中
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
# 使用预处理保存模型 
serialize(z_feature_model, 'z_feature.xml', 'z_feature.bin')
```

完整代码在 [tools/save_ir.py](../tools/save_ir.py) 

