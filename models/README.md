# 增加预处理和后处理的完整FP16精度模型

### 预处理如下：

输入格式从‘NHWC’  转为  ‘NCHW’

unsigned char(int8)  转为  float32

颜色通道BGR  转为  RGB

三通道除以均值 [123.675, 116.28, 103.53]

减标准差 [58.395, 57.12, 57.375]

```python
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
```

### 后处理如下：

分类分支增加 压缩维度+sigmoid

```python
cls_score = sigmoid(np.squeeze(cls))
```

回归分支增加 压缩维度

```python
bbox_pred = np.squeeze(reg)
```



