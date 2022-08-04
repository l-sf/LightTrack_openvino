# LightTrack (inference based on openvino)

官方 pytorch 代码仓库：

https://github.com/researchmm/LightTrack

论文：

[LightTrack: Finding Lightweight Neural Networks for Object Tracking via One-Shot Architecture Search](https://arxiv.org/abs/2104.14545)



## 介绍

本仓库基于 Intel OpenVINO Toolkit 部署 LightTrack 跟踪算法，包含 Python、C++ 两种语言的推理代码 。

**优势**：方便部署，高性能。

本仓库的推理模型将预处理和部分后处理融入模型之中，使部署代码量更少，更加方便，并且推理引擎使得预处理速度更快。



## 推理速度

| Intel CPU | preprocess+inference+postprocess average time |
| :-------: | :-------------------------------------------: |
| i7-11700K |                     3.4ms                     |
| i7-10710U |                     5.5ms                     |
| i7-7700HQ |                     7.5ms                     |

Python 和 C++ 速度测试代码如下，包括了整个 update 过程，并且模型中已经融入了预处理，因此测试的时间为 preprocess+inference+postprocess 的全部时间。

```python
# Python
start = time.time()
self.update(x_in_tensor, self.target_size * scale_z, scale_z)
end = time.time()
print('preprocess+inference+postprocess time = ', 1000 * (end - start), 'ms')
```

```c++
// C++
auto start = std::chrono::steady_clock::now();
update(x_crop, cv::Size(target_bbox_.width * scale_z, target_bbox_.height * scale_z), scale_z);
auto end = std::chrono::steady_clock::now();
std::chrono::duration<double> elapsed = end - start;
double time = 1000 * elapsed.count();
printf("preprocess+inference+postprocess time: %f ms\n", time);
```

**注**：实际程序运行速度取决于跟踪目标的大小，这里计时不包括图像块的剪裁和填充操作，目标越大耗时越多，速度越慢。完整流程C++ 程序比上述所测时间多耗时2ms左右。



## 导出模型 说明

[README](./models/README.md) 



## 安装 OpenVINO Toolkit

参考官网安装教程 [Get Started Guides](https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino_apt.html#doxid-openvino-docs-install-guides-installing-openvino-apt)

```bash
wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
echo "deb https://apt.repos.intel.com/openvino/2022 bionic main" | sudo tee /etc/apt/sources.list.d/intel-openvino-2022.list
sudo apt update
apt-cache search openvino
sudo apt install openvino
```

Run this command in shell. (Every time before using OpenVINO)

```bash
source /opt/intel/openvino_2022/setupvars.sh
```

安装python(3.8)依赖

```bash
cd /opt/intel/openvino_2022/tools
pip install -r requirements[onnx].txt
```



## Python demo Run

**视频文件输入:** 

```bash
python infer.py --mode 0 --video "../images/bag.avi"
```

**摄像头输入:**

```bash
python infer.py --mode 1
```

**图片序列输入:**

```bash
python infer.py --mode 2 --image_path "../images/Woman/img/*.jpg"
```



## C++ demo Build and Run

#### build

```bash
cd /your_path/LightTrack_openvino/Cpp_Infer
mkdir build && cd build
cmake .. && make -j
```

#### run

**视频文件输入:** 

```bash
./LightTrack 0 "../../images/bag.avi"
```

**摄像头输入:**

```bash
./LightTrack 1 0
```

**图片序列输入:**

```bash
./LightTrack 2 "../../images/Woman/img/%04d.jpg"
```



**注**：C++ 的跟踪效果不如Python好，大多数情况没有差别，在目标出现在接近视野右边时，目标框会发生向左的偏差，如果您发现bug还请不吝赐教。

