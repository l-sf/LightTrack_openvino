# LightTrack (inference based on openvino)

官方仓库：

https://github.com/researchmm/LightTrack

论文：

[**LightTrack: Finding Lightweight Neural Networks for Object Tracking via One-Shot Architecture Search**](https://arxiv.org/abs/2104.14545)



## 说明

本仓库基于 Intel OpenVINO 推理引擎部署 LightTrack 跟踪算法，包含 Python、C++ 两种语言的推理代码 。

**优势**：方便部署，高性能。

​			本仓库的推理模型将预处理和部分后处理融入模型之中，使部署代码量更少，更加方便，并且推理引擎使得预处理速度更快。



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
./LightTrack 0 ../images/bag.avi
```

**摄像头输入:**

```bash
./LightTrack 1 0
```

**图片序列输入:**

```bash
./LightTrack 2 ../images/Woman/img/%04d.jpg
```



## Python demo Run

#### install dependency (python3.8)

```bash
pip install -r requirements.txt
```

#### run

```bash
cd /your_path/LightTrack_openvino/Python_Infer
# 视频文件
python infer.py --mode 0 --video '../images/bag.avi'
# 摄像头
python infer.py --mode 1
# 图片序列
python infer.py --mode 2 --image_path '../images/Woman/img/*.jpg'
```



## Export model and add preprocess tutorial

coming soon
