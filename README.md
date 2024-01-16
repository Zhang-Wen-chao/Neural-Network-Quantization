# Neural-Network-Quantization
## [Introduction to Quantization on PyTorch](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/)

### [Dynamic Quantization on BERT](https://pytorch.org/tutorials/intermediate/dynamic_quantization_bert_tutorial.html)
[这个博客](https://blog.csdn.net/zimiao552147572/article/details/105910915)写得挺详细。

基本run成功了。保存序列化文件遇到了问题，发邮件给有同样问题的人了。
### [Dynamic Quantization on an LSTM Word Language Model](https://pytorch.org/tutorials/advanced/dynamic_quantization_tutorial.html)
在colab都运行不了。

wikitext-2数据集也不知道去哪里下载带有validation的txt格式。
### [QUANTIZED TRANSFER LEARNING FOR COMPUTER VISION TUTORIAL](https://pytorch.org/tutorials/intermediate/quantized_transfer_learning_tutorial.html)
这个在colab可以非常顺畅地运行。
### [FX GRAPH MODE POST TRAINING STATIC QUANTIZATION](https://pytorch.org/tutorials/prototype/fx_graph_mode_ptq_static.html)
NameError: name 'qconfig_opt' is not defined

运行不了：本教程描述了一个原型功能。 原型功能通常不作为 PyPI 或 Conda 等二进制发行版的一部分提供，除非有时在运行时标志后面，并且处于反馈和测试的早期阶段。

但imagenet的文件夹完整了。
Neural-Network-Quantization/FX_GRAPH_MODE_POST_TRAINING_STATIC_QUANTIZATION/data/imagenet
### [FX GRAPH MODE POST TRAINING DYNAMIC QUANTIZATION](https://pytorch.org/tutorials/prototype/fx_graph_mode_ptq_dynamic.html)
run成功了。
## [PYTORCH 2 EXPORT POST TRAINING QUANTIZATION](https://pytorch.org/tutorials/prototype/pt2e_quant_ptq.html)
NameError: name 'qconfig_opt' is not defined

运行不了：本教程描述了一个原型功能。 原型功能通常不作为 PyPI 或 Conda 等二进制发行版的一部分提供，除非有时在运行时标志后面，并且处于反馈和测试的早期阶段。
### [PYTORCH 2 EXPORT QUANTIZATION-AWARE TRAINING (QAT)](https://pytorch.org/tutorials/prototype/pt2e_quant_qat.html)
NameError: name 'qconfig_opt' is not defined

运行不了：本教程描述了一个原型功能。 原型功能通常不作为 PyPI 或 Conda 等二进制发行版的一部分提供，除非有时在运行时标志后面，并且处于反馈和测试的早期阶段。
## 进度
复现mmyolo_tensorrt
https://www.bilibili.com/video/BV1Ds4y1k7yr/?vd_source=fb6ecc817428ba6260742f25efd17059

这个视频看完了P3, 
```bash
sudo docker pull ubuntu:18.04

sudo docker run -it -d \
    --name mmyolo_tensorrt \
    -v /dataset01:/dataset01 \
    --network="host" \
    --gpus all \
    ubuntu:18.04

sudo docker exec -it mmyolo_tensorrt /bin/bash

# 然后配置了zsh, 上网端口
export http_proxy="http://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"

sudo docker exec -it mmyolo_tensorrt /bin/zsh

update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 70
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 80

update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 70
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 80

update-alternatives --install /usr/bin/x86_64-linux-gnu-g++ x86_64-linux-gnu-g++ /usr/bin/x86_64-linux-gnu-g++-7 70
update-alternatives --install /usr/bin/x86_64-linux-gnu-g++ x86_64-linux-gnu-g++ /usr/bin/x86_64-linux-gnu-g++-8 80

update-alternatives --config gcc
update-alternatives --config g++
update-alternatives --config x86_64-linux-gnu-g++

gcc --version
g++ --version
x86_64-linux-gnu-g++ --version

wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
sudo sh cuda_10.2.89_440.33.01_linux.run

export PATH=/usr/local/cuda-10.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64:$LD_LIBRARY_PATH
# 安装完CUDA记得安装CUDNN

apt install python3.8
apt-get install python3.8-dev
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3.8 get-pip.py
# then, follow the readme.md of mmyolo_tensorrt.

export PYTHONPATH=/dataset01/zwc/mmyolo-hb/mmengine:$PYTHONPATH
export PYTHONPATH=/dataset01/zwc/mmyolo-hb/mmcv:$PYTHONPATH
export PYTHONPATH=/dataset01/zwc/mmyolo-hb/mmdetection:$PYTHONPATH
```
### 1月14日
才发现，20系显卡对应CUDA 10,30系对应CUDA 11。
经验证，更换4090显卡后，基于cuda10.2编译的pytorch已不受支持；
GeForce RTX 30系显卡支持CUDA 11.1及以上版本
所以我使用CUDA10.2是不行的。

我现在打算拉一个Ubuntu 22.04的docker，安装最新的CUDA12.1
```bash
docker pull ubuntu:22.04
sudo docker run -it -d \
    --name mmyolo_ubuntu22_cuda12 \
    -v /dataset01:/dataset01 \
    --network="host" \
    --gpus all \
    ubuntu:22.04

sudo docker exec -it mmyolo_ubuntu22_cuda12 /bin/bash

export http_proxy="http://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"
apt-get install git zsh wget
sh -c "$(wget https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh -O -)"
git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting
git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions
apt-get install autojump

sudo docker exec -it mmyolo_ubuntu22_cuda12 /bin/zsh

apt-get install gcc-10 g++-10
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 10
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 10
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 11
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 11
update-alternatives --install /usr/bin/x86_64-linux-gnu-g++ x86_64-linux-gnu-g++ /usr/bin/g++-10 100
update-alternatives --install /usr/bin/x86_64-linux-gnu-g++ x86_64-linux-gnu-g++ /usr/bin/g++-11 110
add-apt-repository ppa:ubuntu-toolchain-r/test
apt update
apt install gcc-9 g++-9
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 9
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 9
update-alternatives --install /usr/bin/x86_64-linux-gnu-g++ x86_64-linux-gnu-g++ /usr/bin/g++-9 90


wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run

export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
https://www.cnblogs.com/lycnight/p/17768264.html
tar -xvf cudnn-linux-x86_64-8.9.6.50_cuda12-archive.tar.xz
apt install python3.10
apt install python3.10-dev
apt-get install python3-pip

MMCV_WITH_OPS=1 python3 setup.py develop
# 失败，开始安装gcc9
apt install ninja-build
```
### 1月15日
我被这个tensorrt伤透了心。等有空了，还是直接拉nvidia：tensorrt的docker吧。从Ubuntu的dockers上构建的，明明按照官方支持版本安装了，但是还是出现各种小问题。

[caskBuilderUtils.cpp::trtSmToCaskCCV::548] Error Code 1: Internal Error (Unsupported SM: 0x809)

AttributeError: 'NoneType' object has no attribute 'create_execution_context'
```bash
python projects/easydeploy/tools/export.py \
    configs/yolov5/yolov5_s-v61_fast_1xb12-40e_cat.py \
    work_dirs/yolov5_s-v61_fast_1xb12-40e_cat/epoch_40.pth \
    --work-dir work_dirs/yolov5_s-v61_fast_1xb12-40e_cat \
    --img-size 640 640 \
    --batch 1 \
    --device cpu \
    --simplify \
    --opset 11 \
    --backend 1 \
    --pre-topk 1000 \
    --keep-topk 100 \
    --iou-threshold 0.65 \
    --score-threshold 0.25
```

## 复现论文
https://arxiv.org/abs/2204.06806

https://arxiv.org/pdf/2206.00820.pdf

https://github.com/ECoLab-POSTECH/NIPQ

https://zhuanlan.zhihu.com/p/651390746
## 记录一些想看的网页
https://docs.nvidia.com/deeplearning/tensorrt/pytorch-quantization-toolkit/docs/index.html

https://pytorch.org/blog/quantization-in-practice/#post-training-static-quantization-ptq

https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization

https://www.zhihu.com/question/411393222/answer/2359479242

https://www.cvmart.net/community?keyword=%E9%87%8F%E5%8C%96

https://mp.weixin.qq.com/mp/appmsgalbum?__biz=Mzg4ODA3MDkyMA==&action=getalbum&album_id=1648871870714200067&scene=173&from_msgid=2247483692&from_itemidx=1&count=3&nolastread=1#wechat_redirect

https://zhuanlan.zhihu.com/c_1258047709686231040

https://www.cnblogs.com/bigoldpan/p/16717472.html

https://www.cnblogs.com/bigoldpan/p/16328630.html

https://oldpan.me/archives/a-thought-of-oldpan

https://www.cnblogs.com/bigoldpan/p/16717472.html

https://oldpan.me/

https://www.cnblogs.com/bigoldpan/p/16296035.html

论文
https://arxiv.org/abs/1712.05877?spm=ata.21736010.0.0.5d155919bwSdHC&file=1712.05877

mmdeploy

https://hanlab.mit.edu/songhan

## tips
### jupyter测试代理
```python
import requests

proxies = {
    "http": "http://127.0.0.1:7890",
    "https": "http://127.0.0.1:7890",
}
try:
    response = requests.get("https://www.google.com", proxies=proxies)
    print("Network connection is working.")
except requests.exceptions.RequestException as e: 
    print("Network connection is not working.")

```