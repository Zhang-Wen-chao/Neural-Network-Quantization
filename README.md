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

这个视频看完了P9

开源的大模型量化，github上自己做一做。
cpp开发岗位
工程部署，再多搞一点

可以在简历上写，熟悉什么量化算法，了解什么tensorrt的API接口，写过什么plugin，这样就有针对性了。
### 1月22日
为什么精度没有百分百对齐？这个还是得慢慢看，集中精力看，不能分心，分析过程挺长的。
### 1月14日
20系显卡对应CUDA 10,30系对应CUDA 11。
经验证，更换4090显卡后，基于cuda10.2编译的pytorch已不受支持；
GeForce RTX 30系显卡支持CUDA 11.1及以上版本
所以我使用CUDA10.2是不行的。
```bash
docker pull ubuntu:22.04
sudo docker run -it -d \
    --name mmyolo_ubuntu22_cuda12 \
    -v /dataset01:/dataset01 \
    --network="host" \
    --gpus all \
    ubuntu:22.04
sudo docker exec -it mmyolo_ubuntu22_cuda12 /bin/bash

apt-get install git zsh wget
sh -c "$(wget https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh -O -)"
git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting
git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions
apt-get install autojump
sudo docker exec -it mmyolo_ubuntu22_cuda12 /bin/zsh

apt install gcc-9 g++-9
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 9
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 9
update-alternatives --install /usr/bin/x86_64-linux-gnu-g++ x86_64-linux-gnu-g++ /usr/bin/g++-9 90
update-alternatives --install /usr/bin/x86_64-linux-gnu-gcc x86_64-linux-gnu-gcc /usr/bin/gcc-9 90
```
### 1月15日
我被这个tensorrt伤透了心。等有空了，还是直接拉tensorrt的docker吧。从Ubuntu的dockers上构建的，明明按照官方支持版本安装了，但是还是出现各种小问题。

[caskBuilderUtils.cpp::trtSmToCaskCCV::548] Error Code 1: Internal Error (Unsupported SM: 0x809)

AttributeError: 'NoneType' object has no attribute 'create_execution_context'
### 1月17日
安装成功了
```bash
docker pull nvcr.io/nvidia/tensorrt:22.12-py3
sudo docker run -it -d \
    --name mmyolo_trt8.5.1 \
    -v /dataset01:/dataset01 \
    --network="host" \
    --gpus all \
    nvcr.io/nvidia/tensorrt:22.12-py3

sudo docker exec -it mmyolo_trt8.5.1 /bin/bash

apt install ninja-build
apt-get install libgl1-mesa-glx
```
### 3月5日
从头开始再看一遍，到了p?
## 复现论文
https://arxiv.org/abs/2204.06806

https://arxiv.org/pdf/2206.00820.pdf

https://github.com/ECoLab-POSTECH/NIPQ

https://zhuanlan.zhihu.com/p/651390746

交大的powerinfer

flexgen llm

https://zhuanlan.zhihu.com/p/664164593
## 记录一些想看的网页
cmu catalyst团队出了llm推理的综述，你可以简单介绍一下吗？
大模型如何高效部署？CMU最新万字综述纵览LLM推理MLSys优化技术 - Hsword的文章 - 知乎
https://zhuanlan.zhihu.com/p/677635306

https://www.zhihu.com/question/637480772

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