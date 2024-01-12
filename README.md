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
## jupyter测试代理
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
## 进度
复现mmyolo_tensorrt
https://www.bilibili.com/video/BV1Ds4y1k7yr/?vd_source=fb6ecc817428ba6260742f25efd17059

这个视频看完了P1。
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
```

## 复现论文
https://arxiv.org/abs/2204.06806


https://arxiv.org/pdf/2206.00820.pdf

https://github.com/ECoLab-POSTECH/NIPQ

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