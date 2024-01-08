# Neural-Network-Quantization
## [Dynamic Quantization on BERT](https://pytorch.org/tutorials/intermediate/dynamic_quantization_bert_tutorial.html)
[这个博客](https://blog.csdn.net/zimiao552147572/article/details/105910915)写得挺详细。

基本run成功了。保存序列化文件遇到了问题，发邮件给有同样问题的人了。
## [Quantized Transfer Learning](https://pytorch.org/tutorials/intermediate/quantized_transfer_learning_tutorial.html)
这个在colab可以非常顺畅地运行。
## [FX GRAPH MODE POST TRAINING STATIC QUANTIZATION](https://pytorch.org/tutorials/prototype/fx_graph_mode_ptq_static.html)
运行不了：本教程描述了一个原型功能。 原型功能通常不作为 PyPI 或 Conda 等二进制发行版的一部分提供，除非有时在运行时标志后面，并且处于反馈和测试的早期阶段。

但imagenet的文件夹完整了。
/dataset01/zwc/myGitHub/Neural-Network-Quantization/FX GRAPH MODE POST TRAINING STATIC QUANTIZATION/data/imagenet
## [FX GRAPH MODE POST TRAINING DYNAMIC QUANTIZATION](https://pytorch.org/tutorials/prototype/fx_graph_mode_ptq_dynamic.html)
run成功了。
## [PYTORCH 2 EXPORT POST TRAINING QUANTIZATION](https://pytorch.org/tutorials/prototype/pt2e_quant_ptq.html)
卡在imagenet数据集了。正在下载。

可能运行不了：本教程描述了一个原型功能。 原型功能通常不作为 PyPI 或 Conda 等二进制发行版的一部分提供，除非有时在运行时标志后面，并且处于反馈和测试的早期阶段。
## [PYTORCH 2 EXPORT QUANTIZATION-AWARE TRAINING (QAT)](https://pytorch.org/tutorials/prototype/pt2e_quant_qat.html)
卡在imagenet数据集了。正在下载。

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