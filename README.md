# Neural-Network-Quantization
## Dynamic Quantization on BERT
https://pytorch.org/tutorials/intermediate/dynamic_quantization_bert_tutorial.html
基本run成功了。保存序列化文件遇到了问题，发邮件给有同样问题的人了。

下面这个博客写得挺详细。
https://blog.csdn.net/zimiao552147572/article/details/105910915
## Quantized Transfer Learning
这个在colab可以非常顺畅地运行。
https://pytorch.org/tutorials/intermediate/quantized_transfer_learning_tutorial.html
## FX GRAPH MODE POST TRAINING STATIC QUANTIZATION
卡在imagenet数据集了。正在下载。
cat split.tar.gz.* > split_bak.tar.gz

https://pytorch.org/tutorials/prototype/fx_graph_mode_ptq_static.html
## FX GRAPH MODE POST TRAINING DYNAMIC QUANTIZATION
run成功了。
https://pytorch.org/tutorials/prototype/fx_graph_mode_ptq_dynamic.html

## PYTORCH 2 EXPORT POST TRAINING QUANTIZATION
卡在imagenet数据集了。正在下载。
https://pytorch.org/tutorials/prototype/pt2e_quant_ptq.html

## PYTORCH 2 EXPORT QUANTIZATION-AWARE TRAINING (QAT)
卡在imagenet数据集了。正在下载。
https://pytorch.org/tutorials/prototype/pt2e_quant_qat.html
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