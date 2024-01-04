# Neural-Network-Quantization
## Dynamic Quantization on BERT
保存序列化文件遇到问题，发邮件给作者了。

下面这个博客写得挺详细。
https://blog.csdn.net/zimiao552147572/article/details/105910915

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