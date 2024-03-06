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
# 毕设进度
## 环境安装成功
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
## p4 mmyolo初体验
### 训练
```bash
sudo docker exec -it mmyolo_trt8.5.1 /bin/zsh
cd /dataset01/zwc/mmyolo-hb/mmyolo
python tools/train.py configs/yolov5/yolov5_s-v61_fast_1xb12-40e_cat.py
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.745
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.942
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.876
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.745
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.697
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.797
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.813
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.813
03/06 14:13:12 - mmengine - INFO - bbox_mAP_copypaste: 0.745 0.942 0.876 -1.000 -1.000 0.745
03/06 14:13:12 - mmengine - INFO - Epoch(val) [40][28/28]    coco/bbox_mAP: 0.7450  coco/bbox_mAP_50: 0.9420  coco/bbox_mAP_75: 0.8760  coco/bbox_mAP_s: -1.0000  coco/bbox_mAP_m: -1.0000  coco/bbox_mAP_l: 0.7450  data_time: 0.0873  time: 0.1076
03/06 14:13:12 - mmengine - INFO - The previous best checkpoint /dataset01/zwc/mmyolo-hb/mmyolo/work_dirs/yolov5_s-v61_fast_1xb12-40e_cat/best_coco_bbox_mAP_epoch_30.pth is removed
03/06 14:13:15 - mmengine - INFO - The best checkpoint with 0.7450 coco/bbox_mAP at 40 epoch is saved to best_coco_bbox_mAP_epoch_40.pth.
### 导出onnx
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
ONNX export success, save into work_dirs/yolov5_s-v61_fast_1xb12-40e_cat/end2end.onnx
### onnx用cpu推理
```bash
python projects/easydeploy/tools/image-demo.py \
    data/cat/images/IMG_20210728_205117.jpg \
    configs/yolov5/yolov5_s-v61_fast_1xb12-40e_cat.py \
    work_dirs/yolov5_s-v61_fast_1xb12-40e_cat/end2end.onnx \
    --device cpu
# 推理结果在下面目录：
/dataset01/zwc/mmyolo-hb/mmyolo/output/IMG_20210728_205117.jpg
```
### 导出 tensorrt8 部署需要的onnx
```bash
python projects/easydeploy/tools/export.py \
    configs/yolov5/yolov5_s-v61_fast_1xb12-40e_cat.py \
    work_dirs/yolov5_s-v61_fast_1xb12-40e_cat/epoch_40.pth \
    --work-dir work_dirs/yolov5_s-v61_fast_1xb12-40e_cat \
    --img-size 640 640 \
    --batch 1 \
    --device cuda:0 \
    --simplify \
    --opset 11 \
    --backend 2 \
    --pre-topk 1000 \
    --keep-topk 100 \
    --iou-threshold 0.65 \
    --score-threshold 0.25
```
ONNX export success, save into work_dirs/yolov5_s-v61_fast_1xb12-40e_cat/end2end.onnx
### 用英伟达显卡生成序列化文件
```bash
python projects/easydeploy/tools/build_engine.py \
    work_dirs/yolov5_s-v61_fast_1xb12-40e_cat/end2end.onnx \
    --img-size 640 640 \
    --device cuda:0
```
Save in /dataset01/zwc/mmyolo-hb/mmyolo/work_dirs/yolov5_s-v61_fast_1xb12-40e_cat/end2end.engine
### 用英伟达显卡进行图片推理
```bash
python projects/easydeploy/tools/image-demo.py \
    data/cat/images/IMG_20210728_205312.jpg \
    configs/yolov5/yolov5_s-v61_fast_1xb12-40e_cat.py \
    work_dirs/yolov5_s-v61_fast_1xb12-40e_cat/end2end.engine \
    --device cuda:0
# 推理结果在下面目录：
/dataset01/zwc/mmyolo-hb/mmyolo/output/IMG_20210728_205312.jpg
```
## 导出mmyolo系列模型,onnx tensorrt版本
### yolov5
就还是用mmyolo初体验中，猫检测器的转换命令
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
	--backend 2 \
	--pre-topk 1000 \
	--keep-topk 100 \
	--iou-threshold 0.65 \
	--score-threshold 0.25
```
### yolov6
```bash
pip install openmim==0.3.7
workdir="yolov6_n_syncbn_fast_8xb32-400e_coco"
mim download mmyolo --config $workdir --dest ./work_dirs/$workdir
pth_path=`ls ./work_dirs/$workdir/*.pth`

python projects/easydeploy/tools/export.py \
	work_dirs/$workdir/$workdir.py \
	${pth_path} \
	--work-dir work_dirs/$workdir \
    --img-size 640 640 \
    --batch 1 \
    --device cpu \
    --simplify \
	--opset 11 \
	--backend 2 \
	--pre-topk 30000 \
	--keep-topk 300 \
	--iou-threshold 0.65 \
	--score-threshold 0.001

netron work_dirs/yolov6_n_syncbn_fast_8xb32-400e_coco/end2end.onnx
ssh -L 8080:localhost:8080 student001@10.20.30.160
# backend 2 是trt8, backend 3 是trt7;
# yolov6 用trt8, 检测头是EfficientNMS_TRT; 用 trt7 是BatchedNMSDynamic_TRT;
```
### yolov7
```bash
workdir="yolov7_tiny_syncbn_fast_8x16b-300e_coco"
mim download mmyolo --config $workdir --dest ./work_dirs/$workdir

pth_path=`ls ./work_dirs/$workdir/*.pth`

python projects/easydeploy/tools/export.py \
	work_dirs/$workdir/$workdir.py \
	${pth_path} \
	--work-dir work_dirs/$workdir \
    --img-size 640 640 \
    --batch 1 \
    --device cpu \
    --simplify \
	--opset 11 \
	--backend 2 \
	--pre-topk 1000 \
	--keep-topk 100 \
	--iou-threshold 0.65 \
	--score-threshold 0.25
```
### yolov8
```bash
workdir="yolov8_n_syncbn_fast_8xb16-500e_coco"
mim download mmyolo --config $workdir --dest ./work_dirs/$workdir

pth_path=`ls ./work_dirs/$workdir/*.pth`

python projects/easydeploy/tools/export.py \
	work_dirs/$workdir/$workdir.py \
	${pth_path} \
	--work-dir work_dirs/$workdir \
    --img-size 640 640 \
    --batch 1 \
    --device cpu \
    --simplify \
	--opset 11 \
	--backend 2 \
	--pre-topk 1000 \
	--keep-topk 100 \
	--iou-threshold 0.65 \
	--score-threshold 0.25
```
### yolovx
```bash
workdir="yolox_tiny_fast_8xb8-300e_coco"
mim download mmyolo --config $workdir --dest ./work_dirs/$workdir

pth_path=`ls ./work_dirs/$workdir/*.pth`
python projects/easydeploy/tools/export.py \
	work_dirs/$workdir/$workdir.py \
	${pth_path} \
	--work-dir work_dirs/$workdir \
    --img-size 640 640 \
    --batch 1 \
    --device cpu \
    --simplify \
	--opset 11 \
	--backend 3 \
	--pre-topk 1000 \
	--keep-topk 100 \
	--iou-threshold 0.65 \
	--score-threshold 0.25
```
### ppyoloe_plus
```bash
workdir="ppyoloe_plus_s_fast_8xb8-80e_coco"
mim download mmyolo --config $workdir --dest ./work_dirs/$workdir

pth_path=`ls ./work_dirs/$workdir/*.pth`
python projects/easydeploy/tools/export.py \
	work_dirs/$workdir/$workdir.py \
	${pth_path} \
	--work-dir work_dirs/$workdir \
    --img-size 640 640 \
    --batch 1 \
    --device cpu \
    --simplify \
	--opset 11 \
	--backend 2 \
	--pre-topk 1000 \
	--keep-topk 100 \
	--iou-threshold 0.65 \
	--score-threshold 0.25
```
### rtmdet
```bash
workdir="rtmdet_tiny_syncbn_fast_8xb32-300e_coco"
mim download mmyolo --config $workdir --dest ./work_dirs/$workdir

pth_path=`ls ./work_dirs/$workdir/*.pth`
python projects/easydeploy/tools/export.py \
	work_dirs/$workdir/$workdir.py \
	${pth_path} \
	--work-dir work_dirs/$workdir \
    --img-size 640 640 \
    --batch 1 \
    --device cpu \
    --simplify \
	--opset 11 \
	--backend 2 \
	--pre-topk 1000 \
	--keep-topk 100 \
	--iou-threshold 0.65 \
	--score-threshold 0.25
```
### rtmdet 旋转框
```bash
workdir="rtmdet-r_tiny_fast_1xb8-36e_dota"
mim download mmyolo --config $workdir --dest ./work_dirs/$workdir


pth_path=`ls ./work_dirs/$workdir/*.pth`
python projects/easydeploy/tools/export.py \
	work_dirs/$workdir/$workdir.py \
	${pth_path} \
	--work-dir work_dirs/$workdir \
    --img-size 640 640 \
    --batch 1 \
    --device cpu \
    --simplify \
	--opset 11 \
	--backend 2 \
	--pre-topk 1000 \
	--keep-topk 100 \
	--iou-threshold 0.65 \
	--score-threshold 0.25
```
这个导出有问题。
## 初步导出Tensorrt
```bash
workdir="yolov6_n_syncbn_fast_8xb32-400e_coco"
mim download mmyolo --config $workdir --dest ./work_dirs/$workdir

pth_path=`ls ./work_dirs/$workdir/*.pth`

config_file="work_dirs/$workdir/$workdir.py"

python projects/easydeploy/tools/export.py \
	$config_file \
	${pth_path} \
	--work-dir work_dirs/$workdir \
    --img-size 640 640 \
    --batch 1 \
    --device cpu \
    --simplify \
	--opset 11 \
	--backend 2 \
	--pre-topk 1000 \
	--keep-topk 300 \
	--iou-threshold 0.65 \
	--score-threshold 0.3

python projects/easydeploy/tools/build_engine.py \
     ./work_dirs/$workdir/end2end.onnx \
    --img-size 640 640 \
    --device cuda:0

python projects/easydeploy/tools/image-demo.py \
    demo/dog.jpg \
    $config_file \
    ./work_dirs/$workdir/end2end.engine \
    --device cuda:0
```
## TensorRT模型精度的验证

# 复现论文
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
# tips
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