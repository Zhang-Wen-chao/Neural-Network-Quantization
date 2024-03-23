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
跟着追乐大兵的视频，看完了p13。这个系列的视频，重在分析量化方法。不在实现某个算法。

我感觉我好像搞错了重点，我的毕设名字很好啊，管它什么呢，我先部署了再说。哪有什么提升标准。现在不应该使劲研究追乐大兵的视频了。

现在应该把重点放在王鑫宇的仓库上啊，部署成功一个，就开始写这一部分的论文。
## 评估指标
评估YOLO模型在PyTorch和TensorRT下的输出差异通常涉及比较模型在两种框架下的性能和精度。这种比较有助于确保模型转换过程（从PyTorch到TensorRT）没有引入显著误差，同时保持或提升推理速度。以下是一些关键的评估指标：

1. 精度差异
IoU（Intersection over Union）：评估预测边界框与真实边界框之间的重叠程度。IoU的值越高，表示预测越准确。

mAP（mean Average Precision）：对于检测任务，mAP是最常用的评估指标之一。它计算不同阈值下的平均精度，并对所有类别取平均值。比较PyTorch和TensorRT模型的mAP可以直观地显示模型性能是否在转换后有所下降。

2. 性能差异
推理时间：衡量模型完成一次推理所需的时间。TensorRT优化的目标之一就是减少模型的推理时间，提高处理速度。

FPS（Frames Per Second）：每秒可以处理的图像帧数。高FPS值意味着模型可以更快地处理输入图像，这在实时应用中尤为重要。

3. 资源使用差异
GPU/CPU使用率：在推理过程中GPU和CPU的使用情况，可以通过专门的监测工具来跟踪。

内存消耗：推理过程中的内存使用情况，包括GPU和CPU内存。优化的模型应该更有效地使用内存。

评估步骤
准备相同的数据集：确保PyTorch和TensorRT模型使用完全相同的数据集进行评估。

运行推理：在两种框架下运行模型，记录输出结果、推理时间、以及任何其他相关的性能指标。

计算差异：使用上述指标比较两个模型的输出，评估精度和性能的差异。

统计分析：如果有必要，进行统计分析来判断差异是否具有统计学意义。

注意事项
确保TensorRT优化后的模型保持原有结构的完整性。有时，为了优化性能，一些转换可能会更改模型的部分细节，可能导致精度下降。

在进行比较时，确保两个模型都在相同的硬件和软件条件下运行。

通过综合考虑这些指标和评估步骤，你可以全面地了解模型从PyTorch转换到TensorRT后的性能和精度变化。
## 王鑫宇仓库
首先拉最新的仓库，然后查看所有yolo系列支持的tensorrt版本，固定一下各种软件的版本。
tensorrt8比较多的话，就都用8,不然都不像是2024年的论文。
### yolovp
(base) /dataset01/zwc/tensorrtx/yolop/build (master ✘)✹✭ ᐅ ./yolop -d yolop.trt /dataset01/zwc/tensorrtx/yolop/YOLOP/inference/images
140ms
3ms
3ms
3ms
3ms
3ms
### yolov3
### yolov4
### yolov5
(base) /dataset01/zwc/tensorrtx/yolov5/tensorrtx/yolov5/build (yolov5-v7.0 ✘)✹ ᐅ ./yolov5_det -d yolov5s_det.engine ../images     
inference time: 122ms
inference time: 0ms

现在的问题，yolov5s，部署到tensorrt上后，怎么生成result.json文件呢？


mAPval values are for single-model single-scale on COCO val2017 dataset.
#### yolov5s
(base) /dataset01/zwc/tensorrtx/yolov5/yolov5 (v7.0 ✘)✭ ᐅ python val.py --save-json --data /dataset01/zwc/tensorrtx/yolov5/yolov5/data/coco.yaml   
val: data=/dataset01/zwc/tensorrtx/yolov5/yolov5/data/coco.yaml, weights=yolov5s.pt, batch_size=32, imgsz=640, conf_thres=0.001, iou_thres=0.6, max_det=300, task=val, device=, workers=8, single_cls=False, augment=False, verbose=False, save_txt=False, save_hybrid=False, save_conf=False, save_json=True, project=runs/val, name=exp, exist_ok=False, half=False, dnn=False
YOLOv5 🚀 v7.0-0-g915bbf29 Python-3.9.13 torch-2.1.2+cu121 CUDA:0 (NVIDIA GeForce RTX 4090, 24217MiB)

Fusing layers... 
YOLOv5s summary: 213 layers, 7225885 parameters, 0 gradients, 16.4 GFLOPs
val: Scanning /dataset01/zwc/tensorrtx/yolov5/datasets/coco/val2017... 4952 images, 48 backgrounds, 0 corrupt: 100%|██████████| 5000/5000 00:00
val: New cache created: /dataset01/zwc/tensorrtx/yolov5/datasets/coco/val2017.cache
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 157/157 00:38
                   all       5000      36335       0.67       0.52      0.565      0.371
Speed: 0.1ms pre-process, 1.0ms inference, 1.2ms NMS per image at shape (32, 3, 640, 640)

Evaluating pycocotools mAP... saving runs/val/exp8/yolov5s_predictions.json...
loading annotations into memory...
Done (t=0.39s)
creating index...
index created!
Loading and preparing results...
DONE (t=4.81s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=53.68s).
Accumulating evaluation results...
DONE (t=11.77s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.374
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.571
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.402
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.211
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.423
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.490
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.311
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.516
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.566
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.378
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.625
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.723
Results saved to runs/val/exp8
### yolov6
王鑫宇没有yolov6，追乐有啊！！没有功夫是白费的。而且追乐详细优化了yolov6，但就是没有他那么老的显卡配tensorrt了。

所以还是暂时先放下这部分。
### yolov7
### yolov8
### yolov9
2024年3月11日，看到群里有人完成了yolov9两个子模型的部署,牛逼。
# 追乐大兵的b站视频
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
	--pre-topk 30000 \
	--keep-topk 300 \
	--iou-threshold 0.65 \
	--score-threshold 0.001

python projects/easydeploy/tools/build_engine.py \
     ./work_dirs/$workdir/end2end.onnx \
    --img-size 640 640 \
    --device cuda:0

python self_study_scripts/01cmp_output_with_torch_tensorrt.py \
    demo/dog.jpg \
    $config_file \
    ./work_dirs/$workdir/end2end.engine \
    $pth_path \
    --device cuda:0

python self_study_scripts/01cmp_output_with_torch_tensorrt.py \
    /dataset01/coco2017/val2017/ \
    $config_file \
    ./work_dirs/$workdir/end2end.engine \
    $pth_path \
    --device cuda:0 \
	--result-json-path /dataset01/zwc/mmyolo-hb/mmyolo/self_study_scripts/tensort-1.json
这个命令需要很久，要推理5000个呢。
```
目标：
1. 探索相同输入下tensorrt和pytorch两个框架的输出有无不同
2. 针对 coco-val 数据集下的精度，tensorrt和pytorch得到的有何区别？

得到yolov6 pytorch版本输出

```bash
# 测试精度， 为加快速度还请去掉'/dataset01/zwc/mmyolo-hb/mmyolo/work_dirs/yolov6_n_syncbn_fast_8xb32-400e_coco/yolov6_n_syncbn_fast_8xb32-400e_coco.py'文件中的 visualization hook
# visualization=dict(type='mmdet.DetVisualizationHook')
workdir="yolov6_n_syncbn_fast_8xb32-400e_coco"
pth_path=`ls ./work_dirs/$workdir/*.pth`
python tools/test.py work_dirs/$workdir/$workdir.py \
                     $pth_path \
					 --json-prefix ./self_study_scripts/yolov6_n_coco_result

python ./self_study_scripts/02evel_trt_json_output.py
```
第一个是pytorch，第二个是tensorrt.
loading annotations into memory...
Done (t=0.50s)
creating index...
index created!
number of coco_classes: 80
Loading and preparing results...
DONE (t=12.20s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=106.48s).
Accumulating evaluation results...
DONE (t=26.75s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.362
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.516
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.392
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.166
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.401
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.521
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.312
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.522
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.574
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.340
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.644
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.773

Loading and preparing results...
DONE (t=5.28s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=107.03s).
Accumulating evaluation results...
DONE (t=26.48s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.361
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.516
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.389
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.167
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.399
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.519
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.311
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.521
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.573
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.338
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.642
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.771
## 精度没有百分之百对齐的原因
```bash
python demo/image_demo.py demo/dog.jpg ./work_dirs/yolov6_n_syncbn_fast_8xb32-400e_coco/yolov6_n_syncbn_fast_8xb32-400e_coco.py ./work_dirs/yolov6_n_syncbn_fast_8xb32-400e_coco/yolov6_n_syncbn_fast_8xb32-400e_coco_20221030_202726-d99b2e82.pth
```
出现了版本差异导致的问题，后面的操作都跟不上了。这种分析也并不必写在论文里面。
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