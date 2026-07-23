# Open-GroundingDino 适配海光 BW100 训练

本文档记录 Open-GroundingDino 在海光 BW100 上的完整环境配置、DCU 算子编译、COCO 数据转换以及 Swin-T/Swin-B 训练与验证流程。

基础镜像：

```text
harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.9.0-ubuntu22.04-dtk26.04-py3.10
```

## 1. 创建并进入容器

以下脚本会检查 `zk-opengd-0722` 是否存在：容器已停止时重新启动，不存在时使用指定镜像创建。

```bash
#!/bin/bash

CONTAINER_NAME="zk-opengd-0722"
IMAGE_NAME="harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.9.0-ubuntu22.04-dtk26.04-py3.10"
# 检查容器状态函数

check_container_state() {

    docker inspect "$CONTAINER_NAME" --format '{{.State.Status}}' 2>/dev/null

}

# 主逻辑
if container_state=$(check_container_state); then
    echo "CONTAINER state:====== $container_state ======"
    if [[ "$container_state" == "exited" ]]; then
        if ! docker start "$CONTAINER_NAME"; then
            echo "Failed to start container $CONTAINER_NAME" >&2
            exit 1
        fi
    fi
    exec docker exec -it "$CONTAINER_NAME" /bin/bash
else
    if ! docker run -dit \
        --network=host \
        --name="$CONTAINER_NAME" \
        --privileged \
        --device=/dev/kfd \
        --device=/dev/dri \
        --device=/dev/mkfd \
        --ipc=host \
        --shm-size=256G \
        --group-add video \
        --cap-add=SYS_PTRACE \
        --security-opt seccomp=unconfined \
        -u root \
        --ulimit stack=-1:-1 \
        --ulimit memlock=-1:-1 \
        -v /dev/infiniband:/dev/infiniband \
        -v /opt/hyhal:/opt/hyhal:ro \
        -v /models:/models \
        -v /data/zk:/data/zk \
        "$IMAGE_NAME"
    then
        echo "Failed to create container $CONTAINER_NAME" >&2
        exit 1
    fi
    exec docker exec -it "$CONTAINER_NAME" /bin/bash
fi
```

## 2. 下载代码

```bash
git clone https://github.com/longzw1997/Open-GroundingDino.git && cd Open-GroundingDino/
```

## 3. 安装依赖并编译 DCU 算子

### 3.1 调整依赖

注释掉 `requirements.txt` 中基础镜像已经提供的库，内容如下：

```text
cython
submitit
# scipy
termcolor
addict
yapf==0.40.1
# timm
# torch
# torchvision
# transformers
# numpy
# opencv-python
supervision==0.6.0
pycocotools
pyyaml>3.10
colorlog
jsonlines
```
安装其余依赖：

```bash
pip install -r requirements.txt
```

将 NumPy 调整为已验证版本：

```bash
pip install numpy==1.25.0 --force-reinstall -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3.2 配置编译环境

DCU 适配涉及以下文件：

```text
Open-GroundingDino/models/GroundingDINO/ops/setup.py
```

当前项目仓库中已经是修改后的版本。

配置环境变量并处理 DTK CUDA 兼容头文件：

```bash
source /opt/dtk/cuda/cuda-12/env.sh
cp -r /opt/dtk/cuda/cuda-12/include /opt/dtk/cuda/cuda-12/include-bak
rm -rf /opt/dtk/cuda/cuda-12/include
export HF_ENDPOINT=https://hf-mirror.com
```
### 3.3 编译与验证

编译项目自定义算子：

```bash
cd models/GroundingDINO/ops
python setup.py build install
```

执行验证：

```bash
python test.py
```

实测输出：

```text
root@node3:/data/zk/Open-GroundingDino/models/GroundingDINO/ops# python test.py
Launch params (1024, 1, 1) are larger than launch bounds (256) for kernel _Z31ms_deformable_im2col_gpu_kernelIdEviPKT_PKlS4_S2_S2_iiiiiiiPS0_ please add __launch_bounds__ to kernel define or use --gpu-max-threads-per-block recompile program ! 
* True check_forward_equal_with_pytorch_double: max_abs_err 8.67e-19 max_rel_err 2.35e-16
* True check_forward_equal_with_pytorch_float: max_abs_err 4.66e-10 max_rel_err 1.13e-07
* True check_gradient_numerical(D=30)
* True check_gradient_numerical(D=32)
* True check_gradient_numerical(D=64)
* True check_gradient_numerical(D=71)
```

## 4. 下载并转换 COCO 2017 数据集

### 4.1 下载与解压

```bash
wget -c http://10.17.26.232:8000/ISO/AI/02.DataSets/coco2017_official/train2017.zip

wget -c http://10.17.26.232:8000/ISO/AI/02.DataSets/coco2017_official/val2017.zip

wget -c http://10.17.26.232:8000/ISO/AI/02.DataSets/coco2017_official/annotations_trainval2017.zip

unzip train2017.zip -d coco_2017/
unzip val2017.zip -d coco_2017/
unzip annotations_trainval2017.zip -d coco_2017/
```

### 4.2 转换训练集标注

生成 `coco2017_train_odvg.jsonl`：

```bash
cd tools
python coco2odvg.py \
--input /data/zk/coco_2017/annotations/instances_train2017.json \
--output /data/zk/coco_2017/annotations/coco2017_train_odvg.jsonl \
--idmap coco2017
```

### 4.3 转换验证集标注

生成 `coco2017_val_odvg.jsonl`：

```bash
python coco2odvg.py \
--input /data/zk/coco_2017/annotations/instances_val2017.json \
--output /data/zk/coco_2017/annotations/coco2017_val_odvg.jsonl \
--idmap coco2017
```

### 4.4 生成标签映射文件

生成 `coco2017_label_map.json`：

```bash
python -c "
import json
from coco2odvg import dump_label_map
dump_label_map(output='/data/zk/coco_2017/annotations/coco2017_label_map.json')
"
```

### 4.5 检查数据结构

最终数据目录结构如下：

```text
/data/zk/coco_2017/
├── train2017/ # 解压 train2017.zip 到这里
├── val2017/ # 解压 val2017.zip 到这里
└── annotations/ # 解压 annotations_trainval2017.zip + 放置ODVG标注/标签映射
    ├── instances_train2017.json # 官方检测标注
    ├── instances_val2017.json # 官方检测标注
    ├── coco2017_train_odvg.jsonl # ODVG格式训练标注
    ├── coco2017_val_odvg.jsonl # ODVG格式验证标注
    └── coco2017_label_map.json # 标签映射文件
```

### 4.6 配置数据集路径

更改 `config/datasets_od_example.json`：

```json
{
  "train": [
    {
      "root": "/data/zk/coco_2017/train2017",
      "anno": "/data/zk/coco_2017/annotations/coco2017_train_odvg.jsonl",
      "label_map": "/data/zk/coco_2017/annotations/coco2017_label_map.json",
      "dataset_mode": "odvg"
    }
  ],
  "val": [
    {
      "root": "/data/zk/coco_2017/val2017",
      "anno": "config/instances_val2017.json",
      "label_map": null,
      "dataset_mode": "coco"
    }
  ]
}
```

## 5. 准备预训练模型与 BERT

下载 `groundingdino_swint_ogc.pth` 或 `groundingdino_swinb_cogcoor.pth` 到项目的 `ckpts` 目录。模型可在 Open-GroundingDino 或 GroundingDINO 的 GitHub 项目中找到。

下载 `bert-base-uncased` 到 `ckpts` 目录：

```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download bert-base-uncased --local-dir /data/zk/Open-GroundingDino/ckpts/bert-base-uncased
```

## 6. 训练

### 6.1 配置训练脚本路径

在 `train_dist.sh` 中将模型和 BERT 路径改为实际路径：

```bash
PRETRAIN_MODEL_PATH=${PRETRAIN_MODEL_PATH:-"/data/zk/Open-GroundingDino/ckpts/groundingdino_swint_ogc.pth"}

TEXT_ENCODER_TYPE=${TEXT_ENCODER_TYPE:-"/data/zk/Open-GroundingDino/ckpts/bert-base-uncased"}
```

### 6.2 训练 Swin-T

```bash
bash train_dist.sh 1 config/cfg_coco.py config/datasets_od_example.json ./logs_swint
```

### 6.3 训练 Swin-B

训练 Swin-B 时，需要在 `cfg_coco_swinb.py` 中使用 Swin-B backbone，并在 `train_dist_swinb.sh` 中使用 Swin-B 权重：

```bash
bash train_dist_swinb.sh 1 config/cfg_coco_swinb.py config/datasets_od_example.json ./logs_swinb
```

## 7. 验证

验证 Swin-B 时，需要同步修改 `test_dist.sh` 中的权重路径，并使用对应的 Swin-B 配置：

```bash
        --pretrain_model_path /data/zk/Open-GroundingDino/ckpts/groundingdino_swinb_cogcoor.pth \
        --options text_encoder_type=/data/zk/Open-GroundingDino/ckpts/bert-base-uncased
```

运行验证脚本：

```bash
bash test_dist.sh 1 config/cfg_coco.py config/datasets_od_example.json ./logs
```
