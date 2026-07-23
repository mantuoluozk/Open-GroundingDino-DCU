镜像：harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.9.0-ubuntu22.04-dtk26.04-py3.10

创建容器：
```
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

下载代码：
```
git clone https://github.com/longzw1997/Open-GroundingDino.git && cd Open-GroundingDino/
```
注释掉requirements.txt中的镜像中已有的库
```
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
安装依赖
```
pip install -r requirements.txt
```
降级numpy
```
pip install numpy==1.25.0 --force-reinstall -i https://pypi.tuna.tsinghua.edu.cn/simple
```

修改源码Open-GroundingDino/models/GroundingDINO/ops/setup.py
目前仓库中已经是修改后的版本

配置环境变量
```
source /opt/dtk/cuda/cuda-12/env.sh
cp -r /opt/dtk/cuda/cuda-12/include /opt/dtk/cuda/cuda-12/include-bak
rm -rf /opt/dtk/cuda/cuda-12/include
export HF_ENDPOINT=https://hf-mirror.com
```
项目编译
```
cd models/GroundingDINO/ops
python setup.py build install
```

验证
python test.py

root@node3:/data/zk/Open-GroundingDino/models/GroundingDINO/ops# python test.py
Launch params (1024, 1, 1) are larger than launch bounds (256) for kernel _Z31ms_deformable_im2col_gpu_kernelIdEviPKT_PKlS4_S2_S2_iiiiiiiPS0_ please add __launch_bounds__ to kernel define or use --gpu-max-threads-per-block recompile program ! 
* True check_forward_equal_with_pytorch_double: max_abs_err 8.67e-19 max_rel_err 2.35e-16
* True check_forward_equal_with_pytorch_float: max_abs_err 4.66e-10 max_rel_err 1.13e-07
* True check_gradient_numerical(D=30)
* True check_gradient_numerical(D=32)
* True check_gradient_numerical(D=64)
* True check_gradient_numerical(D=71)

下载数据集并进行转换
```
wget -c http://10.17.26.232:8000/ISO/AI/02.DataSets/coco2017_official/train2017.zip

wget -c http://10.17.26.232:8000/ISO/AI/02.DataSets/coco2017_official/val2017.zip

wget -c http://10.17.26.232:8000/ISO/AI/02.DataSets/coco2017_official/annotations_trainval2017.zip

unzip train2017.zip -d coco_2017/
unzip val2017.zip -d coco_2017/
unzip annotations_trainval2017.zip -d coco_2017/
```

转换训练集标注（生成 coco2017_train_odvg.jsonl）
```
cd tools
python coco2odvg.py \
--input /data/zk/coco_2017/annotations/instances_train2017.json \
--output /data/zk/coco_2017/annotations/coco2017_train_odvg.jsonl \
--idmap coco2017
```

转换验证集标注（生成 coco2017_val_odvg.jsonl）
```
python coco2odvg.py \
--input /data/zk/coco_2017/annotations/instances_val2017.json \
--output /data/zk/coco_2017/annotations/coco2017_val_odvg.jsonl \
--idmap coco2017
```

生成标签映射文件 coco2017_label_map.json
```
python -c "
import json
from coco2odvg import dump_label_map
dump_label_map(output='/data/zk/coco_2017/annotations/coco2017_label_map.json')
"
```

最终数据结构
```
/workspace/datav/zk/coco_2017/
├── train2017/ # 解压 train2017.zip 到这里
├── val2017/ # 解压 val2017.zip 到这里
└── annotations/ # 解压 annotations_trainval2017.zip + 放置ODVG标注/标签映射
├── instances_train2017.json # 官方检测标注
├── instances_val2017.json # 官方检测标注
├── coco2017_train_odvg.jsonl # ODVG格式训练标注
├── coco2017_val_odvg.jsonl # ODVG格式验证标注
└── coco2017_label_map.json # 标签映射文件
```

更改配置文件config/datasets_od_example.json
```
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

下载预训练模型groundingdino_swint_ogc.pth或者groundingdino_swinb_cogcoor.pth到项目的ckpts路径，模型在Open-GroundingDino或者GroundingDino的GitHub中可以找到。

下载bert-base-uncased到ckpts路径下
```
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download bert-base-uncased --local-dir /data/zk/Open-GroundingDino/ckpts/bert-base-uncased
```

train_dist.sh中更改模型路径为实际路径

```
PRETRAIN_MODEL_PATH=${PRETRAIN_MODEL_PATH:-"/data/zk/Open-GroundingDino/ckpts/groundingdino_swint_ogc.pth"}

TEXT_ENCODER_TYPE=${TEXT_ENCODER_TYPE:-"/data/zk/Open-GroundingDino/ckpts/bert-base-uncased"}
```

运行训练脚本
```
bash train_dist.sh 1 config/cfg_coco.py config/datasets_od_example.json ./logs_swint
```
如果要训练swinb，需要修改两处地方，cfg_coco.py中改成swinb，train_dist.sh中模型权重改成winb的
```
bash train_dist_swinb.sh 1 config/cfg_coco_swinb.py config/datasets_od_example.json ./logs_swinb
```

验证
更改test_dist.sh中模型路径，同样，如果要验证swinb，需要修改test_dist.sh和cfg_coco.py中模型参数

```
        --pretrain_model_path /data/zk/Open-GroundingDino/ckpts/groundingdino_swinb_cogcoor.pth \
        --options text_encoder_type=/data/zk/Open-GroundingDino/ckpts/bert-base-uncased
```

运行
```
bash test_dist.sh 1 config/cfg_coco.py config/datasets_od_example.json ./logs
```
