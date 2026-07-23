# Open-GroundingDino：海光 BW100 训练适配

本仓库基于 [Open-GroundingDino](https://github.com/longzw1997/Open-GroundingDino)，完成了海光 BW100（DCU）环境下的训练适配，包含 DCU 版 Multi-Scale Deformable Attention 自定义算子、COCO/ODVG 数据准备以及 GroundingDINO Swin-T、Swin-B 单卡与多卡训练脚本。

当前验证环境：

- DCU：海光 BW100
- 基础镜像：`harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.9.0-ubuntu22.04-dtk26.04-py3.10`
- DTK：26.04
- Python：3.10
- PyTorch：2.9.0

## 1. 获取代码

```bash
git clone https://github.com/mantuoluozk/Open-GroundingDino-DCU.git Open-GroundingDino
cd Open-GroundingDino
```

## 2. 创建并进入容器

下面的脚本会复用同名容器；不存在时创建容器。项目和数据统一放在宿主机 `/data/zk` 下，并挂载到容器中的相同路径。

```bash
#!/bin/bash
set -e

CONTAINER_NAME="zk-opengd-0722"
IMAGE_NAME="harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.9.0-ubuntu22.04-dtk26.04-py3.10"

if docker inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    if [[ "$(docker inspect "$CONTAINER_NAME" --format '{{.State.Status}}')" == "exited" ]]; then
        docker start "$CONTAINER_NAME"
    fi
    exec docker exec -it "$CONTAINER_NAME" /bin/bash
fi

docker run -dit \
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

exec docker exec -it "$CONTAINER_NAME" /bin/bash
```

进入项目目录：

```bash
cd /data/zk/Open-GroundingDino
```

## 3. 安装依赖

基础镜像已提供 PyTorch、torchvision、Transformers、timm、NumPy、SciPy 和 OpenCV，`requirements.txt` 已避免重复安装这些核心组件。

```bash
pip install -r requirements.txt
pip install numpy==1.25.0 --force-reinstall \
    -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 4. 编译 DCU 自定义算子

本仓库的 `models/GroundingDINO/ops/setup.py` 及 `src/hip/` 已包含 BW100 适配代码。

```bash
source /opt/dtk/cuda/cuda-12/env.sh

# 避免 CUDA 兼容头与编译过程冲突；同一容器只需处理一次。
if [[ -d /opt/dtk/cuda/cuda-12/include ]]; then
    cp -a /opt/dtk/cuda/cuda-12/include \
        /opt/dtk/cuda/cuda-12/include-bak
    rm -rf /opt/dtk/cuda/cuda-12/include
fi

cd /data/zk/Open-GroundingDino/models/GroundingDINO/ops
python setup.py build install
python test.py
```

测试中应看到前向精度和梯度检查均为 `True`。以下信息在当前环境中是编译器提示，已验证不会阻止训练继续执行：

```text
Launch params (1024, 1, 1) are larger than launch bounds (256) ...
```

## 5. 准备 COCO 2017 数据

将 COCO 2017 解压为：

```text
/data/zk/coco_2017/
├── train2017/
├── val2017/
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

把训练标注转换为 ODVG：

```bash
cd /data/zk/Open-GroundingDino/tools

python coco2odvg.py \
    --input /data/zk/coco_2017/annotations/instances_train2017.json \
    --output /data/zk/coco_2017/annotations/coco2017_train_odvg.jsonl \
    --idmap coco2017

python -c "from coco2odvg import dump_label_map; dump_label_map(output='/data/zk/coco_2017/annotations/coco2017_label_map.json')"
```

`config/datasets_od_example.json` 已按上述路径配置。训练集使用 ODVG，验证集使用 COCO 标注。

## 6. 准备预训练权重和 BERT

权重与 BERT 文件不提交到 Git，请放入：

```text
/data/zk/Open-GroundingDino/ckpts/
├── groundingdino_swint_ogc.pth
├── groundingdino_swinb_cogcoor.pth
└── bert-base-uncased/
```

下载 BERT：

```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download bert-base-uncased \
    --local-dir /data/zk/Open-GroundingDino/ckpts/bert-base-uncased
```

## 7. 训练 Swin-T

Swin-T 配置为 `config/cfg_coco.py`：

```python
backbone = 'swin_T_224_1k'
batch_size = 4
```

选择一张空闲 DCU 并启动训练：

```bash
cd /data/zk/Open-GroundingDino
export HIP_VISIBLE_DEVICES=5

PRETRAIN_MODEL_PATH=/data/zk/Open-GroundingDino/ckpts/groundingdino_swint_ogc.pth \
TEXT_ENCODER_TYPE=/data/zk/Open-GroundingDino/ckpts/bert-base-uncased \
bash train_dist.sh \
    1 \
    config/cfg_coco.py \
    config/datasets_od_example.json \
    ./logs_swint
```

## 8. 训练 Swin-B

Swin-B 使用独立配置 `config/cfg_coco_swinb.py`，避免修改或覆盖 Swin-T 配置：

```python
backbone = 'swin_B_384_22k'
batch_size = 4
```

启动训练：

```bash
cd /data/zk/Open-GroundingDino
export HIP_VISIBLE_DEVICES=5

PRETRAIN_MODEL_PATH=/data/zk/Open-GroundingDino/ckpts/groundingdino_swinb_cogcoor.pth \
TEXT_ENCODER_TYPE=/data/zk/Open-GroundingDino/ckpts/bert-base-uncased \
bash train_dist_swinb.sh \
    1 \
    config/cfg_coco_swinb.py \
    config/datasets_od_example.json \
    ./logs_swinb
```

当前服务器已使用单卡 `batch_size=4` 完成 Swin-B 训练验证。Swin-B 显存占用显著高于 Swin-T；如果更换输入尺度、数据增强或设备后出现显存不足，可先把该值降为 1，再根据显存余量逐步增加。

### 权重必须与配置匹配

```text
cfg_coco.py        + groundingdino_swint_ogc.pth
cfg_coco_swinb.py  + groundingdino_swinb_cogcoor.pth
```

如果 Swin-B 配置误加载 Swin-T 权重，会出现 `size mismatch`，例如 checkpoint 通道为 384/768，而当前模型通道为 512/1024。`strict=False` 不能忽略同名参数的形状不一致。

## 9. 参数含义、多卡和断点续训

命令中的第一个数字表示训练进程数（可见 DCU 数量），不是 batch size。全局 batch size 为：

```text
配置文件中的 batch_size × DCU 数量
```

例如使用 4 张卡训练 Swin-T：

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3

PRETRAIN_MODEL_PATH=/data/zk/Open-GroundingDino/ckpts/groundingdino_swint_ogc.pth \
TEXT_ENCODER_TYPE=/data/zk/Open-GroundingDino/ckpts/bert-base-uncased \
bash train_dist.sh 4 \
    config/cfg_coco.py \
    config/datasets_od_example.json \
    ./logs_swint_4dcu
```

如果输出目录中已有 `checkpoint.pth`，`main.py` 会自动断点续训。开始新实验时必须使用新的空目录，Swin-T 和 Swin-B 不得共用输出目录。

训练脚本中不能添加 `--eval`；该参数只用于验证。

## 10. 验证模型

验证 Swin-T：

```bash
export HIP_VISIBLE_DEVICES=5

PRETRAIN_MODEL_PATH=/data/zk/Open-GroundingDino/ckpts/groundingdino_swint_ogc.pth \
TEXT_ENCODER_TYPE=/data/zk/Open-GroundingDino/ckpts/bert-base-uncased \
bash test_dist.sh 1 \
    config/cfg_coco.py \
    config/datasets_od_example.json \
    ./eval_swint
```

验证 Swin-B：

```bash
export HIP_VISIBLE_DEVICES=5

PRETRAIN_MODEL_PATH=/data/zk/Open-GroundingDino/ckpts/groundingdino_swinb_cogcoor.pth \
TEXT_ENCODER_TYPE=/data/zk/Open-GroundingDino/ckpts/bert-base-uncased \
bash test_dist.sh 1 \
    config/cfg_coco_swinb.py \
    config/datasets_od_example.json \
    ./eval_swinb
```

## 11. 已验证结果

在 BW100 单卡上已完成以下验证：

- DCU 自定义算子前向精度测试通过；
- `D=30/32/64/71` 梯度检查通过；
- Swin-T、batch size 4 完成前向、反向传播和多 iteration 训练；
- Swin-B 权重与 `swin_B_384_22k` 模型参数完全匹配，并已完成训练验证。

## 12. 注意事项

- `ckpts/`、训练日志、checkpoint、编译目录和 Python 缓存不应提交到 Git。
- `torch.distributed.launch` 会输出弃用警告，但当前环境仍可正常训练；后续可迁移到 `torchrun`。
- 首个 iteration 包含初始化开销，耗时通常明显高于后续 iteration。
- 修改 backbone 后必须同步切换预训练权重与输出目录。

## License

本项目沿用上游仓库的 [Apache License 2.0](LICENSE)。
