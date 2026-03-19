# CLAUDE.md

此文件为 Claude Code (claude.ai/code) 在本仓库中工作时提供指导。

## 项目概述

本仓库包含人体姿态估计和基于骨架的动作识别研究实现：

- **VIBE**: Video Inference for Human Body Pose and Shape Estimation (CVPR 2020) - 从视频中预测 SMPL 身体模型参数
- **CTR-GCN**: Channel-wise Topology Refinement Graph Convolution (ICCV 2021) - 基于骨架的动作识别

## 命令

### CTR-GCN

```bash
# 在 NTU RGB+D 120 跨被试数据集上训练
cd CTR_GCN
python main.py --config config/nturgbd120-cross-subject/default.yaml --work-dir work_dir/ntu120/csub/ctrgcn --device 0

# 测试训练好的模型
python main.py --config <work_dir>/config.yaml --work-dir <work_dir> --phase test --save-score True --weights <work_dir>/xxx.pt --device 0

# 集成多个模态
python ensemble.py --datasets ntu120/xsub --joint-dir work_dir/... --bone-dir work_dir/... --joint-motion-dir work_dir/... --bone-motion-dir work_dir/...
```

### VIBE

```bash
# 运行视频 demo
cd VIBE
python demo.py --vid_file sample_video.mp4 --output_folder output/ --display

# 训练
python train.py --cfg configs/config.yaml
```

## 架构

### CTR-GCN (`CTR_GCN/`)
- `model/ctrgcn.py` - CTR-GCN 模型实现
- `model/baseline.py` - 基线模型
- `graph/` - NTU 和 NW-UCLA 数据集的图拓扑定义
- `feeders/` - NTU 和 UCLA 数据集的数据加载器
- `torchlight/` - 自定义 PyTorch 工具

### VIBE (`VIBE/`)
- `demo.py` / `demo_alter.py` - Demo 推理脚本
- `train.py` - 训练脚本
- `lib/models/vibe.py` - VIBE 模型
- `lib/models/smpl.py` - SMPL 身体模型集成
- `lib/smplify/` - Temporal SMPLify 实现
- `lib/dataset/` - 数据集加载器 (AMASS, 3DPW, MPI-INF-3DHP 等)
- `lib/utils/` - 渲染、几何、姿态追踪工具

## 环境

- CTR-GCN: Python 3.6+, PyTorch >= 1.1.0, 使用 conda 环境 `ctr_gcn`
- VIBE: Python >= 3.7, 需要 SMPL 模型文件 (从 https://smpl.is.tue.mpg.de/ 下载)
- 两者训练/推理均需要 GPU

## 数据

- **NTU RGB+D 60/120**: 基于骨架的动作识别数据集 (申请地址: https://rose1.ntu.edu.sg/dataset/actionRecognition)
- **NW-UCLA**: 跨视角动作识别数据集
- **3DPW, AMASS, MPI-INF-3DHP**: VIBE 训练的 3D 姿态估计数据集
