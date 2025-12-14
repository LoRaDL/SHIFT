# CRL视频分析 + VLM上下文构建

## 快速开始

### 1. 基本视频分析
```bash
python crl_video_analysis.py
```
生成带OSD的分析视频，实时显示检测结果。

### 2. 关键帧提取
```bash
python crl_keymoment_video_fast.py
```
提取关键时刻，生成高亮视频。

### 3. VLM上下文构建（新功能）
```bash
python crl_vlm_context_builder.py
```
提取关键帧序列，为VLM构建上下文。

### 4. 完整VLM工作流
```bash
python vlm_workflow_example.py
```
一键完成：关键帧提取 → VLM分析 → 结果保存

## 配置

编辑 `crl_config.py`：
- `VIDEO_PATH`: 视频路径
- `CRITERION`: 检测目标（如"chair detection"）
- `BASIS_TEXTS`: 检测描述词

## VLM使用 (LMStudio)

支持两种方式：
1. **自动API调用**: LMStudio服务器运行时自动分析
2. **手动上传**: 导出图片和提示词到LMStudio界面

### LMStudio设置
1. 下载并安装 [LMStudio](https://lmstudio.ai/)
2. 下载视觉语言模型 (推荐: LLaVA, MiniCPM-V)
3. 启动本地服务器 (默认: http://localhost:1234)

## 输出

- `vlm_context_output/`: 关键帧图片 + JSON上下文
- `vlm_workflow_output/`: 完整分析结果
- `*.mp4`: 分析视频

就这么简单！