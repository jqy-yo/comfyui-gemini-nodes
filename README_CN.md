# ComfyUI Gemini 节点

[English](README.md)

一个用于将 Google Gemini API 集成到 ComfyUI 的自定义节点集合，提供强大的 AI 功能，包括文本生成、图像生成和视频分析。

## 功能特性

### 🤖 Gemini 文本 API
- 使用任意 Gemini 模型生成文本（通过文本输入灵活选择模型）
- 支持所有 Gemini 模型，包括 gemini-2.5-flash-lite、gemini-2.5-pro、gemini-2.0-flash、gemini-1.5-pro 等
- 可配置的生成参数（温度、最大令牌数、top_p、top_k）
- 支持自定义系统指令
- 完善的错误处理和重试逻辑

### 🎨 Gemini 图像编辑器
- 使用任意具有图像生成能力的 Gemini 模型生成图像
- 灵活的模型输入支持 gemini-2.5-flash、models/gemini-2.0-flash-exp、imagen-3.0-generate-001 等
- 支持最多 4 张参考图像作为输入
- 批量生成（最多 8 张图像）
- 自动图像尺寸调整（最小 1024x1024）
- 异步并行处理以提高效率

### 🚀 Gemini 高级图像生成
- 多插槽输入系统（最多 100 个输入组合）
- 每个插槽独立的图像和提示词
- 异步并行 API 调用
- 带进度跟踪的批处理
- 自动图像填充和格式转换

### 🎬 Gemini 视频字幕生成器
- 为视频生成描述性字幕
- 支持视频文件路径或图像批量输入
- 自动视频格式转换（WebM）
- 智能文件大小控制（30MB 以下）
- 帧提取和时间戳处理

## 安装

1. 将此仓库克隆到您的 ComfyUI 自定义节点目录：
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/jqy-yo/comfyui-gemini-nodes.git
```

2. 安装所需依赖：
```bash
cd comfyui-gemini-nodes
pip install -r requirements.txt
```

3. 设置您的 Google Gemini API 密钥作为环境变量：
```bash
export GOOGLE_API_KEY="your-api-key-here"
```

## 配置

### API 密钥设置

节点按以下顺序查找 Gemini API 密钥：
1. 环境变量：`GOOGLE_API_KEY`
2. ComfyUI 设置文件

为了安全起见，我们建议使用环境变量。

### 模型选择

所有节点现在都支持通过文本字段灵活输入模型。您可以输入任何有效的 Gemini 模型名称：

- **文本生成**：任何 Gemini 模型（例如：gemini-2.5-flash-lite、gemini-2.5-pro、gemini-2.0-flash、gemini-1.5-pro）
- **图像生成**：任何具有图像生成能力的模型（例如：gemini-2.5-flash、models/gemini-2.0-flash-exp、imagen-3.0-generate-001）
- **视频分析**：任何具有视频分析能力的模型（例如：gemini-2.5-flash-lite、gemini-2.0-flash、gemini-1.5-pro）

只需在模型字段中输入模型名称。节点将自动处理 API 端点和配置。

### API 版本选择

所有节点都支持 API 版本选择：
- **auto**（默认）：自动根据模型选择合适的版本
- **v1**：稳定版本，用于生产环境
- **v1beta**：测试版本，包含新功能
- **v1alpha**：早期访问版本，实验性功能

## 使用方法

安装后，节点将出现在 ComfyUI 的"🤖 Gemini"类别下：

1. **Gemini Text API**：用于文本生成任务
2. **Gemini Image Editor**：用于使用参考图像生成图像
3. **Gemini Image Gen Advanced**：用于复杂的多输入图像生成
4. **Gemini Video Captioner**：用于视频分析和字幕生成

## 示例

### 文本生成
- 将提示词连接到 Gemini Text API 节点
- 配置温度和其他参数
- 获取生成的文本输出

### 图像生成
- 输入参考图像（可选）
- 提供文本提示词
- 配置批量大小和模型
- 接收生成的图像

### 视频字幕
- 输入视频文件路径或图像序列
- 配置提示词和模型
- 获取描述性字幕

## 系统要求

- ComfyUI
- Python 3.8+
- google-generativeai
- Pillow (PIL)
- numpy
- torch
- cv2 (opencv-python)
- moviepy
- aiohttp

## 故障排除

### API 密钥问题
- 确保您的 API 密钥已正确设置在环境变量中
- 检查您要使用的模型的 API 密钥权限

### 模型访问
- 某些模型需要特定的 API 访问级别
- Imagen 模型可能有地区限制

### 内存问题
- 大批量大小可能会导致内存问题
- 如需要，请减少批量大小或图像分辨率

### 地区限制
如果遇到"User location is not supported for the API use"错误：
- 使用 VPN 连接到支持的地区（美国、欧洲、日本等）
- 确保 API 密钥是在支持的地区创建的

## 许可证

本项目采用 MIT 许可证 - 详见 LICENSE 文件。

## 致谢

本项目基于 [ComfyUI_Fill-Nodes](https://github.com/filliptm/ComfyUI_Fill-Nodes) 的代码，由 filliptm 开发。特别感谢 filliptm 的原始实现和灵感。

节点已经过重构和增强，专注于 Gemini API 集成，并添加了以下功能：
- 通过文本输入灵活选择模型
- API 版本选择支持
- 改进的错误处理和兼容性
- 支持最新的 Gemini 模型

## 鸣谢

- 原始代码和概念由 [filliptm](https://github.com/filliptm) 提供
- 基于 [ComfyUI_Fill-Nodes](https://github.com/filliptm/ComfyUI_Fill-Nodes)
- 由 [jqy-yo](https://github.com/jqy-yo) 重构和维护

## 支持

如有问题、疑问或贡献，请在 GitHub 上提交 issue。