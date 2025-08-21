# ComfyUI Gemini 节点

[English](README.md)

一个用于将 Google Gemini API 集成到 ComfyUI 的自定义节点集合，提供强大的 AI 功能，包括文本生成、结构化输出、图像生成、视频分析和 JSON 处理。

## 📋 目录
- [功能特性](#功能特性)
- [安装](#安装)
- [配置](#配置)
- [节点文档](#节点文档)
- [使用示例](#使用示例)
- [故障排除](#故障排除)
- [许可证](#许可证)
- [致谢](#致谢)

## 功能特性

### 核心功能
- **文本生成**: 使用所有 Gemini 模型进行高级文本生成
- **结构化输出**: 基于 JSON Schema 的响应和数据提取
- **图像生成**: 支持多参考图像的批量图像生成
- **视频分析**: 智能视频字幕和分析
- **JSON 处理**: 提取、解析和操作 JSON 数据

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

3. 重启 ComfyUI

## 配置

### API 密钥设置

设置您的 Google Gemini API 密钥的方式：

1. **环境变量（推荐）**：
```bash
export GOOGLE_API_KEY="your-api-key-here"
```

2. **节点输入**：直接在节点的 API 密钥字段中输入

### 模型选择

所有节点都支持灵活的模型输入。输入任何有效的 Gemini 模型名称：
- 文本/结构化：`gemini-2.0-flash`、`gemini-1.5-pro`
- 图像生成：`imagen-3.0-generate-001`、`gemini-2.0-flash-exp`
- 视频分析：`gemini-2.0-flash`、`gemini-1.5-pro`

## 节点文档

### 🤖 Gemini 文本 API
使用任何 Gemini 模型生成文本响应，支持灵活的模型选择和高级生成参数。

**输入参数：**
- `prompt`（STRING）：输入文本提示
- `api_key`（STRING）：您的 Google API 密钥（可使用环境变量 GOOGLE_API_KEY）
- `model`（STRING）：模型名称（默认："gemini-2.0-flash"）
  - 示例：gemini-2.5-flash-lite、gemini-2.0-flash、gemini-1.5-pro
- `temperature`（FLOAT，0.0-1.0，默认：0.7）：控制生成的随机性
  - 0.0：最确定性/聚焦
  - 1.0：最有创意/多样化
- `max_output_tokens`（INT，64-8192，默认：1024）：最大响应长度
- `seed`（INT，0-2147483647，默认：0）：随机种子用于可重现性
- `system_instructions`（STRING，可选）：系统级指令来指导模型行为
- `top_p`（FLOAT，0.0-1.0，默认：0.95）：核采样阈值
- `top_k`（INT，1-100，默认：64）：Top-k 采样参数
- `api_version`（ENUM，可选）：API 版本选择（auto/v1/v1beta/v1alpha）

**输出：**
- `response`（STRING）：生成的文本响应

**使用示例：**
1. 基础文本生成：提示词="写一首关于人工智能的俳句"，温度=0.8
2. 技术文档：提示词="解释二叉搜索树的实现"，温度=0.3，系统指令="提供详细的技术解释和 Python 代码示例"
3. 创意写作：提示词="续写这个故事：门缓缓打开，露出了..."，温度=0.9，最大令牌=2048

### 📊 Gemini 结构化输出
使用 JSON Schema 验证生成特定 JSON 结构的响应。

**输入参数：**
- `prompt`（STRING）：描述要生成什么的输入提示
- `api_key`（STRING）：您的 Google API 密钥
- `model`（STRING，默认："gemini-2.0-flash"）：模型名称
- `output_mode`（ENUM）：
  - `json_schema`：使用自定义 JSON 模式
  - `enum`：从预定义选项生成
- `schema_json`（STRING）：JSON Schema 定义（用于 json_schema 模式）
- `temperature`（FLOAT，0.0-1.0，默认：0.7）：生成温度
- `max_output_tokens`（INT，64-8192，默认：1024）：最大令牌数
- `seed`（INT）：随机种子
- `system_instructions`（STRING，可选）：系统级指令
- `enum_options`（STRING，可选）：枚举选项的 JSON 数组（用于 enum 模式）
- `property_ordering`（STRING，可选）：逗号分隔的属性顺序

**输出：**
- `structured_output`（STRING）：格式化的 JSON 输出
- `raw_json`（STRING）：原始 JSON 响应

**Schema 示例：**
```json
{
  "type": "object",
  "properties": {
    "product_name": {"type": "string"},
    "price": {"type": "number"},
    "in_stock": {"type": "boolean"},
    "categories": {
      "type": "array",
      "items": {"type": "string"}
    }
  },
  "required": ["product_name", "price"]
}
```

### 🔍 Gemini JSON 提取器
使用字段定义从文本中提取特定信息到 JSON 格式。

**输入参数：**
- `prompt`（STRING）：描述要提取什么的提取指令
- `api_key`（STRING）：您的 Google API 密钥
- `model`（STRING，默认："gemini-2.0-flash"）：模型名称
- `extract_fields`（STRING）：简单格式的字段定义
- `temperature`（FLOAT，0.0-1.0，默认：0.3）：较低值用于准确提取
- `seed`（INT）：随机种子
- `input_text`（STRING，可选）：要提取的文本
- `system_instructions`（STRING，可选）：默认："从提供的文本中提取请求的信息。准确且简洁。"

**字段定义格式：**
```
field_name: type
field_name2: type[]
field_name3: type?
```

**支持的类型：**
- `string`：文本值
- `number`/`float`：数值
- `integer`/`int`：整数
- `boolean`/`bool`：真/假值
- `string[]`：字符串数组
- `?` 后缀：可选字段（非必需）

**使用示例：**
```
提取字段：
title: string
author: string
keywords: string[]
rating: number

输入文本："这篇由李博士撰写的突破性研究..."
输出：{
  "title": "突破性研究",
  "author": "李博士",
  "keywords": ["AI", "医疗", "创新"],
  "rating": 4.8
}
```

### 📌 Gemini 字段提取器
使用路径符号从 JSON 数据中提取特定字段。此节点不使用 Gemini API - 它是纯 JSON 处理工具。

**输入参数：**
- `json_input`（STRING）：要提取的 JSON 数据
- `field_path`（STRING）：使用点符号的字段路径
- `default_value`（STRING）：字段未找到时返回的值
- `output_format`（ENUM）：
  - `auto`：自动确定格式
  - `string`：转换为字符串
  - `json`：格式化为 JSON
  - `number`：提取为数字
  - `boolean`：转换为布尔值
  - `list`：格式化为列表/数组
- `array_handling`（ENUM，可选）：
  - `all`：返回整个数组
  - `first`：返回第一个元素
  - `last`：返回最后一个元素
  - `join`：连接数组元素
- `join_separator`（STRING，默认：", "）：连接数组时的分隔符

**路径符号示例：**
- 简单字段：`name`、`email`、`id`
- 嵌套对象：`user.profile.age`、`settings.theme.color`
- 按索引访问数组：`items[0]`、`users[2].name`
- 负索引：`items[-1]`（最后一项）
- 所有数组项：`users[*].email`、`products[*].price`
- 复杂路径：`data.users[0].orders[*].items[0].price`

**使用示例：**
```
JSON 输入：{"user": {"name": "张三", "email": "zhang@example.com"}}
字段路径：user.email
输出：zhang@example.com
```

### 🛠️ Gemini JSON 解析器
解析和操作 JSON 数据。此节点不使用 Gemini API - 它是纯 JSON 处理实用程序。

**输入参数：**
- `json_input`（STRING）：要处理的 JSON 数据
- `operation`（ENUM）：要执行的操作
  - `validate`：检查 JSON 是否有效并显示结构信息
  - `format`：使用缩进美化打印
  - `minify`：删除不必要的空白以紧凑存储
  - `extract_keys`：获取 JSON 中的所有键路径
  - `get_type`：分析 JSON 的类型结构
  - `count_items`：按类型计数元素
- `indent`（INT，0-8，默认：2）：格式化的缩进空格
- `sort_keys`（BOOLEAN，默认：False）：按字母顺序排序对象键

**输出：**
- `result`（STRING）：操作结果
- `info`（STRING）：详细操作信息
- `success`（BOOLEAN）：操作成功为 True

**操作示例：**
1. 验证：检查 JSON 有效性并获取结构信息
2. 格式化：美化打印 JSON，可选排序键
3. 压缩：减小 JSON 大小用于存储/传输
4. 提取键：获取所有嵌套键路径
5. 类型分析：查看数据类型结构
6. 计数项目：统计不同类型元素的数量

### 🎨 Gemini 图像编辑器
使用 Gemini 的图像生成模型生成带有可选参考图像的图像。

**输入参数：**
- `prompt`（STRING）：详细的图像生成提示
- `api_key`（STRING）：您的 Google API 密钥
- `model`（STRING）：模型名称
  - 默认："models/gemini-2.0-flash-preview-image-generation"
  - 替代："imagen-3.0-generate-001"、"gemini-2.5-flash"
- `temperature`（FLOAT，0.0-2.0，默认：1.0）：生成创意度
- `max_retries`（INT，1-5，默认：3）：API 重试次数
- `batch_size`（INT，1-8，默认：1）：要生成的图像数量
- `seed`（INT，可选）：随机种子
- `image1` 到 `image4`（IMAGE，可选）：用于风格/内容指导的参考图像
- `api_version`（ENUM，可选）：API 版本

**功能：**
- 自动图像填充至最小 1024x1024（白色边框）
- 批量生成的异步并行处理
- 带指数退避的内置重试逻辑
- 失败时生成错误图像（带错误文本的黑色图像）
- 支持最多 4 张参考图像

**使用示例：**
1. 简单图像生成："一个宁静的日本花园与樱花"，批量大小=4
2. 风格转换："将此转换为水彩画风格"，参考图像1=[原始照片]
3. 产品可视化："白色背景上的现代极简产品摄影"，参考图像=[产品草图]
4. 角色设计变化："创建此角色的不同姿势变化"，批量大小=6

### 🚀 Gemini 高级图像生成
用于批量处理多个图像/提示组合的高级多插槽图像生成系统。

**输入参数：**
- `inputcount`（INT，1-100，默认：1）：要使用的输入插槽数
- `api_key`（STRING）：您的 Google API 密钥
- `model`（STRING）：模型名称
  - 默认："models/gemini-2.0-flash-preview-image-generation"
- `temperature`（FLOAT，0.0-2.0，默认：1.0）：生成创意度
- `max_retries`（INT，1-5，默认：3）：每个插槽的 API 重试次数
- `prompt_1`（STRING，必需）：第一个生成提示
- `image_1`（IMAGE，可选）：第一个参考图像
- `seed`（INT，可选）：随机种子
- `retry_indefinitely`（BOOLEAN，可选）：失败时持续重试

**动态输入（基于 inputcount）：**
- `input_image_X`（IMAGE）：插槽 X 的参考图像
- `input_prompt_X`（STRING）：插槽 X 的生成提示
- 其中 X 范围从 1 到 inputcount

**功能：**
- 所有插槽的异步并行处理
- 带实时更新的进度条
- 自动图像填充至 1024x1024 最小值
- 批量结果聚合
- 带回退图像的错误处理

**使用示例：**
1. 批量产品变体：5 个插槽，每个插槽不同颜色的手提包
2. 场景变化：10 个插槽，同一场景的不同时间
3. 风格探索：8 个插槽，同一图像的不同艺术风格
4. 营销资产生成：20 个插槽的社交媒体内容批量生成

### 🎬 Gemini 视频字幕生成器
使用 Gemini 的多模态功能为视频生成智能字幕和描述。

**输入参数：**
- `api_key`（STRING）：您的 Google API 密钥
- `model`（STRING）：模型名称
  - 默认："gemini-2.0-flash"
  - 替代："gemini-2.5-flash-lite"、"gemini-1.5-pro"
- `frames_per_second`（FLOAT，0.1-10.0，默认：1.0）：帧采样率
- `max_duration_minutes`（FLOAT，0.1-45.0，默认：2.0）：要处理的最大视频时长
- `prompt`（STRING）：分析指令
  - 默认："详细描述此视频场景。包括任何重要的动作、主题、设置和氛围。"
- `process_audio`（ENUM ["false", "true"]，默认："false"）：包括音频分析
- `temperature`（FLOAT，0.0-1.0，默认：0.7）：生成温度
- `max_output_tokens`（INT，50-8192，默认：1024）：最大字幕长度
- `video_path`（STRING，可选）：视频文件路径
- `image`（IMAGE，可选）：作为视频帧的图像序列/批次

**功能：**
- 自动视频格式转换为 WebM
- 基于 FPS 设置的智能帧采样
- 文件大小优化（30MB 限制下）
- 支持视频文件和图像序列
- 音频处理能力（取决于模型）
- 带时间戳的帧提取

**输出：**
- `caption`（STRING）：生成的视频描述/字幕
- `sampled_frame`（IMAGE）：视频的代表性帧

**使用示例：**
1. 基础视频描述："描述这个视频中发生了什么"，FPS=1.0
2. 技术分析："分析摄影技术、相机运动和视觉构图"，FPS=2.0
3. 动作检测："列出视频中人物执行的所有动作及时间戳"，FPS=5.0
4. 教程总结："创建此教程视频的分步总结"，最大时长=10.0

## 使用示例

### 示例 1：提取产品信息
```
1. GeminiStructuredOutput → 定义产品 schema
2. 输入：产品描述文本
3. 输出：包含名称、价格、特性的结构化 JSON
4. GeminiFieldExtractor → 仅提取价格
5. 在下游节点中使用价格
```

### 示例 2：带数据的批量图像生成
```
1. GeminiJSONExtractor → 从文本中提取图像描述
2. GeminiFieldExtractor → 获取描述数组
3. GeminiImageGenAdvanced → 为每个描述生成图像
4. 输出：批量生成的图像
```

### 示例 3：视频分析流程
```
1. 加载视频 → GeminiVideoCaptioner
2. 字幕 → GeminiJSONExtractor（提取关键时刻）
3. 关键时刻 → GeminiStructuredOutput（时间线格式）
4. 输出：结构化视频时间线
```

### 示例 4：复杂数据处理
```
1. API 响应 → GeminiJSONParser（验证）
2. 如果有效 → GeminiFieldExtractor（提取嵌套数据）
3. 提取的数据 → GeminiStructuredOutput（重新格式化）
4. 输出：清理、重构的数据
```

## 最佳实践

### 结构化输出
- 保持模式简单且专注
- 谨慎使用必需字段
- 首先使用示例数据测试模式
- 提供清晰的属性描述

### 字段提取
- 使用特定路径避免歧义
- 设置有意义的默认值
- 使用示例 JSON 测试路径语法
- 适当使用数组处理选项

### 图像生成
- 提供详细、具体的提示
- 尽可能使用参考图像
- 保持合理的批量大小以节省内存
- 使用负面提示来提高质量

### 视频分析
- 将视频保持在 30MB 以下
- 使用清晰、具体的分析提示
- 考虑对长视频进行帧采样
- 结合结构化输出进行数据提取

## 故障排除

### 常见问题

**API 密钥错误：**
- 验证密钥是否有效且处于活动状态
- 检查密钥是否具有所需权限
- 确保密钥中没有多余的空格

**模型访问问题：**
- 某些模型需要特定的访问级别
- 检查地区可用性
- 验证模型名称拼写

**JSON/Schema 错误：**
- 在输入前验证 JSON 语法
- 检查 schema 是否遵循 JSON Schema 规范
- 如果错误持续，使用更简单的 schema

**内存问题：**
- 减少批量大小
- 降低图像分辨率
- 分段处理视频

**字段提取问题：**
- 验证 JSON 是否有效
- 检查字段路径语法
- 首先使用更简单的路径测试

## 系统要求

- ComfyUI
- Python 3.8+
- google-genai
- Pillow (PIL)
- numpy
- torch
- opencv-python
- moviepy
- aiohttp

## 许可证

本项目采用 MIT 许可证 - 详见 LICENSE 文件。

## 致谢

本项目基于 [ComfyUI_Fill-Nodes](https://github.com/filliptm/ComfyUI_Fill-Nodes) 的代码，由 filliptm 开发。特别感谢原始实现。

### 增强功能包括：
- 结构化输出支持
- JSON 处理能力
- 字段提取系统
- 灵活的模型选择
- 改进的错误处理

## 鸣谢

- 原始代码由 [filliptm](https://github.com/filliptm) 提供
- 基于 [ComfyUI_Fill-Nodes](https://github.com/filliptm/ComfyUI_Fill-Nodes)
- 由 [jqy-yo](https://github.com/jqy-yo) 维护

## 支持

如有问题、疑问或贡献，请在 GitHub 上提交 issue。