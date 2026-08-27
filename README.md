# 基于 PyTorch 的 MNIST 手写数字识别

这是一个面向深度学习初学者的完整 MNIST 项目，包含数据加载、MLP 与
CNN 训练、测试集评估、模型保存、现实手写图片预测、错误样本分析和
Streamlit 可视化界面。

为了方便沿着一条完整流程阅读代码，MLP 和 CNN 分别整合在各自的文件中：

- `mlp.py`：MLP 模型、训练、评估、预测和错误分析。
- `cnn.py`：CNN 模型、训练、评估、现实图片增强预处理、预测和错误分析。

## 项目结构

```text
mnist_project/
├── mlp.py                    # MLP 完整流程
├── cnn.py                    # CNN 完整流程
├── compare.py                # 公平比较 MLP 与 CNN
├── app.py                    # 统一的 Streamlit 界面
├── requirements.txt          # Python 依赖
├── tests/
│   └── test_project.py       # 自动化测试
├── data/                     # 自动下载的 MNIST 数据
├── models/                   # 训练生成的模型权重
└── images/                   # 测试图片和输出图表
```

`data/` 中的数据和 `models/` 中的权重都可以重新生成，因此不会提交到 Git。
第一次训练或评估时，torchvision 会自动下载 MNIST。

## 环境要求

- Python 3.12
- macOS、Linux 或 Windows
- CPU 即可运行

推荐使用 [uv](https://docs.astral.sh/uv/)：

```bash
git clone <your-repository-url>
cd mnist_project
uv venv --python 3.12 --seed .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

不使用 uv 时：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows 激活虚拟环境：

```powershell
.venv\Scripts\activate
```

## MLP 完整流程

```bash
# 训练并保存 models/mlp_state_dict.pth
python mlp.py train

# 在 10,000 张 MNIST 测试图片上评估
python mlp.py evaluate

# 识别一张现实图片
python mlp.py predict images/my_8.png

# 可视化正确和错误预测
python mlp.py analyze
```

## CNN 完整流程

```bash
# 训练并保存 models/cnn_state_dict.pth
python cnn.py train

# 在 10,000 张 MNIST 测试图片上评估
python cnn.py evaluate

# 识别一张现实图片
python cnn.py predict images/my_4.jpg

# 可视化正确和错误预测
python cnn.py analyze
```

必须先训练对应模型，才能执行评估、预测、错误分析或 Web 界面。

CNN 会把现实图片转换为灰度图，自动判断笔画颜色、去除灰色背景、裁剪
数字、保持宽高比缩放，并根据笔画重心放入 28×28 黑色画布。

现实图片建议只包含一个数字，并尽量保证背景均匀、笔画清晰、数字完整。
模型置信度表示模型自身的确定程度，不代表预测一定正确。

## 比较 MLP 与 CNN

```bash
python compare.py
```

比较实验使用相同的 epoch、batch size、学习率、随机种子和训练数据顺序，
输出参数量、测试准确率和训练时间。

## Web 界面

```bash
python -m streamlit run app.py
```

在页面中选择 MLP 或 CNN，上传 PNG/JPG 图片并点击“开始识别”。

## 自动化测试

测试不下载 MNIST，也不依赖已经训练好的权重：

```bash
python -m unittest discover -s tests -v
```

测试覆盖：

- MLP 和 CNN 的前向传播、输出 shape 与反向传播。
- 白底黑字、灰底白字、空白图片等预处理情况。
- 临时保存并重新加载 `state_dict` 后的预测接口。

## 设计参考

训练流程参考了 PyTorch 官方
[MNIST 示例](https://github.com/pytorch/examples/blob/main/mnist/main.py)：使用
`DataLoader` 读取数据，通过 `model.train()` 和 `model.eval()` 切换模式，在
推理阶段使用 `torch.no_grad()`，并用 `state_dict` 保存模型参数。
