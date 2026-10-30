# BCI IV-2a：PowerCNN 改进模型

本仓库只交付改进后的 **PowerCNN**，包括源码、固定配置、45份预训练模型、一键复现脚本、Conda依赖、运行日志和参考结果。

参考实测：**2a 四分类平均准确率63.19% ±17.78%（受试者间SD），κ=0.5092，macro-F1=0.6256**。
计算方式是每名受试者先平均全部5个随机种子，再对9名受试者等权平均。

## 一键复现

需要Windows、已安装的Conda以及可访问官方下载站的网络。项目自动复用现有`bci2a` Conda环境，环境不存在时根据`environment.yml`创建。无需旧项目源码。

```powershell
git clone https://github.com/gaojun1234/-BCI-Competition-IV-2a-2b-.git powercnn
cd powercnn
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_all.ps1
```

默认依次执行：静态检查、单元测试、官方数据下载/校验、预处理、9人×5种子的**从头训练**、全部预测冻结、一次批量评分、CSV和JSON汇总。首次需下载约420MiB GDF数据；GPU优先，也支持CPU。

指定设备或只核验已训练模型：

```powershell
.\run_all.ps1 -Device cuda
.\run_all.ps1 -Device cpu
.\run_all.ps1 -VerifyOnly
```

`-VerifyOnly`解压并校验`pretrained/models.zip`中的45份模型，用相同数据重做预测和评分，不进行训练。
若需要手动创建环境：

```powershell
conda env create -f environment.yml
conda activate bci2a
python run.py all --device auto
```

## 预期输出

|文件/目录|内容|
|---|---|
|`logs/run_all.log`|一键运行完整日志，包含检查、45次训练与评分输出|
|`outputs/results.csv`|45行，含dataset、subject、model、seed、accuracy、kappa、macro_f1、train_time等|
|`outputs/subject_results.csv`|9人各自的五种子均值与种子SD|
|`outputs/summary.json`|总体准确率、受试者SD、κ、F1及参考准确率|
|`outputs/models/`、`outputs/predictions/`|45份新训练模型和等长四类概率|
|`outputs/runs/`|45份训练记录、每轮学习曲线、配置及数据/模型/预测哈希|
|`outputs/recipe_frozen.json`、`outputs/frozen_predictions.json`|先冻结配置，后冻结全部模型与预测，评分前校验|
|`outputs/environment.json`|实际Python、PyTorch、CUDA版本和随机种子|
|`reference/`|已完成实验的参考CSV、概率和复现核验记录|
|`pretrained/models.zip`|9名受试者×5种子的45份已训练PowerCNN|

同一已验证软件与GPU配置的参考均值为`0.6318672839506172`，共有`12960=9×5×288`次试次预测。
跨硬件/依赖版本可能存在数值差异；程序实际打印本次结果，不把参考准确率写成计算结果。
逐试次预测概率不随仓库公开上传；运行时在本地生成。可选的本地参考概率文件不存在时，一键训练和评分仍可完整执行。

## 模型与固定协议

模型为手写PyTorch网络：时间卷积(1×25、步长2、8滤波器)→BN→空间卷积(22×1、16滤波器)→BN→平方功率→平均池化→log→Dropout0.25→全连接4分类。
全局池化版本3156参数，局部池化版本3732参数；均小于5k。部分受试者采用**仅由T训练数据拟合的CSP空间滤波器初始化**，随后由网络梯度训练。

每人的池化/初始化配置和训练轮数已按T内部真实run分组验证选定，固定保存在`config/subjects.json`。本工程复现已选定的最终方案，不重跑架构搜索，不按E结果选择种子或配置。

- 固定种子：`11,29,47,71,101`，全部报告。
- 2a每人T训练、E测试；保留全部伪迹试次，不使用EOG。
- 8–30Hz四阶Butterworth零相位滤波、50Hz陷波、提示后0.5–3.5秒、128Hz、trial/channel z-score；输入22×384。
- Adam学习率0.001，batch32。最终训练使用固定轮数，**不在E上早停**。
- 所有45份模型和E预测先保存并计算哈希，评分程序通过完整性检查后才能读取真实E标签。
- 内部标签：0左手、1右手、2双足、3舌部。

这是改进架构，不冒充课题指定的原始极简CNN基线。此前已经接触过2a E标签，本结果属于透明记录的后续探索实验，不能宣称为全新盲测。

官方数据：[BCI Competition IV](https://www.bbci.de/competition/iv/)、[2a说明](https://www.bbci.de/competition/iv/desc_2a.pdf)。

## 检查命令

```powershell
conda run -n bci2a python -m ruff check .
conda run -n bci2a python -m ruff format --check .
conda run -n bci2a python -m pytest -q
```

原始GDF会自动校验18份文件的SHA256。源码中没有对旧工程的导入依赖；若旁边已有原始数据目录，会直接复用原始文件。
