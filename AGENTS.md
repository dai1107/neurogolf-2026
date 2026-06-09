## 一、项目定位

本仓库用于参加 NeuroGolf / ARC-AGI ONNX 网络高尔夫类比赛。

本项目的目标不是训练一个大型神经网络，而是为每个 ARC 任务构造一个尽可能小的 ONNX 神经网络，使它能够正确完成对应的图形变换。

核心目标：

1. 每个任务生成一个对应的 ONNX 网络。
2. 网络必须正确完成该任务的输入到输出变换。
3. 在正确的前提下，尽量减少：
   - 参数数量
   - 内存占用
   - ONNX 文件大小
   - 图结构复杂度

正确性永远优先于模型大小。

如果一个模型更小但不稳定、不确定或没有通过验证，则不能作为最终模型。

## 二、全局原则

在本仓库中进行任何修改时，必须始终按照以下优先级决策：

1. 功能正确性
2. ONNX 合法性
3. 比赛约束合规性
4. 可复现性
5. 参数量最小化
6. 内存占用最小化
7. 文件体积最小化
8. 代码清晰度和可维护性

不得为了压缩模型而牺牲正确性。

不得在验证失败时生成最终提交模型。

不得把不确定是否合规的模型放入 `submission.zip`。

## 三、比赛约束

生成的每个 ONNX 网络都必须满足以下要求：

1. 所有 tensor 和参数 shape 必须是静态定义的。
2. 单个 ONNX 文件大小不得超过 1.44MB。
3. 禁止使用以下 ONNX 操作：
   - Loop
   - Scan
   - NonZero
   - Unique
   - Script
   - Function
4. 推理阶段不得依赖 Python 运行时。
5. 推理阶段不得依赖外部文件。
6. 不得使用随机推理。
7. 不得依赖训练模式行为。
8. 生成的 ONNX 必须能被 `onnx.checker.check_model` 检查通过。
9. 如果有官方 validator，必须尽量兼容官方 validator。

如果某个 ONNX 算子是否允许存在疑问，应默认认为它不允许，直到被明确验证。

## 四、数据表示约定

ARC 任务中的 grid 是离散颜色网格。

颜色通常表示为整数：

```text
0, 1, 2, ..., 9

内部神经网络表示优先使用 one-hot channel encoding：

颜色 0 -> channel 0
颜色 1 -> channel 1
颜色 2 -> channel 2
...
颜色 9 -> channel 9

常用张量格式为：

NCHW

即：

[batch, channels, height, width]

通常可以使用：

[1, 10, 30, 30]

但最终实现必须以项目中的官方工具、validator 或比赛要求为准。

如果使用 padding，必须明确 padding 语义。

不得把 padding 区域和真实颜色 0 区域混淆。

如果输出需要从 one-hot tensor 转回 grid，必须使用确定性规则。

如果某个位置的多个 channel 数值接近或无法唯一判断颜色，则应视为验证失败，而不是强行通过。

五、项目方法论

应把每个 ARC 任务看成一个小型符号变换问题，然后把这个变换编译成一个小型 ONNX 网络。

优先使用规则化、可解释、可验证的网络，而不是通用大模型。

推荐的解法类型包括：

恒等映射
颜色替换
局部卷积规则
固定方向平移
裁剪与补齐
镜像
旋转
对象掩码
包围盒操作
填充、擦除、重染色
图形扩张或收缩
模式补全
连接线生成
基于颜色、大小、位置的对象选择

不应优先训练大型 CNN。

大型神经网络只能作为非常靠后的实验性方案，并且必须满足：

ONNX 合法
文件足够小
参数足够少
完整验证通过
不违反比赛限制
六、Agent 角色体系

Codex 在本仓库中应始终按照多个专业 Agent 协同的方式工作。

同一次 Codex 会话可以承担多个角色，但每个角色的职责边界必须清楚。

1. 任务分析 Agent

负责理解 ARC 任务的输入输出变换规律。

职责：

检查所有 train 输入输出样例。
分析输入输出尺寸关系。
分析颜色集合变化。
推断背景色。
找出发生变化的格子。
判断是否存在对象级对应关系。
判断任务属于以下哪类：
局部规则
全局规则
几何变换
颜色变换
对象选择
对象计数
模式补全
区域填充
明确指出不确定性。
不得只根据单个样例草率归纳规则。

任务分析 Agent 必须保证推断出的规则能够解释所有已知训练样例。

如果规则只能解释部分样例，应标记为不确定，而不是直接生成最终模型。

2. 规则工程 Agent

负责把分析出的变换规律实现为可复用规则。

职责：

实现可复用的规则类或规则函数。
为每个规则提供 matcher。
为每个规则提供 ONNX builder。
为每个规则提供验证路径。
为每个规则提供失败解释。
为每个规则提供 cost 估计。
尽量让规则可以复用于多个任务。
保持规则匹配保守。

规则匹配结果应分为：

MATCH
POSSIBLE
REJECT

含义：

MATCH   = 规则明确匹配，可以生成候选模型
POSSIBLE = 规则可能匹配，但需要实验性验证
REJECT  = 规则不适用

只有 MATCH 可以直接生成正式候选模型。

POSSIBLE 只能生成实验候选，不能未经验证进入最终提交。

规则不得因为“刚好在一个样例上有效”就判定为匹配。

3. ONNX 构造 Agent

负责生成合法、紧凑、可验证的 ONNX 图。

职责：

使用静态 shape。
只使用允许的 ONNX 算子。
保持 input/output 名称稳定。
减少 initializer 数量。
减少参数数量。
减少无用节点。
避免无意义的 Cast、Reshape、Transpose。
避免大型常量表。
构造完成后运行 ONNX checker。
保持图结构简单、可解释。

优先使用以下结构：

1x1 Conv
小 kernel Conv，例如 3x3 或 5x5
Add
Sub
Mul
Relu
Clip
MaxPool 或 AveragePool，仅在确认允许且必要时使用
固定权重线性变换
简单组合网络

如果某个效果可以用更小的线性层或卷积实现，不要使用更复杂的图结构。

4. 验证 Agent

负责判断生成的模型是否真正可用。

职责：

使用 onnxruntime 或官方 validator 运行模型。
在所有 train 样例上比较输出。
检查输出 grid 是否完全等于目标 grid。
检查输出尺寸是否正确。
检查颜色是否正确。
检查是否存在浮点歧义。
检查 forbidden ops。
检查静态 shape。
检查文件大小。
检查参数数量和内存估计。

验证必须是严格的。

不能用“看起来差不多”代替精确匹配。

不能只验证部分样例。

不能在失败时吞掉异常。

如果验证失败，日志必须说明：

哪个样例失败
哪些位置不同
期望颜色是什么
实际颜色是什么
是否是尺寸错误
是否是数值歧义
失败模型由哪个规则生成
5. Cost 优化 Agent

负责在模型正确之后降低成本。

职责：

统计参数量。
估计 initializer 内存占用。
统计 ONNX 文件大小。
比较同一任务的多个候选模型。
选择通过验证且 cost 最低的模型。
删除无用 initializer。
合并可合并的线性层。
用 1x1 Conv 替换不必要的大 kernel。
用更小的组合结构替换大常量表。
保留每次优化前后的 cost 记录。

只有已经通过正确性验证的模型才允许进行 cost 优化。

不得优化一个未通过验证的模型。

如果优化后模型变小但验证失败，必须回滚。

6. 提交 Agent

负责生成最终 submission.zip。

职责：

每个任务最多包含一个 ONNX 文件。
只打包通过验证的 ONNX 文件。
文件命名必须精确，例如：
task001.onnx
task002.onnx
...
task400.onnx
不得打包日志文件。
不得打包临时模型。
不得打包失败候选。
不得打包源代码。
不得打包 notebook。
不得打包缓存目录。
生成提交前必须检查 zip 内容。

如果一个任务有多个通过验证的候选模型，应选择 cost 最低的模型。

不得用更差的模型覆盖更好的模型。

7. 代码审查 Agent

负责维护代码质量。

职责：

检查代码是否确定性运行。
检查是否存在硬编码绝对路径。
检查脚本是否能从仓库根目录运行。
检查函数输入输出是否清晰。
检查异常是否被正确暴露。
检查日志是否有调试价值。
检查测试是否覆盖关键逻辑。
检查生成物是否可复现。
检查新增依赖是否必要。
检查是否破坏已有 pipeline。

代码审查 Agent 应该保持怀疑态度。

如果某个实现看起来能跑但不可解释、不可验证或不可复现，应要求重构。

七、仓库结构规范

推荐按以下类别组织代码：

数据读取
grid 编码
任务分析
规则匹配
ONNX 构造
ONNX 验证
cost 估计
提交打包
测试
日志

推荐目录结构：

project/
├── data/
├── outputs/
│   ├── onnx/
│   ├── logs/
│   ├── reports/
│   └── submission.zip
├── src/
├── tests/
├── notebooks/
├── AGENTS.md
├── README.md
└── requirements.txt

生成文件不得混入源码目录。

临时文件不得进入最终提交。

八、代码风格

优先使用普通、清晰、可维护的 Python。

要求：

公共函数应有类型注解。
关键函数应有 docstring。
不写死用户本地路径。
不依赖 notebook 才能运行。
不在核心 solver 中发起网络请求。
不使用隐藏的全局状态。
不吞掉异常。
错误信息必须具体。
批量运行结果应保存为 JSON 或 CSV。
每个模块职责应单一。

新增规则时，应同时提供：

matcher
builder
validator 路径
failure reason
cost estimate
合理的测试或合成样例

九、验证策略

一个模型只有同时满足以下条件，才算可用：

能被加载。
能通过 ONNX checker。
不包含禁用算子。
input/output shape 是静态的。
在所有训练样例上输出完全正确。
文件大小不超过限制。
参数量已统计。
内存占用已估计。
日志中记录了生成它的规则。
输出 grid 的解码是确定性的。

精确 grid equality 是必须的。

视觉相似不算正确。

局部正确不算正确。

只在一个样例上正确不算可靠。

十、Cost 记录策略

每个候选模型都应该记录以下字段：

task_id
rule_name
passed_validation
num_parameters
memory_bytes
file_size_bytes
estimated_cost
estimated_score
model_path
failure_reason

比赛 cost 的基本概念为：

cost = total_parameters + memory_footprint_in_bytes

估计分数：

score = max(1, 25 - ln(cost))

注意：本地估计分数不一定等于官方最终分数。

日志中必须把它标为 estimated score，而不是 guaranteed score。

十一、防止过拟合策略

不得优先构造只记忆训练样例的 lookup-table 模型。

不得只针对某个输入 grid 的绝对像素位置硬编码，除非该任务本身明确是固定位置规则。

应优先寻找具有泛化性的规则，例如：

对象移动
颜色角色转换
图形补全
对称关系
邻域规则
边界规则
形状关系
数量关系
包围盒关系

如果某个解法明显只是训练样例记忆，必须在日志中标记为 overfit。

overfit 模型不得作为稳定方案。

十二、规则匹配策略

规则匹配必须保守。

一个规则应该拒绝以下情况：

只能解释部分样例。
需要没有证据支持的假设。
依赖偶然像素位置。
对颜色角色判断不稳定。
对对象划分不稳定。
对尺寸变化无法解释。
可能无法通过 private benchmark。
只能通过硬编码训练输出完成。

如果规则不确定，应返回 POSSIBLE，而不是 MATCH。

十三、日志策略

每个任务的尝试都应生成机器可读日志。

推荐日志格式：

{
  "task_id": "task001",
  "status": "solved | failed | skipped",
  "rule_name": "...",
  "analysis_summary": "...",
  "validation_passed": true,
  "num_train_cases": 0,
  "failed_cases": [],
  "num_parameters": 0,
  "memory_bytes": 0,
  "file_size_bytes": 0,
  "estimated_cost": 0,
  "estimated_score": 0,
  "model_path": "...",
  "notes": []
}

日志必须能回答以下问题：

这个任务是否解决？
用了什么规则？
为什么认为规则匹配？
模型是否通过验证？
如果失败，失败在哪里？
参数量是多少？
文件大小是多少？
是否进入最终提交？
十四、测试要求

修改代码后，应运行相关测试。

如果修改了编码逻辑，应测试：

grid -> one-hot
one-hot -> grid
padding 行为
不同尺寸 grid
颜色 0 到 9

如果修改了 ONNX 构造逻辑，应测试：

ONNX checker
onnxruntime 推理
输出 shape
禁用算子检查
参数统计

如果修改了规则匹配逻辑，应测试：

正例是否匹配
反例是否拒绝
不确定样例是否返回 POSSIBLE
多样例一致性

如果修改了提交逻辑，应测试：

zip 文件是否生成
文件名是否正确
是否只包含 ONNX
是否排除失败模型
是否排除日志和临时文件

不得在未运行相关验证的情况下声称成功。

如果环境缺少依赖，必须明确说明哪些测试无法运行。

十五、依赖策略

优先使用稳定、轻量依赖。

推荐依赖：

numpy
onnx
onnxruntime
pytest
pandas

其中：

numpy 用于 grid 和 tensor 处理。
onnx 用于构造和检查模型。
onnxruntime 用于本地推理验证。
pytest 用于测试。
pandas 仅用于报告汇总。

不要随意引入大型依赖。

不要默认加入 PyTorch 或 TensorFlow。

如果确实需要新增重型依赖，必须说明原因。

十六、ONNX 设计偏好

在满足正确性的前提下，优先选择简单结构。

颜色替换

优先考虑：

1x1 Conv
固定 channel mixing
小型线性变换
平移

优先考虑：

小 kernel Conv
固定 offset kernel
局部邻域规则

优先考虑：

3x3 Conv
5x5 Conv
少层组合
mask 构造

优先考虑：

one-hot channel 线性组合
简单激活函数
固定阈值
组合规则

优先考虑：

最少层数
最少 initializer
最少中间 tensor

如果两个模型都正确，应按以下顺序选择：

参数更少
内存更小
文件更小
算子更简单
图结构更容易解释

十七、禁止行为

不得执行以下行为：

未验证就生成最终 ONNX。
验证失败仍加入 submission。
使用比赛禁止的 ONNX 算子。
使用动态 shape。
生成超过 1.44MB 的 ONNX。
静默忽略异常。
硬编码本机路径。
把训练输出直接塞进巨大常量表。
把临时文件打包进提交。
把日志文件打包进提交。
为了让样例通过而破坏泛化性。
用大模型代替规则分析作为默认方案。
在没有证据时声称任务已解决。
未说明失败原因就跳过任务。

十八、提交文件安全规则

submission.zip 中只能包含通过验证的 ONNX 文件。

允许的文件名形式：

task001.onnx
task002.onnx
...
task400.onnx

不允许包含：

*.json
*.csv
*.py
*.ipynb
*.txt
*.md
*.log
*.tmp
__pycache__/
outputs/logs/
outputs/reports/

生成提交前必须检查 zip 内容。

如果 zip 中出现非 ONNX 文件，应视为提交构建失败。

十九、失败处理原则

失败是正常情况，但必须可诊断。

当任务无法解决时，应记录：

task_id
已尝试的规则
每个规则失败原因
是否是分析失败
是否是 ONNX 构造失败
是否是验证失败
是否是约束检查失败
后续可能改进方向

不要伪造成功。

不要输出空洞的 “failed”。

失败日志必须帮助后续继续改进。

二十、与用户沟通原则

当向用户汇报时，应清楚说明：

完成了什么
哪些任务通过验证
哪些任务失败
失败原因是什么
当前 solved 数量
当前估计 cost
当前 submission 是否可用

不得夸大结果。

不得把本地 train 验证分数说成官方榜单分数。

不得把 estimated score 说成 guaranteed score。

二十一、最终准则

本项目的核心不是“写一个能跑的模型”。

本项目的核心是：

把 ARC 图形变换规则
稳定地、可验证地、低成本地
编译成合法 ONNX 网络。

任何代码、规则、模型、日志、提交文件，都必须服务于这个目标。

补充：每轮结束前更新 PROGRESS.md 和 EXPERIMENT_LOG.md

## Sandbox fallback note

Observed on 2026-06-07: Windows sandbox setup may fail before a command starts
with `windows sandbox: setup refresh failed with status exit code: 1`.
Detailed Codex logs showed the setup helper was trying to update ACL/write
access for `C:\Windows\Temp`, but this machine returned access denied. User
environment variables `TEMP` and `TMP` were changed to
`C:\Users\dell\.codex\tmp`; restart Codex/terminal for new processes to inherit
that value.

If the same setup failure appears again, retry the same command with
`sandbox_permissions: "require_escalated"` and the narrowest reasonable
`prefix_rule`. Do not stop to ask in chat first. Never use broad or destructive
fallback prefixes, and never bypass approval for destructive operations.
