# 双色球数据处理与分析系统

项目自动更新历史开奖数据，使用统计特征和 LightGBM 为号码评分，经过规则过滤后生成单式与复式候选，并在开奖后核对历史推荐。

> 彩票开奖结果具有随机性。本项目用于数据分析和策略回测，不保证收益。

## 三文件结构

生产代码已由原先的 85 个模块合并为 `py/` 下三个自包含文件：

| 文件 | 职责 | 原模块数 |
|---|---|---|
| `py/ssq_data_prep.py` | 数据准备：抓取、交叉核对、校验、原子落盘 | 15 |
| `py/ssq_analysis.py` | 数据分析：特征工程、模型训练、规则过滤、回测、报告 | 65 |
| `py/ssq_bonus_check.py` | 开奖分析：解析历史推荐、计算奖金、生成核对报告 | 13 |

三条流水线相互独立（原模块闭包互不相交），可单独运行：

```powershell
python py\ssq_data_prep.py
python py\ssq_analysis.py --compare-pools --non-interactive
python py\ssq_bonus_check.py
```

合并过程做了等价性验证：三个入口在相同数据与参数下产出的报告与合并前**逐字节一致**（仅报告生成时间戳不同）。一次性合并工具保留在 `tools/merge/`。

合并中需要人工裁决的部分记录在 `tools/merge/signatures.json`（公开签名与依赖装配）和 `tools/merge/overrides.json`（同名冲突取哪一个实现）。

## 测试

原测试套件按旧模块名导入，`py/ssq_compat.py` 把这些名字映射到合并后的模块，因此测试仍然直接验证合并后的代码：

```powershell
python py\run_tests.py
```

其中断言「旧 85 模块结构」本身的用例（re-export 的 `is` 同一性、旧门面是否被生产代码导入、旧模块间路径对象共享等）已列入 `py/known_obsolete_tests.py` 并报告为跳过——这些边界正是合并要消除的对象。其余 289 个用例全部通过，覆盖真实行为。

## 流程

1. `ssq_data_prep.py` 抓取并交叉核对 TXT/HTML 数据，校验、合并并原子保存开奖记录，同时统一三个入口的数据规范化。
2. `ssq_analysis.py` 负责特征工程与号码评分、候选池构建与组合选择、历史规则审计、滚动回测、完整分析流程编排以及报告生成，并提供命令行选项解析。
3. `ssq_bonus_check.py` 解析投注并计算参考奖金，格式化核对报告并编排核对流程。
4. GitHub Actions 每周一、三、五北京时间 06:00 自动运行并提交结果。

> 合并前每个步骤由多个细粒度模块承担，文件与文件的对应关系见上表及 `tools/merge/overrides.json`。

## 选号策略

红球基础评分综合时间衰减频率、遗漏值和机器学习概率。基础分不直接解释为开奖概率：最终组合排序对排名两端降权、对中间排名加权，以落实“高分和低分都不盲从”的选号逻辑。默认候选池为 17 个号码：

- 高分段 4 个；
- 中间分段 9 个；
- 低分段 4 个。

默认红球权重为频率 0.4、遗漏 0.5、机器学习 0.1，蓝球权重为频率 0.6、机器学习 0.4。权重使用连续 200 期严格滚动结果评估：较早 100 期用于调参，最近 100 期用于验证，报告分别展示四种候选池在两个窗口的覆盖、单注命中、3+ 红比例和排序增益。策略选择优先看最近窗口及跨窗口稳定性，不以聚合 ROI 或一次高奖作为依据。

候选组合先通过和值、跨度、连号、三区、近期重合、尾数和关联号等高覆盖硬规则。AC 值、质合比、大小比、奇偶比、余数路、首尾范围和斜连号作为软评分，不再一票否决真实开奖中并不少见的形态。最后按模型信号和结构均衡度排序，并优先保证任意两注最多重合 4 个红球，选出 10 注覆盖更分散的单式组合；若候选不足，重合限制会逐级放宽，报告会声明最终实际最大重合数。

报告同时展示每条规则的独立历史覆盖率，以及全部硬规则串联后的累计覆盖率。后者直接反映为了排除特殊形态而放弃了多少真实开奖，随机撞号不混入这项统计。

报告默认使用最近 200 期做严格滚动回测，每一期只使用此前数据重新训练模型。回测只计算上述 10 注单式，不把 7+N 复式成本和奖金混入结果。除奖金和 ROI 外，报告还对照候选池覆盖、单注红球命中、3+ 红比例，以及最终 10 注相对全部有效候选的排序增益，避免一次偶然中奖主导策略判断。

回测和开奖核对中的奖金采用固定参考金额，便于不同策略使用同一尺度比较；一等奖、二等奖的实际浮动奖金仍以官方派奖结果为准，因此报告中的收益和 ROI 都是参考值。

高端、中段、低端的实际命中分布会按各段包含的排名数量换算随机基线。不能直接比较原始命中数，因为中段包含 9 个排名位置，高端和低端各只有 4 个。

随机排除库作为反撞号扰动保留。基础种子与目标期号共同派生当期种子，LightGBM 同时启用确定性训练；在数据、参数和依赖环境相同的前提下，同一期完整结果可复现，不同期不会永久排除同一批组合。报告记录完整开奖历史指纹、运行时依赖版本、模型特征顺序和 LightGBM 训练参数，便于追溯复现条件。回测与正式选号使用相同规则。反撞号不代表开奖概率提升。

## 本地运行

```powershell
python -m pip install -r py\requirements.txt
python py\ssq_data_prep.py
python py\ssq_analysis.py --compare-pools --non-interactive
python py\ssq_bonus_check.py
```

分析器支持以下参数：

```text
--backtest-periods N    回测期数
--rejection-size N      随机排除库规模
--seed N                随机种子
--pool-mode MODE        mixed、high、middle 或 low
--compare-pools         在同一次回测中比较四种候选池
--non-interactive       跳过键盘等待，适用于自动化运行
--rule-audit-periods N  统计每条规则对真实开奖的独立覆盖率
```

快速验证示例：

```powershell
python py\ssq_analysis.py --backtest-periods 1 --rejection-size 0 --compare-pools
python -m pip install -r py\requirements-dev.txt
ruff check py
python py\run_tests.py
```

## 输出

- `report/ssq_analysis_output_*.txt`：分析和推荐报告；
- `report/ssq_bonus_check_*.txt`：开奖核对报告；
- `latest_ssq_analysis.txt`：最新分析报告副本；
- `latest_ssq_calculation.txt`：最新核对报告副本。
