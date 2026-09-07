# 双色球数据处理与分析系统

项目自动更新历史开奖数据，使用统计特征和 LightGBM 为号码评分，经过规则过滤后生成单式与复式候选，并在开奖后核对历史推荐。

> 彩票开奖结果具有随机性。本项目用于数据分析和策略回测，不保证收益。

## 流程

1. `ssq_data_sources.py` 获取并交叉核对 TXT/HTML 数据，`ssq_data_store.py` 校验、合并并原子保存开奖记录，`ssq_data_workflow.py` 编排更新流程；`ssq_draw_data.py` 统一三个入口的数据规范化，`ssq_data_processor.py` 保留为命令入口。
2. `ssq_config.py` 管理策略默认值与命令行选项，`ssq_modeling.py` 负责特征工程和号码评分，`ssq_selection.py` 构建候选池并执行组合选择，`ssq_backtesting.py` 负责历史审计与滚动回测，`ssq_workflow.py` 编排完整分析流程，`ssq_rules.py` 管理硬规则与软评分，`ssq_reporting.py` 生成报告；`ssq_analyzer.py` 保留为命令入口。
3. `ssq_prizes.py` 解析投注并计算参考奖金，`ssq_bonus_reporting.py` 格式化核对报告，`ssq_bonus_workflow.py` 编排核对流程；`ssq_bonus_calculation.py` 保留为命令入口。
4. GitHub Actions 每周一、三、五北京时间 06:00 自动运行并提交结果。

## 选号策略

红球基础评分综合时间衰减频率、遗漏值和机器学习概率。基础分不直接解释为开奖概率：最终组合排序对排名两端降权、对中间排名加权，以落实“高分和低分都不盲从”的选号逻辑。默认候选池为 17 个号码：

- 高分段 4 个；
- 中间分段 9 个；
- 低分段 4 个。

默认红球权重为频率 0.4、遗漏 0.5、机器学习 0.1，蓝球权重为频率 0.6、机器学习 0.4。权重使用连续 200 期严格滚动结果评估：较早 100 期用于调参，最近 100 期用于验证，优先比较候选池覆盖和单注红球命中等稳定指标，不以 ROI 或一次高奖作为选择依据。

候选组合先通过和值、跨度、连号、三区、近期重合、尾数和关联号等高覆盖硬规则。AC 值、质合比、大小比、奇偶比、余数路、首尾范围和斜连号作为软评分，不再一票否决真实开奖中并不少见的形态。最后按模型信号和结构均衡度排序，并优先保证任意两注最多重合 4 个红球，选出 10 注覆盖更分散的单式组合。

报告同时展示每条规则的独立历史覆盖率，以及全部硬规则串联后的累计覆盖率。后者直接反映为了排除特殊形态而放弃了多少真实开奖，随机撞号不混入这项统计。

报告默认使用最近 200 期做严格滚动回测，每一期只使用此前数据重新训练模型。回测只计算上述 10 注单式，不把 7+N 复式成本和奖金混入结果。除奖金和 ROI 外，报告还对照候选池覆盖、单注红球命中、3+ 红比例，以及最终 10 注相对全部有效候选的排序增益，避免一次偶然中奖主导策略判断。

回测和开奖核对中的奖金采用固定参考金额，便于不同策略使用同一尺度比较；一等奖、二等奖的实际浮动奖金仍以官方派奖结果为准，因此报告中的收益和 ROI 都是参考值。

高端、中段、低端的实际命中分布会按各段包含的排名数量换算随机基线。不能直接比较原始命中数，因为中段包含 9 个排名位置，高端和低端各只有 4 个。

随机排除库作为反撞号扰动保留。基础种子与目标期号共同派生当期种子，因此同一期结果可复现，不同期不会永久排除同一批组合；回测与正式选号使用相同规则。它不代表开奖概率提升。

## 本地运行

```powershell
python -m pip install -r py\requirements.txt
python py\ssq_data_processor.py
python py\ssq_analyzer.py --compare-pools --non-interactive
python py\ssq_bonus_calculation.py
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
python py\ssq_analyzer.py --backtest-periods 1 --rejection-size 0 --compare-pools
python -m pip install -r py\requirements-dev.txt
ruff check py
python -m unittest py.test_ssq_analyzer -v
```

## 输出

- `report/ssq_analysis_output_*.txt`：分析和推荐报告；
- `report/ssq_bonus_check_*.txt`：开奖核对报告；
- `latest_ssq_analysis.txt`：最新分析报告副本；
- `latest_ssq_calculation.txt`：最新核对报告副本。
