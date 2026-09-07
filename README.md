# 双色球数据处理与分析系统

项目自动更新历史开奖数据，使用统计特征和 LightGBM 为号码评分，经过规则过滤后生成单式与复式候选，并在开奖后核对历史推荐。

> 彩票开奖结果具有随机性。本项目用于数据分析和策略回测，不保证收益。

## 流程

1. `ssq_data_processor.py` 从两个数据源更新 `shuangseqiu.csv`。
2. `ssq_analyzer.py` 完成特征工程、滚动回测、号码评分和组合筛选。
3. `ssq_bonus_calculation.py` 使用最新开奖结果核对对应预测报告。
4. GitHub Actions 每周一、三、五北京时间 06:00 自动运行并提交结果。

## 选号策略

红球评分综合时间衰减频率、遗漏值和机器学习概率。默认候选池为 17 个号码：

- 高分段 4 个；
- 中间分段 9 个；
- 低分段 4 个。

候选组合先通过和值、跨度、连号、三区、近期重合、尾数和关联号等高覆盖硬规则。AC 值、质合比、大小比、奇偶比、余数路、首尾范围和斜连号作为软评分，不再一票否决真实开奖中并不少见的形态。最后按模型信号和结构均衡度选出前 10 注。

随机排除库作为反撞号扰动保留，使用固定种子以保证结果可复现。它不代表开奖概率提升。

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
python -m unittest py.test_ssq_analyzer -v
```

## 输出

- `report/ssq_analysis_output_*.txt`：分析和推荐报告；
- `report/ssq_bonus_check_*.txt`：开奖核对报告；
- `latest_ssq_analysis.txt`：最新分析报告副本；
- `latest_ssq_calculation.txt`：最新核对报告副本。
