# 策略风险分析系统

基于 Streamlit 的交易策略风险分析工具，用于评估策略表现和风险指标，展示交易信号图

## 功能特性

- **数据加载**：自动加载交易记录（jiaoyi.csv）
- **风险指标计算**：计算收益、风险、交易统计等19项指标
- **可视化图表**：交易信号图、累计收益曲线、回撤分析等
- **基准对比**：与黄金期货基准进行对比分析
- **对数轴显示**：支持对数轴查看累计收益曲线

## 安装依赖

```bash
pip install streamlit pandas numpy plotly
```

## 使用方法

```bash
streamlit run risk.py
```

## 数据

 `jiaoyi.csv` ,使用GBK格式编码


## 主要指标

### 收益指标
- Total Returns（策略收益）
- Total Annualized Returns（策略年化收益）
- Alpha（阿尔法）
- Beta（贝塔）
- AEI（日均超额收益）
- 超额收益
- 对数轴上的超额收益

### 风险指标
- Sharpe（夏普比率）
- Sortino（索提诺比率）
- Information Ratio（信息比率）
- Algorithm Volatility（策略波动率）
- Benchmark Volatility（基准波动率）
- Max Drawdown（最大回撤）
- Downside Risk（下行波动率）

### 交易统计
- 胜率
- 日胜率
- 盈亏比
- 超额收益最大回撤
- 超额收益夏普比率

## 注意事项

- 无风险利率设置为4%（年化）
- 基准数据使用黄金期货（AU0主力连续合约）
- 数据文件需要使用GBK编码
- 对数轴模式下自动处理负值数据

## 文件说明

- `risk.py`：主程序文件
- `jiaoyi.csv`：交易记录数据文件
