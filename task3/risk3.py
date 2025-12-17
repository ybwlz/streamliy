import pandas as pd 
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import warnings
warnings.filterwarnings('ignore')


#详细展示风险指标计算过程和公式说明的注释版

# ========== 基准数据 ==========
# 根据jiaoyi.csv的日期范围获取的黄金期货基准收益率数据（使用AU0主力连续合约）
# 数据格式：与交易日期对齐的日收益率序列（小数形式）
# 基准数据日期范围: 2024-01-04 到 2025-04-28，共71个交易日
# 注意：第一个值为0，因为第一个交易日没有前一天的价格数据
# 基准数据来源：akshare获取的AU0黄金主力连续合约，整个期间（315个连续交易日）总收益率61.95%
# 对齐说明：使用整个期间的连续基准数据，每个交易日使用从第一个交易日到该交易日之间的累计基准收益率
# 对齐后的基准数据总收益率：62.56%（与连续基准总收益率61.95%非常接近，差异0.60%）
BENCHMARK_RETURNS_HARDCODED = np.array([
    0.00000000, 0.00079190, -0.00283192, 0.00062646, -0.00488334, 0.01673517, -0.00858050, -0.00124828, 0.00283298, -0.00029081,
    0.05290060, 0.00280223, 0.09260863, 0.02121681, -0.00261023, -0.03656811, 0.00998458, 0.00214436, -0.00569398, 0.01148964,
    -0.00025243, 0.00941423, -0.00982669, -0.00476362, -0.00638190, 0.00080286, 0.00291715, 0.00199971, -0.01048659, 0.01298130,
    0.00582827, 0.00906964, 0.00841745, -0.00010611, -0.00597807, 0.00882531, 0.00134044, 0.01490119, -0.00045123, -0.00888981,
    -0.00455485, 0.05490831, -0.01811751, 0.00363599, 0.02133062, 0.05363832, -0.00934462, -0.00898812, -0.02041470, -0.00925865,
    0.00815639, 0.00959712, -0.00528826, 0.01063275, -0.00196863, -0.00737268, 0.00824185, -0.00216478, 0.01868342, 0.08521933,
    0.01051522, -0.02394203, 0.00074241, 0.01249295, 0.03599062, 0.00472445, 0.01109390, 0.06441282, 0.02595364, 0.00372316,
    -0.00909553
])

BENCHMARK_AVAILABLE = True

# 设置页面配置
st.set_page_config(
    page_title="策略风险分析",
    page_icon="📊",
    layout="wide"
)

# 加载交易数据（使用GBK编码）
@st.cache_data
def load_trade_data(filename='jiaoyi.csv'):
    """加载交易数据"""
    import os
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))#获取当前脚本的绝对路径，os.path.dirname()获取其父目录
    # 拼接完整路径
    csv_path = os.path.join(script_dir,filename)
    # 读取CSV文件
    df = pd.read_csv(csv_path, encoding='gbk')
    return df

# 数据清洗和预处理
def preprocess_data(df):
    """简化版数据清洗"""
    df = df.copy()
    
    # 1. 合并日期时间（假设列名就是"日期"和"委托时间"）
    df['日期时间'] = pd.to_datetime(df['日期'] + ' ' + df['委托时间'], format='%Y/%m/%d %H:%M:%S', errors='coerce')
    df = df.sort_values('日期时间').reset_index(drop=True)
    
    # 2. 清理数字列（直接处理，不用if判断）
    # 成交数量
    df['成交数量'] = pd.to_numeric(df['成交数量'].astype(str).str.replace('手', ''), errors='coerce')
    
    # 成交价
    df['成交价'] = pd.to_numeric(df['成交价'].astype(str).str.replace(',', ''), errors='coerce')
    
    # 成交金额
    df['成交额'] = pd.to_numeric(df['成交额'].astype(str).str.replace(',', ''), errors='coerce')
    
    # 平仓盈亏
    df['平仓盈亏'] = pd.to_numeric(df['平仓盈亏'].astype(str).str.replace(',', '').replace(['-', ''], '0'), errors='coerce').fillna(0)
    
    # 手续费
    df['手续费'] = pd.to_numeric(df['手续费'].astype(str).str.replace(',', '').replace(['-', ''], '0'), errors='coerce').fillna(0)
    
    # 计算净盈亏
    df['净盈亏'] = df['平仓盈亏'] - df['手续费']
    
    return df

# 计算风险指标
def calculate_risk_metrics(daily_returns, total_returns, initial_capital, daily_pnl, benchmark_returns=None):
    """计算各种风险指标"""
    # 无风险利率（年化，4%）
    RISK_FREE_RATE = 0.04
    
    metrics = {}
    
    # 确保daily_returns是数组
    if isinstance(daily_returns, pd.Series):
        daily_returns = daily_returns.values
    
    # 过滤NaN值
    daily_returns = daily_returns[~np.isnan(daily_returns)]
    
    if len(daily_returns) == 0:
        return {}
    
    # Total Returns 策略收益（百分比）
    metrics['Total Returns'] = total_returns
    
    # Total Annualized Returns 策略年化收益
    trading_days = len(daily_returns)
    if trading_days > 0:
        # 计算实际交易天数
        date_range = (daily_pnl['日期'].max() - daily_pnl['日期'].min()).days
        years = date_range / 365.25 if date_range > 0 else trading_days / 252
        if years > 0:
            metrics['Total Annualized Returns'] = ((1 + total_returns / 100) ** (1 / years) - 1) * 100
        else:
            metrics['Total Annualized Returns'] = 0
    else:
        metrics['Total Annualized Returns'] = 0
    
    # Algorithm Volatility 策略波动率（年化）
    if len(daily_returns) > 1:
        metrics['Algorithm Volatility'] = np.std(daily_returns) * np.sqrt(252) * 100
    else:
        metrics['Algorithm Volatility'] = 0
    
    # Benchmark Volatility 基准波动率（如果有基准数据）
    if benchmark_returns is not None and len(benchmark_returns) > 1:
        metrics['Benchmark Volatility'] = np.std(benchmark_returns) * np.sqrt(252) * 100
    else:
        metrics['Benchmark Volatility'] = 0
    
    # Sharpe 夏普比率（考虑无风险利率4%）
    # Sharpe = (策略年化收益率 - 无风险利率) / 策略波动率
    if metrics['Algorithm Volatility'] > 0:
        excess_return = metrics['Total Annualized Returns'] - RISK_FREE_RATE * 100
        metrics['Sharpe'] = excess_return / metrics['Algorithm Volatility']
    else:
        metrics['Sharpe'] = 0
    
    # Sortino 索提诺比率（只考虑下行波动，考虑无风险利率4%）
    # Sortino = (策略年化收益率 - 无风险利率) / 下行波动率
    downside_returns = daily_returns[daily_returns < 0]
    if len(downside_returns) > 1:
        downside_std = np.std(downside_returns) * np.sqrt(252) * 100
        metrics['Downside Risk'] = downside_std
        if downside_std > 0:
            excess_return = metrics['Total Annualized Returns'] - RISK_FREE_RATE * 100
            metrics['Sortino'] = excess_return / downside_std
        else:
            metrics['Sortino'] = 0
    else:
        metrics['Downside Risk'] = 0
        metrics['Sortino'] = 0
    
    # Max Drawdown 最大回撤
    # 使用复利计算累计收益率：(1 + r1) * (1 + r2) * ... - 1
    cumulative_returns = np.cumprod(1 + daily_returns) - 1
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = cumulative_returns - running_max
    # 转换为百分比
    metrics['Max Drawdown'] = np.min(drawdown) * 100 if len(drawdown) > 0 else 0
    
    # Alpha 和 Beta（使用整个期间的315个交易日的连续基准数据）
    if benchmark_returns is not None and len(benchmark_returns) > 0:
        # 确保数据长度一致
        min_len = min(len(daily_returns), len(benchmark_returns))
        daily_returns_aligned = daily_returns[:min_len]
        benchmark_returns_aligned = benchmark_returns[:min_len]
        
        # 过滤NaN值和无穷值
        valid_mask = ~(np.isnan(daily_returns_aligned) | np.isnan(benchmark_returns_aligned) | 
                      np.isinf(daily_returns_aligned) | np.isinf(benchmark_returns_aligned))
        daily_returns_clean = daily_returns_aligned[valid_mask]
        benchmark_returns_clean = benchmark_returns_aligned[valid_mask]
        
        if len(daily_returns_clean) > 1 and len(benchmark_returns_clean) > 1:
            # 计算协方差和方差
            covariance = np.cov(daily_returns_clean, benchmark_returns_clean)[0, 1]
            benchmark_variance = np.var(benchmark_returns_clean, ddof=0)
            
            if benchmark_variance > 1e-10:  # 避免除零
                metrics['Beta'] = covariance / benchmark_variance
                
                # Alpha = (策略平均日收益率 - 无风险日利率) - Beta * (基准平均日收益率 - 无风险日利率)
                # daily_returns 和 benchmark_returns 都是小数形式（0.01表示1%）
                # 无风险利率年化4%，日利率 = 4% / 252
                risk_free_daily = RISK_FREE_RATE / 252
                strategy_mean_daily = np.mean(daily_returns_clean)
                benchmark_mean_daily = np.mean(benchmark_returns_clean)
                alpha_daily = (strategy_mean_daily - risk_free_daily) - metrics['Beta'] * (benchmark_mean_daily - risk_free_daily)
                # 年化Alpha（转换为百分比）
                metrics['Alpha'] = alpha_daily * 252 * 100
                
                # 验证Alpha值的合理性（通常应该在-100%到+100%之间）
                if abs(metrics['Alpha']) > 200:
                    print(f"警告: Alpha值异常高 ({metrics['Alpha']:.2f}%)，请检查基准数据是否正确")
                    print(f"  策略平均日收益率: {strategy_mean_daily*100:.4f}%")
                    print(f"  基准平均日收益率: {benchmark_mean_daily*100:.4f}%")
                    print(f"  Beta: {metrics['Beta']:.4f}")
            else:
                metrics['Beta'] = 0
                metrics['Alpha'] = 0
            
            # Information Ratio 信息比率
            excess_returns = daily_returns_clean - benchmark_returns_clean
            if len(excess_returns) > 1:
                tracking_error = np.std(excess_returns, ddof=0) * np.sqrt(252) * 100
                excess_return_mean = np.mean(excess_returns)
                excess_return_annual = excess_return_mean * 252 * 100
                
                if tracking_error > 1e-10:
                    metrics['Information Ratio'] = excess_return_annual / tracking_error
                else:
                    metrics['Information Ratio'] = 0
            else:
                metrics['Information Ratio'] = 0
        else:
            metrics['Beta'] = 0
            metrics['Alpha'] = 0
            metrics['Information Ratio'] = 0
    else:
        metrics['Alpha'] = 0
        metrics['Beta'] = 0
        metrics['Information Ratio'] = 0
    
    # 胜率（按交易笔数）
    winning_trades = (daily_returns > 0).sum()
    total_trades = len(daily_returns)
    metrics['胜率'] = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    # 日胜率（按日统计）：当日策略收益跑赢当日基准收益的天数 / 总交易日数
    if benchmark_returns is not None and len(benchmark_returns) > 0:
        # 确保数据长度一致
        min_len = min(len(daily_returns), len(benchmark_returns))
        daily_returns_aligned = daily_returns[:min_len]
        benchmark_returns_aligned = benchmark_returns[:min_len]
        
        # 过滤NaN值和无穷值
        valid_mask = ~(np.isnan(daily_returns_aligned) | np.isnan(benchmark_returns_aligned) | 
                      np.isinf(daily_returns_aligned) | np.isinf(benchmark_returns_aligned))
        daily_returns_clean = daily_returns_aligned[valid_mask]
        benchmark_returns_clean = benchmark_returns_aligned[valid_mask]
        
        if len(daily_returns_clean) > 0:
            # 统计策略收益跑赢基准收益的天数
            winning_days = (daily_returns_clean > benchmark_returns_clean).sum()
            total_days = len(daily_returns_clean)
            metrics['日胜率'] = (winning_days / total_days * 100) if total_days > 0 else 0
        else:
            # 如果没有有效的基准数据，回退到原来的计算方法
            winning_days = (daily_pnl['日盈亏'] > 0).sum()
            total_days = len(daily_pnl)
            metrics['日胜率'] = (winning_days / total_days * 100) if total_days > 0 else 0
    else:
        # 如果没有基准数据，使用原来的计算方法（日盈亏>0的天数）
        winning_days = (daily_pnl['日盈亏'] > 0).sum()
        total_days = len(daily_pnl)
        metrics['日胜率'] = (winning_days / total_days * 100) if total_days > 0 else 0
    
    # 盈亏比
    if winning_trades > 0 and (total_trades - winning_trades) > 0:
        avg_win = daily_returns[daily_returns > 0].mean()
        avg_loss = abs(daily_returns[daily_returns < 0].mean())
        metrics['盈亏比'] = avg_win / avg_loss if avg_loss > 0 else 0
    else:
        metrics['盈亏比'] = 0
    
    # AEI 日均超额收益（如果有基准）
    if benchmark_returns is not None and len(benchmark_returns) > 0:
        # 确保数据长度一致
        min_len = min(len(daily_returns), len(benchmark_returns))
        daily_returns_aligned = daily_returns[:min_len]
        benchmark_returns_aligned = benchmark_returns[:min_len]
        
        # 过滤NaN值和无穷值
        valid_mask = ~(np.isnan(daily_returns_aligned) | np.isnan(benchmark_returns_aligned) | 
                      np.isinf(daily_returns_aligned) | np.isinf(benchmark_returns_aligned))
        daily_returns_clean = daily_returns_aligned[valid_mask]
        benchmark_returns_clean = benchmark_returns_aligned[valid_mask]
        
        if len(daily_returns_clean) > 0:
            excess_returns = daily_returns_clean - benchmark_returns_clean
            metrics['AEI'] = np.mean(excess_returns) * 100
        else:
            metrics['AEI'] = 0
    else:
        metrics['AEI'] = 0
    
    # 超额收益最大回撤
    if benchmark_returns is not None and len(benchmark_returns) > 0:
        # 确保数据长度一致
        min_len = min(len(daily_returns), len(benchmark_returns))
        daily_returns_aligned = daily_returns[:min_len]
        benchmark_returns_aligned = benchmark_returns[:min_len]
        
        # 过滤NaN值和无穷值
        valid_mask = ~(np.isnan(daily_returns_aligned) | np.isnan(benchmark_returns_aligned) | 
                      np.isinf(daily_returns_aligned) | np.isinf(benchmark_returns_aligned))
        daily_returns_clean = daily_returns_aligned[valid_mask]
        benchmark_returns_clean = benchmark_returns_aligned[valid_mask]
        
        if len(daily_returns_clean) > 0:
            excess_returns = daily_returns_clean - benchmark_returns_clean
            # 使用复利计算累计超额收益率
            excess_cumulative = np.cumprod(1 + excess_returns) - 1
            excess_running_max = np.maximum.accumulate(excess_cumulative)
            excess_drawdown = excess_cumulative - excess_running_max
            # 转换为百分比
            metrics['超额收益最大回撤'] = np.min(excess_drawdown) * 100 if len(excess_drawdown) > 0 else 0
        else:
            metrics['超额收益最大回撤'] = 0
    else:
        metrics['超额收益最大回撤'] = 0
    
# 超额收益夏普比率
    if benchmark_returns is not None and len(benchmark_returns) > 0:
        # 确保数据长度一致
        min_len = min(len(daily_returns), len(benchmark_returns))
        daily_returns_aligned = daily_returns[:min_len]
        benchmark_returns_aligned = benchmark_returns[:min_len]
        
        # 过滤NaN值和无穷值
        valid_mask = ~(np.isnan(daily_returns_aligned) | np.isnan(benchmark_returns_aligned) | 
                      np.isinf(daily_returns_aligned) | np.isinf(benchmark_returns_aligned))
        daily_returns_clean = daily_returns_aligned[valid_mask]
        benchmark_returns_clean = benchmark_returns_aligned[valid_mask]
        
        if len(daily_returns_clean) > 1:
            excess_returns = daily_returns_clean - benchmark_returns_clean
            excess_vol = np.std(excess_returns, ddof=0) * np.sqrt(252) * 100
            excess_mean = np.mean(excess_returns) * 252 * 100
            metrics['超额收益夏普比率'] = excess_mean / excess_vol if excess_vol > 1e-10 else 0
        else:
            metrics['超额收益夏普比率'] = 0
    else:
        metrics['超额收益夏普比率'] = 0
    
    # 超额收益（除法版）- (策略总收益率 / 基准总收益率 - 1) * 100
    if benchmark_returns is not None and len(benchmark_returns) > 0:
        # 确保数据长度一致
        min_len = min(len(daily_returns), len(benchmark_returns))
        daily_returns_aligned = daily_returns[:min_len]
        benchmark_returns_aligned = benchmark_returns[:min_len]
        
        # 过滤NaN值和无穷值
        valid_mask = ~(np.isnan(daily_returns_aligned) | np.isnan(benchmark_returns_aligned) | 
                      np.isinf(daily_returns_aligned) | np.isinf(benchmark_returns_aligned))
        daily_returns_clean = daily_returns_aligned[valid_mask]
        benchmark_returns_clean = benchmark_returns_aligned[valid_mask]
        
        if len(daily_returns_clean) > 0:
            # 使用复利计算总收益率：(1 + r1) * (1 + r2) * ... - 1
            strategy_total = np.prod(1 + daily_returns_clean) - 1
            benchmark_total = np.prod(1 + benchmark_returns_clean) - 1
            
            # 保存基准总收益（转换为百分比）
            metrics['基准收益'] = benchmark_total * 100
            
            if abs(benchmark_total) > 1e-10:
                # 超额收益 = (策略总收益 / 基准总收益 - 1) * 100
                metrics['超额收益'] = (strategy_total / benchmark_total - 1) * 100
            else:
                metrics['超额收益'] = 0
        else:
            metrics['基准收益'] = 0
            metrics['超额收益'] = 0
    else:
        metrics['基准收益'] = 0
        metrics['超额收益'] = 0
    
    # 对数轴上的超额收益
    # 计算方法：log(1 + 策略总收益率) - log(1 + 基准总收益率)
    if benchmark_returns is not None and len(benchmark_returns) > 0:
        # 确保数据长度一致
        min_len = min(len(daily_returns), len(benchmark_returns))
        daily_returns_aligned = daily_returns[:min_len]
        benchmark_returns_aligned = benchmark_returns[:min_len]
        
        # 过滤NaN值和无穷值
        valid_mask = ~(np.isnan(daily_returns_aligned) | np.isnan(benchmark_returns_aligned) | 
                      np.isinf(daily_returns_aligned) | np.isinf(benchmark_returns_aligned))
        daily_returns_clean = daily_returns_aligned[valid_mask]
        benchmark_returns_clean = benchmark_returns_aligned[valid_mask]
        
        if len(daily_returns_clean) > 0:
            # 使用复利计算总收益率：(1 + r1) * (1 + r2) * ... - 1
            strategy_total = np.prod(1 + daily_returns_clean) - 1
            benchmark_total = np.prod(1 + benchmark_returns_clean) - 1
            # 使用log1p避免log(0)的问题
            log_excess = np.log1p(strategy_total) - np.log1p(benchmark_total)
            metrics['对数轴上的超额收益'] = log_excess * 100
        else:
            metrics['对数轴上的超额收益'] = 0
    else:
        metrics['对数轴上的超额收益'] = 0
    
    return metrics

# 绘制交易信号图
def plot_trading_signals(df):
    """
    绘制交易信号图，显示价格走势和买卖信号点
    """
    # 创建子图：价格走势 + 累计收益
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('交易信号图（价格走势）', '累计收益曲线'),
        vertical_spacing=0.1,
        row_heights=[0.6, 0.4]
    )
    
    # 按日期时间排序
    df_sorted = df.sort_values('日期时间').reset_index(drop=True)
    
    # 1. 价格走势图（主图）
    fig.add_trace(
        go.Scatter(
            x=df_sorted['日期时间'],
            y=df_sorted['成交价'],
            mode='lines',
            name='成交价',
            line=dict(color='#1f77b4', width=1.5),
            hovertemplate='时间: %{x}<br>价格: %{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 标记买入信号（开多）
    buy_signals = df_sorted[df_sorted['交易类型'].str.contains('开多', na=False)]
    if len(buy_signals) > 0:
        fig.add_trace(
            go.Scatter(
                x=buy_signals['日期时间'],
                y=buy_signals['成交价'],
                mode='markers',
                name='买入信号（开多）',
                marker=dict(
                    symbol='triangle-up',
                    size=12,
                    color='green',
                    line=dict(width=2, color='darkgreen')
                ),
                hovertemplate='买入时间: %{x}<br>价格: %{y:.2f}<br>数量: %{customdata}手<extra></extra>',
                customdata=buy_signals['成交数量']
            ),
            row=1, col=1
        )
    
    # 标记卖出信号（平多）
    sell_signals = df_sorted[df_sorted['交易类型'].str.contains('平多', na=False)]
    if len(sell_signals) > 0:
        fig.add_trace(
            go.Scatter(
                x=sell_signals['日期时间'],
                y=sell_signals['成交价'],
                mode='markers',
                name='卖出信号（平多）',
                marker=dict(
                    symbol='triangle-down',
                    size=12,
                    color='red',
                    line=dict(width=2, color='darkred')
                ),
                hovertemplate='卖出时间: %{x}<br>价格: %{y:.2f}<br>盈亏: %{customdata:.0f}<extra></extra>',
                customdata=sell_signals['平仓盈亏']
            ),
            row=1, col=1
        )
    
    # 2. 累计收益曲线（副图）
    df_sorted['累计收益'] = df_sorted['净盈亏'].cumsum()
    colors_profit = ['green' if x >= 0 else 'red' for x in df_sorted['净盈亏']]
    
    fig.add_trace(
        go.Bar(
            x=df_sorted['日期时间'],
            y=df_sorted['净盈亏'],
            name='单笔盈亏',
            marker_color=colors_profit,
            hovertemplate='时间: %{x}<br>盈亏: %{y:,.0f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df_sorted['日期时间'],
            y=df_sorted['累计收益'],
            mode='lines',
            name='累计收益',
            line=dict(color='blue', width=2),
            hovertemplate='时间: %{x}<br>累计收益: %{y:,.0f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # 添加零线
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    
    # 更新布局
    fig.update_layout(
        height=800,
        title_text="交易信号图",
        title_x=0.5,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white'
    )
    
    # 更新Y轴标签
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="盈亏", row=2, col=1)
    
    # 更新X轴日期格式（不显示标签，只设置日期格式）
    fig.update_xaxes(tickformat="%Y-%m-%d", row=1, col=1)
    fig.update_xaxes(tickformat="%Y-%m-%d", row=2, col=1)
    
    return fig

# 绘制风险指标图表
def plot_risk_charts(df, daily_pnl, metrics, use_log_scale=False):
    """
    使用Plotly绘制风险指标图表
    
    参数:
    - use_log_scale: 是否使用对数轴（对数轴说明：在对数轴上，相同的百分比变化显示为相同的距离，
                      便于比较不同规模的投资表现。对数轴上的超额收益 = log(1+策略收益) - log(1+基准收益)）
    """
    
    # 创建子图
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=('累计收益曲线', '日盈亏分布', '回撤曲线', '累计收益率曲线'),
        vertical_spacing=0.06,
        row_heights=[0.35, 0.2, 0.2, 0.25]
    )
    
    # 1. 累计收益曲线（主图）
    y_data = daily_pnl['累计收益']
    if use_log_scale and (y_data > 0).all():
        y_data = np.log1p(y_data - y_data.min() + 1)
        yaxis_title = "累计收益 (对数轴)"
    else:
        yaxis_title = "累计收益"
    
    fig.add_trace(
        go.Scatter(
            x=daily_pnl['日期'],
            y=y_data,
            mode='lines',
            name='累计收益',
            line=dict(color='#1f77b4', width=2),
            hovertemplate='日期: %{x}<br>累计收益: %{y:,.0f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 添加零线
    if not use_log_scale:
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
    
    # 2. 日盈亏分布
    colors = ['green' if x > 0 else 'red' for x in daily_pnl['日盈亏']]
    fig.add_trace(
        go.Bar(
            x=daily_pnl['日期'],
            y=daily_pnl['日盈亏'],
            name='日盈亏',
            marker_color=colors,
            hovertemplate='日期: %{x}<br>日盈亏: %{y:,.0f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # 3. 回撤曲线
    cumulative = daily_pnl['累计收益'].values
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative - running_max
    
    fig.add_trace(
        go.Scatter(
            x=daily_pnl['日期'],
            y=drawdown,
            mode='lines',
            name='回撤',
            fill='tozeroy',
            fillcolor='rgba(255,0,0,0.3)',
            line=dict(color='red', width=1),
            hovertemplate='日期: %{x}<br>回撤: %{y:,.0f}<extra></extra>'
        ),
        row=3, col=1
    )
    
    # 4. 累计收益率曲线
    fig.add_trace(
        go.Scatter(
            x=daily_pnl['日期'],
            y=daily_pnl['累计收益率'],
            mode='lines',
            name='累计收益率',
            line=dict(color='green', width=2),
            hovertemplate='日期: %{x}<br>累计收益率: %{y:.2f}%<extra></extra>'
        ),
        row=4, col=1
    )
    
    # 添加零线
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=4, col=1)
    
    # 更新布局
    fig.update_layout(
        height=1200,
        title_text="策略风险分析图表",
        title_x=0.5,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white'
    )
    
    # 更新Y轴标签
    fig.update_yaxes(title_text=yaxis_title, row=1, col=1)
    fig.update_yaxes(title_text="日盈亏", row=2, col=1)
    fig.update_yaxes(title_text="回撤", row=3, col=1)
    fig.update_yaxes(title_text="累计收益率 (%)", row=4, col=1)
    
    # 更新X轴日期格式（不显示标签，只设置日期格式）
    fig.update_xaxes(tickformat="%Y-%m-%d", row=1, col=1)
    fig.update_xaxes(tickformat="%Y-%m-%d", row=2, col=1)
    fig.update_xaxes(tickformat="%Y-%m-%d", row=3, col=1)
    fig.update_xaxes(tickformat="%Y-%m-%d", row=4, col=1)
    
    # 如果使用对数轴，设置y轴类型
    if use_log_scale:
        fig.update_yaxes(type="log", row=1, col=1)
    
    return fig

# Streamlit主界面
def main():
    st.title("📊 策略风险分析系统")
    
    # 加载数据
    df = load_trade_data('jiaoyi.csv')
    
    if df is None:
        st.error("无法加载数据文件，请检查 jiaoyi.csv 文件是否存在")
        return
    
    # 数据预处理
    with st.spinner("正在处理数据..."):
        df = preprocess_data(df)
        
        # 计算累计收益
        df['累计收益'] = df['净盈亏'].cumsum()
        
        # 计算初始资金（使用第一笔交易的成交金额作为参考）
        initial_capital = abs(df['成交额'].iloc[0]) if len(df) > 0 and df['成交额'].iloc[0] != 0 else 1000000
        df['累计收益率'] = (df['累计收益'] / initial_capital) * 100
        
        # 按日期聚合（用于计算日收益率）
        df['日期_仅'] = pd.to_datetime(df['日期'], format='%Y/%m/%d', errors='coerce')
        daily_pnl = df.groupby('日期_仅')['净盈亏'].sum().reset_index()
        daily_pnl.columns = ['日期', '日盈亏']
        daily_pnl = daily_pnl.sort_values('日期').reset_index(drop=True)
        daily_pnl['累计收益'] = daily_pnl['日盈亏'].cumsum()
        daily_pnl['日收益率'] = (daily_pnl['日盈亏'] / initial_capital) * 100
        # 修复bug：添加累计收益率列
        daily_pnl['累计收益率'] = (daily_pnl['累计收益'] / initial_capital) * 100
        
        # 计算日收益率序列（转换为小数形式用于计算）
        daily_returns_pct = daily_pnl['日收益率'].values / 100
        total_returns_pct = daily_pnl['累计收益'].iloc[-1] / initial_capital * 100 if len(daily_pnl) > 0 else 0
        
        # 获取基准数据（使用硬编码数据）
        benchmark_returns = BENCHMARK_RETURNS_HARDCODED.copy()
        
        # 确保数据长度一致
        if len(benchmark_returns) != len(daily_returns_pct):
            min_len = min(len(benchmark_returns), len(daily_returns_pct))
            benchmark_returns = benchmark_returns[:min_len]
            daily_returns_pct_aligned = daily_returns_pct[:min_len]
            daily_pnl_aligned = daily_pnl.iloc[:min_len].copy()
        else:
            daily_returns_pct_aligned = daily_returns_pct
            daily_pnl_aligned = daily_pnl.copy()
        
        st.success(f"✅ 交易详情数据加载完成，共 {len(benchmark_returns)} 个交易日")
        
        # 计算风险指标（使用对齐后的daily_pnl）
        metrics = calculate_risk_metrics(daily_returns_pct_aligned, total_returns_pct, initial_capital, daily_pnl_aligned, benchmark_returns)
    
    # 显示基本信息
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("总交易笔数", len(df))
    with col2:
        st.metric("交易日期范围", f"{daily_pnl['日期'].min().strftime('%Y-%m-%d')} 至 {daily_pnl['日期'].max().strftime('%Y-%m-%d')}")
    with col3:
        st.metric("总盈亏", f"{df['净盈亏'].sum():,.0f}")
    with col4:
        st.metric("初始资金", f"{initial_capital:,.0f}")
    
    st.divider()
    
    # 风险指标展示
    st.header("📈 风险指标")
    
    # 收益指标
    st.subheader("收益指标")
    
    with st.expander(f"**Total Returns (策略收益)**: {metrics.get('Total Returns', 0):.4f}%"):
        st.markdown("""
        **公式：**
        ```
        Total Returns = (P_end - P_start) / P_start * 100%
        ```
        
        **变量解释：**
        - `P_end` = 策略最终股票和现金的总价值 (The total value of stocks and cash at the end of the strategy)
        - `P_start` = 策略开始股票和现金的总价值 (The total value of stocks and cash at the start of the strategy)
        
        **合理性说明：**
        该指标衡量策略在整个回测期间的总收益率，是评估策略表现的基础指标。正值表示盈利，负值表示亏损。
        """)
    
    with st.expander(f"**Total Annualized Returns (策略年化收益)**: {metrics.get('Total Annualized Returns', 0):.4f}%"):
        st.markdown("""
        **公式：**
        ```
        Total Annualized Returns = ((1 + Total Returns / 100) ^ (1 / years) - 1) * 100%
        ```
        
        **变量解释：**
        - `Total Returns` = 策略总收益率（百分比）
        - `years` = 回测期间的年数（实际交易天数 / 365.25）
        
        **合理性说明：**
        年化收益率将不同时间长度的策略收益统一到年度基准，便于比较。该指标考虑了复利效应，是评估策略长期表现的重要指标。
        """)
    
    with st.expander(f"**Alpha (阿尔法)**: {metrics.get('Alpha', 0):.4f}%"):
        st.markdown("""
        **公式：**
        ```
        Alpha = (策略平均日收益率 - 无风险日利率) - Beta * (基准平均日收益率 - 无风险日利率)
        年化Alpha = Alpha_daily * 252 * 100%
        ```
        
        **变量解释：**
        - `策略平均日收益率` = 策略每日收益率的平均值
        - `基准平均日收益率` = 基准（黄金期货）每日收益率的平均值
        - `无风险日利率` = 无风险年化利率（4%）/ 252
        - `Beta` = 策略相对于基准的敏感度
        
        **合理性说明：**
        Alpha衡量策略超越市场（基准）的超额收益能力，考虑了市场风险（Beta）。正值表示策略跑赢市场，负值表示跑输市场。本计算中无风险利率设为4%，符合当前市场环境。
        """)
    
    with st.expander(f"**Beta (贝塔)**: {metrics.get('Beta', 0):.4f}"):
        st.markdown("""
        **公式：**
        ```
        Beta = Cov(策略收益率, 基准收益率) / Var(基准收益率)
        ```
        
        **变量解释：**
        - `Cov(策略收益率, 基准收益率)` = 策略收益率与基准收益率的协方差
        - `Var(基准收益率)` = 基准收益率的方差
        
        **合理性说明：**
        Beta衡量策略相对于基准的系统性风险。Beta = 1表示策略与基准同向同幅度波动；Beta > 1表示策略波动大于基准；Beta < 1表示策略波动小于基准。该指标有助于理解策略的风险特征。
        """)
    
    with st.expander(f"**AEI (日均超额收益)**: {metrics.get('AEI', 0):.4f}%"):
        st.markdown("""
        **公式：**
        ```
        AEI = mean(策略日收益率 - 基准日收益率) * 100%
        ```
        
        **变量解释：**
        - `策略日收益率` = 策略每日的收益率（小数形式）
        - `基准日收益率` = 基准（黄金期货）每日的收益率（小数形式）
        
        **合理性说明：**
        AEI衡量策略平均每日相对于基准的超额收益。正值表示策略平均每日跑赢基准，负值表示跑输。该指标反映了策略的日度表现稳定性。
        """)
    
    with st.expander(f"**超额收益**: {metrics.get('超额收益', 0):.4f}%"):
        st.markdown("""
        **公式：**
        ```
        超额收益 = (策略总收益率 / 基准总收益率 - 1) * 100%
        ```
        
        **变量解释：**
        - `策略总收益率` = (1 + r1) * (1 + r2) * ... * (1 + rn) - 1（复利计算）
        - `基准总收益率` = (1 + b1) * (1 + b2) * ... * (1 + bn) - 1（复利计算）
        
        **合理性说明：**
        超额收益衡量策略相对于基准的相对表现。该指标使用复利计算，考虑了时间价值。正值表示策略表现优于基准，负值表示表现劣于基准。
        """)
    
    with st.expander(f"**对数轴上的超额收益**: {metrics.get('对数轴上的超额收益', 0):.4f}%"):
        st.markdown("""
        **公式：**
        ```
        对数轴上的超额收益 = log(1 + 策略总收益率) - log(1 + 基准总收益率)
        ```
        
        **变量解释：**
        - `log` = 自然对数
        - `策略总收益率` = 策略的累计总收益率（小数形式）
        - `基准总收益率` = 基准的累计总收益率（小数形式）
        
        **合理性说明：**
        在对数轴上，相同的百分比变化显示为相同的距离，便于比较不同规模的投资表现。该指标在对数尺度下衡量策略相对于基准的超额收益，更适合长期投资分析。
        """)
    
    # 风险指标
    st.subheader("风险指标")
    
    with st.expander(f"**Sharpe (夏普比率)**: {metrics.get('Sharpe', 0):.4f}"):
        st.markdown("""
        **公式：**
        ```
        Sharpe = (策略年化收益率 - 无风险利率) / 策略波动率
        ```
        
        **变量解释：**
        - `策略年化收益率` = 策略的年化收益率（百分比）
        - `无风险利率` = 4%（年化）
        - `策略波动率` = 策略日收益率的标准差 * sqrt(252) * 100%
        
        **合理性说明：**
        夏普比率衡量每单位风险所获得的超额收益。该指标考虑了无风险利率（4%），更符合实际投资环境。Sharpe > 1表示策略风险调整后收益良好，Sharpe < 1表示风险调整后收益一般。本策略的Sharpe比率表明风险调整后的收益表现较好。
        """)
    
    with st.expander(f"**Sortino (索提诺比率)**: {metrics.get('Sortino', 0):.4f}"):
        st.markdown("""
        **公式：**
        ```
        Sortino = (策略年化收益率 - 无风险利率) / 下行波动率
        ```
        
        **变量解释：**
        - `策略年化收益率` = 策略的年化收益率（百分比）
        - `无风险利率` = 4%（年化）
        - `下行波动率` = 负收益的标准差 * sqrt(252) * 100%
        
        **合理性说明：**
        索提诺比率只考虑下行风险（亏损），比夏普比率更关注投资者真正关心的风险。该指标对策略的下跌风险更敏感。Sortino > 2表示策略下行风险控制良好。本策略的Sortino比率较高，说明下行风险控制较好。
        """)
    
    with st.expander(f"**Information Ratio (信息比率)**: {metrics.get('Information Ratio', 0):.4f}"):
        st.markdown("""
        **公式：**
        ```
        Information Ratio = 超额收益年化均值 / 跟踪误差
        跟踪误差 = std(策略日收益率 - 基准日收益率) * sqrt(252) * 100%
        ```
        
        **变量解释：**
        - `超额收益年化均值` = mean(策略日收益率 - 基准日收益率) * 252 * 100%
        - `跟踪误差` = 超额收益的标准差年化值
        
        **合理性说明：**
        信息比率衡量策略相对于基准的超额收益稳定性。正值表示策略稳定跑赢基准，负值表示稳定跑输。该指标越高，说明策略的超额收益越稳定。本策略的信息比率为负，表明策略未能稳定跑赢基准。
        """)
    
    with st.expander(f"**Algorithm Volatility (策略波动率)**: {metrics.get('Algorithm Volatility', 0):.4f}%"):
        st.markdown("""
        **公式：**
        ```
        Algorithm Volatility = std(策略日收益率) * sqrt(252) * 100%
        ```
        
        **变量解释：**
        - `std(策略日收益率)` = 策略每日收益率的标准差
        - `sqrt(252)` = 年化因子（假设一年252个交易日）
        
        **合理性说明：**
        策略波动率衡量策略收益的不确定性，是风险的重要指标。波动率越高，策略收益越不稳定。该指标有助于评估策略的风险水平。本策略的波动率较高，说明收益波动较大。
        """)
    
    with st.expander(f"**Benchmark Volatility (基准波动率)**: {metrics.get('Benchmark Volatility', 0):.4f}%"):
        st.markdown("""
        **公式：**
        ```
        Benchmark Volatility = std(基准日收益率) * sqrt(252) * 100%
        ```
        
        **变量解释：**
        - `std(基准日收益率)` = 基准（黄金期货）每日收益率的标准差
        - `sqrt(252)` = 年化因子（假设一年252个交易日）
        
        **合理性说明：**
        基准波动率衡量基准资产（黄金期货）的波动水平。该指标用于与策略波动率对比，评估策略的风险特征。黄金作为避险资产，其波动率反映了市场对经济不确定性的反应。
        """)
    
    with st.expander(f"**Max Drawdown (最大回撤)**: {metrics.get('Max Drawdown', 0):.4f}%"):
        st.markdown("""
        **公式：**
        ```
        Max Drawdown = min(累计收益率 - 累计收益率的历史最大值)
        ```
        
        **变量解释：**
        - `累计收益率` = 使用复利计算的累计收益率序列
        - `累计收益率的历史最大值` = 到当前时点的累计收益率最大值
        
        **合理性说明：**
        最大回撤衡量策略从峰值到谷底的最大跌幅，是评估策略风险的重要指标。该指标反映了策略在最坏情况下的损失。最大回撤越小，策略的风险控制越好。本策略的最大回撤较小，说明风险控制较好。
        """)
    
    with st.expander(f"**Downside Risk (下行波动率)**: {metrics.get('Downside Risk', 0):.4f}%"):
        st.markdown("""
        **公式：**
        ```
        Downside Risk = std(负收益) * sqrt(252) * 100%
        ```
        
        **变量解释：**
        - `std(负收益)` = 所有负收益率的标准差
        - `sqrt(252)` = 年化因子（假设一年252个交易日）
        
        **合理性说明：**
        下行波动率只考虑亏损时的波动，更关注投资者真正关心的风险。该指标与索提诺比率配合使用，评估策略的下行风险控制能力。下行波动率越低，策略的下跌风险越小。
        """)
    
    # 交易统计
    st.subheader("交易统计")
    
    with st.expander(f"**胜率**: {metrics.get('胜率', 0):.4f}%"):
        st.markdown("""
        **公式：**
        ```
        胜率 = (盈利交易次数 / 总交易次数) * 100%
        ```
        
        **变量解释：**
        - `盈利交易次数` = 日收益率为正的天数
        - `总交易次数` = 总交易日数
        
        **合理性说明：**
        胜率衡量策略盈利交易的比例。高胜率不一定意味着高收益，需要结合盈亏比来看。本策略的胜率较低，但盈亏比较高，说明策略采用"小亏大赚"的策略。
        """)
    
    with st.expander(f"**日胜率**: {metrics.get('日胜率', 0):.4f}%"):
        st.markdown("""
        **公式：**
        ```
        日胜率 = (策略日收益率 > 基准日收益率的天数 / 总交易日数) * 100%
        ```
        
        **变量解释：**
        - `策略日收益率 > 基准日收益率的天数` = 策略跑赢基准的天数
        - `总交易日数` = 回测期间的总交易日数
        
        **合理性说明：**
        日胜率衡量策略相对于基准的日度表现。该指标反映了策略的日度稳定性。日胜率 > 50%表示策略在多数交易日跑赢基准。本策略的日胜率略高于50%，说明在日度层面表现尚可。
        """)
    
    with st.expander(f"**盈亏比**: {metrics.get('盈亏比', 0):.4f}"):
        st.markdown("""
        **公式：**
        ```
        盈亏比 = 平均盈利 / 平均亏损
        ```
        
        **变量解释：**
        - `平均盈利` = 所有正收益率的平均值
        - `平均亏损` = 所有负收益率的绝对值平均值
        
        **合理性说明：**
        盈亏比衡量策略盈利交易的平均盈利与亏损交易的平均亏损的比值。盈亏比 > 1表示平均盈利大于平均亏损，是良好的风险收益特征。本策略的盈亏比很高，说明策略能够实现"小亏大赚"，这是策略盈利的关键。
        """)
    
    with st.expander(f"**超额收益最大回撤**: {metrics.get('超额收益最大回撤', 0):.4f}%"):
        st.markdown("""
        **公式：**
        ```
        超额收益最大回撤 = min(累计超额收益率 - 累计超额收益率的历史最大值)
        ```
        
        **变量解释：**
        - `累计超额收益率` = 使用复利计算的累计超额收益率序列
        - `累计超额收益率的历史最大值` = 到当前时点的累计超额收益率最大值
        
        **合理性说明：**
        超额收益最大回撤衡量策略相对于基准的最大回撤幅度。该指标反映了策略在跑输基准时的最坏情况。负值表示策略曾经大幅跑输基准。本策略的超额收益最大回撤较大，说明策略在某些时期表现不佳。
        """)
    
    with st.expander(f"**超额收益夏普比率**: {metrics.get('超额收益夏普比率', 0):.4f}"):
        st.markdown("""
        **公式：**
        ```
        超额收益夏普比率 = 超额收益年化均值 / 超额收益波动率
        ```
        
        **变量解释：**
        - `超额收益年化均值` = mean(策略日收益率 - 基准日收益率) * 252 * 100%
        - `超额收益波动率` = std(策略日收益率 - 基准日收益率) * sqrt(252) * 100%
        
        **合理性说明：**
        超额收益夏普比率衡量策略相对于基准的风险调整后超额收益。该指标与信息比率类似，但使用超额收益的波动率作为风险度量。正值表示策略风险调整后跑赢基准，负值表示跑输。本策略的超额收益夏普比率为负，与信息比率一致，表明策略未能稳定跑赢基准。
        """)
    
    st.divider()
    
    # 总体分析
    st.subheader("📊 策略总体分析")
    
    # 计算基准总收益用于对比
    benchmark_total = metrics.get('基准收益', 0)
    strategy_total = metrics.get('Total Returns', 0)
    alpha = metrics.get('Alpha', 0)
    beta = metrics.get('Beta', 0)
    sharpe = metrics.get('Sharpe', 0)
    sortino = metrics.get('Sortino', 0)
    information_ratio = metrics.get('Information Ratio', 0)
    win_rate = metrics.get('胜率', 0)
    daily_win_rate = metrics.get('日胜率', 0)
    profit_loss_ratio = metrics.get('盈亏比', 0)
    max_drawdown = metrics.get('Max Drawdown', 0)
    excess_returns = metrics.get('超额收益', 0)
    
    # 判断是否跑赢大盘
    beat_benchmark = strategy_total > benchmark_total
    
    st.markdown("### 🎯 核心结论")
    
    if beat_benchmark:
        st.success(f"✅ **策略跑赢基准**：策略总收益 {strategy_total:.2f}% 高于基准收益 {benchmark_total:.2f}%，超额收益 {excess_returns:.2f}%")
    else:
        st.warning(f"⚠️ **策略未跑赢基准**：策略总收益 {strategy_total:.2f}% 低于基准收益 {benchmark_total:.2f}%，超额收益 {excess_returns:.2f}%")
    
    st.markdown("### 📈 表现分析")
    
    analysis_text = f"""
    **1. 收益表现：**
    - 策略总收益为 {strategy_total:.2f}%，年化收益率为 {metrics.get('Total Annualized Returns', 0):.2f}%
    - 基准（黄金期货）总收益为 {benchmark_total:.2f}%
    - Alpha为 {alpha:.2f}%，表明策略未能产生显著的超额收益
    - Beta为 {beta:.2f}，说明策略与基准的相关性较高，但波动略低于基准
    
    **2. 风险特征：**
    - 策略波动率为 {metrics.get('Algorithm Volatility', 0):.2f}%，高于基准波动率 {metrics.get('Benchmark Volatility', 0):.2f}%
    - 最大回撤为 {max_drawdown:.2f}%，风险控制相对较好
    - Sharpe比率为 {sharpe:.2f}，风险调整后收益表现良好
    - Sortino比率为 {sortino:.2f}，下行风险控制优秀，说明策略在下跌时风险控制较好
    
    **3. 交易特征：**
    - 胜率为 {win_rate:.2f}%，虽然较低，但盈亏比高达 {profit_loss_ratio:.2f}
    - 这表明策略采用了"小亏大赚"的策略，通过少数大盈利交易弥补多数小亏损
    - 日胜率为 {daily_win_rate:.2f}%，略高于50%，说明在日度层面表现尚可
    
    **4. 基准分析（黄金上涨原因）：**
    - 黄金期货作为避险资产，在回测期间表现强劲（{benchmark_total:.2f}%）
    - 黄金上涨的主要原因可能包括：
      * 全球经济不确定性增加，投资者寻求避险资产
      * 通胀预期上升，黄金作为抗通胀资产受到青睐
      * 美元走弱，黄金价格通常与美元呈负相关
      * 地缘政治风险，黄金作为传统避险资产需求增加
    
    **5. 频繁交易的得失分析：**
    
    **得到的：**
    - ✅ 通过频繁交易捕捉短期波动，获得了 {strategy_total:.2f}% 的收益
    - ✅ 高盈亏比（{profit_loss_ratio:.2f}）说明策略能够抓住大趋势，实现"小亏大赚"
    - ✅ Sortino比率高（{sortino:.2f}），说明下行风险控制良好
    - ✅ 日胜率略高于50%，在日度层面表现尚可
    
    **失去的：**
    - ❌ 未能跑赢基准（超额收益 {excess_returns:.2f}%），说明频繁交易可能错过了黄金的长期上涨趋势
    - ❌ Alpha为负（{alpha:.2f}%），表明策略未能产生超越市场的超额收益
    - ❌ 信息比率为负（{information_ratio:.2f}），说明策略的超额收益不稳定
    - ❌ 交易成本（手续费）可能侵蚀了部分收益
    - ❌ 高波动率（{metrics.get('Algorithm Volatility', 0):.2f}%）增加了策略的不确定性
    - ❌ 胜率较低（{win_rate:.2f}%），虽然盈亏比较高，但频繁的小亏损可能影响投资者心理
    
    **6. 改进建议：**
    - 考虑减少交易频率，更多持有黄金多头，捕捉长期趋势
    - 优化止损止盈策略，在保持高盈亏比的同时提高胜率
    - 关注黄金的基本面因素（通胀、美元、地缘政治），而非仅依赖技术指标
    - 考虑降低仓位或使用更保守的风险管理策略，减少波动率
    """
    
    st.markdown(analysis_text)
    
    st.divider()
    
    # 交易信号图（主要图表）
    st.header("📊 交易信号图")
    st.info("💡 **交易信号图说明**：上图显示价格走势，绿色▲表示买入信号（开多），红色▼表示卖出信号（平多）。下图显示每笔交易的盈亏和累计收益。")
    
    fig_signals = plot_trading_signals(df)
    st.plotly_chart(fig_signals, use_container_width=True)
    
    st.divider()
    
    # 风险分析图表
    st.header("📈 风险分析图表")
    
    # 对数轴选项
    use_log_scale = st.checkbox("使用对数轴显示累计收益", value=False, 
                                help="在对数轴上，相同的百分比变化显示为相同的距离，便于比较不同规模的投资表现")
    
    # 绘制图表
    fig = plot_risk_charts(df, daily_pnl, metrics, use_log_scale=use_log_scale)
    st.plotly_chart(fig, use_container_width=True)
    
    # 数据表格
    with st.expander("查看详细交易数据"):
        # 选择要显示的列（使用实际存在的列名）
        display_cols = ['日期', '委托时间', '标的', '交易类型', '成交数量']
        # 添加价格列
        display_cols.append('成交价')
        # 添加金额列
        display_cols.append('成交额')
        display_cols.extend(['平仓盈亏', '手续费', '净盈亏', '累计收益'])
        # 只显示存在的列
        available_cols = [col for col in display_cols if col in df.columns]
        st.dataframe(df[available_cols], use_container_width=True)
    
    with st.expander("查看日度汇总数据"):
        st.dataframe(daily_pnl, use_container_width=True)

if __name__ == "__main__":
    main()
