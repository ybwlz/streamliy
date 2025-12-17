import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager
import os
import warnings
warnings.filterwarnings('ignore')

"""直接画图，不用streamlit"""

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ========== 基准数据 ==========
# 从risk.py中复制的基准数据
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

def load_trade_data(filename='jiaoyi.csv'):
    """加载交易数据"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, filename)
    df = pd.read_csv(csv_path, encoding='gbk')
    return df

def preprocess_data(df):
    """数据清洗和预处理"""
    df = df.copy()
    
    # 合并日期时间
    df['日期时间'] = pd.to_datetime(df['日期'] + ' ' + df['委托时间'], format='%Y/%m/%d %H:%M:%S', errors='coerce')
    df = df.sort_values('日期时间').reset_index(drop=True)
    
    # 清理数字列
    df['成交数量'] = pd.to_numeric(df['成交数量'].astype(str).str.replace('手', ''), errors='coerce')
    df['成交价'] = pd.to_numeric(df['成交价'].astype(str).str.replace(',', ''), errors='coerce')
    df['成交额'] = pd.to_numeric(df['成交额'].astype(str).str.replace(',', ''), errors='coerce')
    df['平仓盈亏'] = pd.to_numeric(df['平仓盈亏'].astype(str).str.replace(',', '').replace(['-', ''], '0'), errors='coerce').fillna(0)
    df['手续费'] = pd.to_numeric(df['手续费'].astype(str).str.replace(',', '').replace(['-', ''], '0'), errors='coerce').fillna(0)
    
    # 计算净盈亏
    df['净盈亏'] = df['平仓盈亏'] - df['手续费']
    
    return df

def calculate_cumulative_returns(daily_returns):
    """计算累积收益率（使用复利）"""
    # daily_returns是小数形式（0.01表示1%）
    # 累积收益率 = (1 + r1) * (1 + r2) * ... - 1
    cumulative = np.cumprod(1 + daily_returns) - 1
    return cumulative * 100  # 转换为百分比

def calculate_metrics(daily_returns, benchmark_returns, dates, initial_capital, daily_pnl):
    """计算关键绩效指标"""
    metrics = {}
    
    # 总收益率
    strategy_total = calculate_cumulative_returns(daily_returns)[-1]
    benchmark_total = calculate_cumulative_returns(benchmark_returns)[-1]
    metrics['策略收益'] = strategy_total
    metrics['基准收益'] = benchmark_total
    
    # 年化收益率
    # dates是numpy数组，timedelta64没有.days属性，需要转换为pandas Timedelta或使用除法
    date_diff = dates[-1] - dates[0]
    if isinstance(date_diff, np.timedelta64):
        date_range = date_diff / np.timedelta64(1, 'D')
    else:
        date_range = pd.Timedelta(date_diff).days
    years = date_range / 365.25 if date_range > 0 else len(daily_returns) / 252
    if years > 0:
        metrics['策略年化收益'] = ((1 + strategy_total / 100) ** (1 / years) - 1) * 100
    else:
        metrics['策略年化收益'] = 0
    
    # 波动率
    if len(daily_returns) > 1:
        metrics['算法波动率'] = np.std(daily_returns) * np.sqrt(252) * 100
    else:
        metrics['算法波动率'] = 0
    
    if len(benchmark_returns) > 1:
        metrics['基准波动率'] = np.std(benchmark_returns) * np.sqrt(252) * 100
    else:
        metrics['基准波动率'] = 0
    
    # Sharpe比率
    if metrics['算法波动率'] > 0:
        metrics['Sharpe'] = metrics['策略年化收益'] / metrics['算法波动率']
    else:
        metrics['Sharpe'] = 0
    
    # Sortino比率
    downside_returns = daily_returns[daily_returns < 0]
    if len(downside_returns) > 1:
        downside_std = np.std(downside_returns) * np.sqrt(252) * 100
        if downside_std > 0:
            metrics['Sortino'] = metrics['策略年化收益'] / downside_std
        else:
            metrics['Sortino'] = 0
    else:
        metrics['Sortino'] = 0
    
    # Alpha和Beta
    if len(daily_returns) > 1 and len(benchmark_returns) > 1:
        min_len = min(len(daily_returns), len(benchmark_returns))
        dr_clean = daily_returns[:min_len]
        br_clean = benchmark_returns[:min_len]
        
        valid_mask = ~(np.isnan(dr_clean) | np.isnan(br_clean) | 
                      np.isinf(dr_clean) | np.isinf(br_clean))
        dr_valid = dr_clean[valid_mask]
        br_valid = br_clean[valid_mask]
        
        if len(dr_valid) > 1:
            covariance = np.cov(dr_valid, br_valid)[0, 1]
            benchmark_variance = np.var(br_valid, ddof=0)
            
            if benchmark_variance > 1e-10:
                metrics['Beta'] = covariance / benchmark_variance
                strategy_mean = np.mean(dr_valid)
                benchmark_mean = np.mean(br_valid)
                alpha_daily = strategy_mean - metrics['Beta'] * benchmark_mean
                metrics['Alpha'] = alpha_daily * 252 * 100
            else:
                metrics['Beta'] = 0
                metrics['Alpha'] = 0
            
            # Information Ratio
            excess_returns = dr_valid - br_valid
            if len(excess_returns) > 1:
                tracking_error = np.std(excess_returns, ddof=0) * np.sqrt(252) * 100
                excess_mean = np.mean(excess_returns) * 252 * 100
                metrics['信息比率'] = excess_mean / tracking_error if tracking_error > 1e-10 else 0
            else:
                metrics['信息比率'] = 0
        else:
            metrics['Beta'] = 0
            metrics['Alpha'] = 0
            metrics['信息比率'] = 0
    else:
        metrics['Beta'] = 0
        metrics['Alpha'] = 0
        metrics['信息比率'] = 0
    
    # 最大回撤
    strategy_cumulative = calculate_cumulative_returns(daily_returns)
    running_max = np.maximum.accumulate(strategy_cumulative)
    drawdown = strategy_cumulative - running_max
    metrics['最大回撤'] = np.min(drawdown)
    
    # 找到最大回撤的日期范围
    max_dd_idx = np.argmin(drawdown)
    if max_dd_idx > 0:
        peak_idx = np.argmax(strategy_cumulative[:max_dd_idx+1])
        metrics['最大回撤开始日期'] = dates[peak_idx] if peak_idx < len(dates) else dates[0]
        metrics['最大回撤结束日期'] = dates[max_dd_idx] if max_dd_idx < len(dates) else dates[-1]
    
    # 胜率
    winning_trades = (daily_returns > 0).sum()
    total_trades = len(daily_returns)
    metrics['胜率'] = (winning_trades / total_trades) if total_trades > 0 else 0
    
    # 日胜率
    if len(daily_returns) > 0 and len(benchmark_returns) > 0:
        min_len = min(len(daily_returns), len(benchmark_returns))
        winning_days = (daily_returns[:min_len] > benchmark_returns[:min_len]).sum()
        metrics['日胜率'] = (winning_days / min_len) if min_len > 0 else 0
    else:
        winning_days = (daily_pnl['日盈亏'] > 0).sum()
        total_days = len(daily_pnl)
        metrics['日胜率'] = (winning_days / total_days) if total_days > 0 else 0
    
    # 盈亏比
    if winning_trades > 0 and (total_trades - winning_trades) > 0:
        avg_win = daily_returns[daily_returns > 0].mean()
        avg_loss = abs(daily_returns[daily_returns < 0].mean())
        metrics['盈亏比'] = avg_win / avg_loss if avg_loss > 0 else 0
    else:
        metrics['盈亏比'] = 0
    
    # 盈利次数和亏损次数
    metrics['盈利次数'] = winning_trades
    metrics['亏损次数'] = total_trades - winning_trades
    
    return metrics

def plot_strategy_comparison():
    """绘制策略收益、基准收益和超额收益对比图"""
    
    # 加载数据
    print("正在加载数据...")
    df = load_trade_data('jiaoyi.csv')
    
    # 数据预处理
    print("正在处理数据...")
    df = preprocess_data(df)
    
    # 计算累计收益
    df['累计收益'] = df['净盈亏'].cumsum()
    
    # 计算初始资金
    initial_capital = abs(df['成交额'].iloc[0]) if len(df) > 0 and df['成交额'].iloc[0] != 0 else 1000000
    
    # 按日期聚合
    df['日期_仅'] = pd.to_datetime(df['日期'], format='%Y/%m/%d', errors='coerce')
    daily_pnl = df.groupby('日期_仅')['净盈亏'].sum().reset_index()
    daily_pnl.columns = ['日期', '日盈亏']
    daily_pnl = daily_pnl.sort_values('日期').reset_index(drop=True)
    daily_pnl['累计收益'] = daily_pnl['日盈亏'].cumsum()
    daily_pnl['日收益率'] = (daily_pnl['日盈亏'] / initial_capital) * 100
    
    # 计算日收益率序列（转换为小数形式）
    daily_returns_pct = daily_pnl['日收益率'].values / 100
    
    # 获取基准数据
    benchmark_returns = BENCHMARK_RETURNS_HARDCODED.copy()
    
    # 确保数据长度一致
    min_len = min(len(benchmark_returns), len(daily_returns_pct))
    benchmark_returns = benchmark_returns[:min_len]
    daily_returns_pct = daily_returns_pct[:min_len]
    daily_pnl = daily_pnl.iloc[:min_len].copy()
    
    print(f"数据对齐完成，共 {min_len} 个交易日")
    
    # 计算关键指标
    print("正在计算关键指标...")
    metrics = calculate_metrics(daily_returns_pct, benchmark_returns, daily_pnl['日期'].values, 
                                initial_capital, daily_pnl)
    
    # 计算累积收益率
    strategy_cumulative = calculate_cumulative_returns(daily_returns_pct)
    benchmark_cumulative = calculate_cumulative_returns(benchmark_returns)
    excess_cumulative = strategy_cumulative - benchmark_cumulative
    
    # 获取日期序列
    dates = daily_pnl['日期'].values
    
    # 创建图表（更大的尺寸以匹配图片）
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # 绘制策略收益（蓝色线，带浅蓝色填充）
    ax.plot(dates, strategy_cumulative, label='策略收益', color='#1f77b4', linewidth=2.5, alpha=0.9, zorder=3)
    ax.fill_between(dates, 0, strategy_cumulative, alpha=0.15, color='#1f77b4', zorder=1)
    
    # 绘制基准收益（橙色/棕色线）
    ax.plot(dates, benchmark_cumulative, label='基准收益', color='#ff7f0e', linewidth=2.5, alpha=0.9, zorder=3)
    
    # 绘制超额收益（红色线）
    ax.plot(dates, excess_cumulative, label='超额收益', color='#d62728', linewidth=2, alpha=0.8, zorder=2)
    
    # 添加零线
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.4, zorder=0)
    
    # 标记最大回撤点和峰值点（如果存在）
    strategy_running_max = np.maximum.accumulate(strategy_cumulative)
    strategy_drawdown = strategy_cumulative - strategy_running_max
    max_drawdown_idx = np.argmin(strategy_drawdown)
    
    peak_idx = np.argmax(strategy_cumulative)
    
    # 标记峰值点（绿色圆点）
    if peak_idx > 0 and peak_idx < len(strategy_cumulative) - 1:
        ax.plot(dates[peak_idx], strategy_cumulative[peak_idx], 
                'o', color='green', markersize=8, zorder=4, label='关键点')
    
    # 标记最大回撤点（绿色圆点，如果与峰值点不同）
    if max_drawdown_idx > 0 and max_drawdown_idx != peak_idx:
        ax.plot(dates[max_drawdown_idx], strategy_cumulative[max_drawdown_idx], 
                'o', color='green', markersize=8, zorder=4)
    
    # 设置标题和标签
    ax.set_title('策略收益 vs 基准收益 vs 超额收益', fontsize=18, fontweight='bold', pad=25)
    ax.set_xlabel('日期', fontsize=13, fontweight='bold')
    ax.set_ylabel('累积收益率 (%)', fontsize=13, fontweight='bold')
    
    # 设置网格（更细致的网格）
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5, which='both')
    ax.set_axisbelow(True)
    
    # 设置图例（右上角，带边框）
    ax.legend(loc='upper left', fontsize=12, framealpha=0.95, edgecolor='black', fancybox=True, shadow=True)
    
    # 格式化X轴日期
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    # 根据数据长度自动调整日期刻度
    if len(dates) > 100:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
    elif len(dates) > 50:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_minor_locator(mdates.WeekdayLocator())
    else:
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=10)
    
    # 设置Y轴格式（百分比，保留2位小数）
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}%'))
    ax.tick_params(axis='y', labelsize=10)
    
    # 设置背景色为白色
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'strategy_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存至: {output_path}")
    
    # 显示图表
    plt.show()
    
    # 打印关键指标（类似图片中的格式）
    print("\n" + "="*80)
    print("关键绩效指标 (KPIs)")
    print("="*80)
    print(f"策略收益 (Strategy Returns): {metrics['策略收益']:.2f}%")
    print(f"策略年化收益 (Annualized Strategy Returns): {metrics['策略年化收益']:.2f}%")
    print(f"基准收益 (Benchmark Returns): {metrics['基准收益']:.2f}%")
    print(f"Alpha: {metrics['Alpha']:.3f}")
    print(f"Beta: {metrics['Beta']:.3f}")
    print(f"Sharpe (夏普比率): {metrics['Sharpe']:.3f}")
    print(f"Sortino (索提诺比率): {metrics['Sortino']:.3f}")
    print(f"Information Ratio (信息比率): {metrics['信息比率']:.3f}")
    print(f"Algorithm Volatility (算法波动率): {metrics['算法波动率']:.3f}")
    print(f"Benchmark Volatility (基准波动率): {metrics['基准波动率']:.3f}")
    print(f"胜率 (Win Rate): {metrics['胜率']:.3f}")
    print(f"日胜率 (Daily Win Rate): {metrics['日胜率']:.3f}")
    print(f"盈亏比 (Profit/Loss Ratio): {metrics['盈亏比']:.3f}")
    print(f"盈利次数 (Number of Winning Trades): {metrics['盈利次数']}")
    print(f"亏损次数 (Number of Losing Trades): {metrics['亏损次数']}")
    print(f"最大回撤 (Max Drawdown): {metrics['最大回撤']:.3f}%")
    if '最大回撤开始日期' in metrics and '最大回撤结束日期' in metrics:
        start_date = pd.to_datetime(metrics['最大回撤开始日期']).strftime('%Y-%m-%d')
        end_date = pd.to_datetime(metrics['最大回撤结束日期']).strftime('%Y-%m-%d')
        print(f"最大回撤发生时间区间: {start_date} 至 {end_date}")
    print(f"数据日期范围: {pd.to_datetime(dates[0]).strftime('%Y-%m-%d')} 至 {pd.to_datetime(dates[-1]).strftime('%Y-%m-%d')}")
    print("="*80)

if __name__ == "__main__":
    plot_strategy_comparison()
