import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# 设置页面配置
st.set_page_config(
    page_title="策略风险分析",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式 - 现代化深色科技风格
# 注意：CSS样式放在main函数中，避免在模块加载时执行
def apply_custom_styles():
    st.markdown("""
    <style>
        /* ========== 全局样式 ========== */
        * {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Microsoft YaHei', sans-serif;
        }
        
        /* 主背景 - 深色渐变 */
        .stApp {
            background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 50%, #0f1419 100%);
            color: #e4e6eb;
        }
        
        /* 主内容区域 */
        .main .block-container {
            background: rgba(15, 20, 25, 0.95);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(100, 120, 150, 0.15);
            border-radius: 16px;
            padding: 2rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        }
        
        /* ========== 标题样式 ========== */
        h1 {
            color: #ffffff;
            font-weight: 700;
            letter-spacing: -0.5px;
            font-size: 2.2rem;
            margin-bottom: 1rem;
        }
        
        h2, h3 {
            color: #e4e6eb;
            font-weight: 600;
            letter-spacing: -0.3px;
        }
        
        h4, h5, h6 {
            color: #b8bcc8;
            font-weight: 600;
        }
        
        /* ========== 文本样式 ========== */
        p, div, span, label {
            color: #b8bcc8;
            line-height: 1.6;
        }
        
        strong {
            color: #ffffff;
            font-weight: 600;
        }
        
        /* ========== Header样式 ========== */
        header[data-testid="stHeader"],
        .stAppHeader,
        [data-testid="stHeader"],
        div[data-testid="stHeader"],
        header.stAppHeader,
        header[class*="stAppHeader"],
        header[class*="st-emotion-cache"] {
            background: rgba(15, 20, 25, 0.95) !important;
            background-color: rgba(15, 20, 25, 0.95) !important;
            border-bottom: 1px solid rgba(100, 120, 150, 0.2) !important;
            backdrop-filter: blur(10px);
        }
        
        /* 覆盖所有可能的header背景色 */
        header {
            background: rgba(15, 20, 25, 0.95) !important;
            background-color: rgba(15, 20, 25, 0.95) !important;
        }
        
        header[data-testid="stHeader"] *,
        .stAppHeader *,
        [data-testid="stHeader"] *,
        header * {
            color: #e4e6eb !important;
        }
        
        /* 确保header内的所有元素都是深色 */
        header[data-testid="stHeader"] div,
        header[data-testid="stHeader"] span,
        header[data-testid="stHeader"] p,
        header div,
        header span,
        header p {
            color: #e4e6eb !important;
        }
        
        /* ========== 侧边栏样式 ========== */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f1419 0%, #1a1f2e 100%);
            border-right: 1px solid rgba(100, 120, 150, 0.2);
        }
        
        [data-testid="stSidebar"] * {
            color: #b8bcc8 !important;
        }
        
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #ffffff !important;
        }
        
        /* ========== Radio Button样式 ========== */
        .stRadio > div > label {
            color: #e4e6eb !important;
            font-weight: 500;
            padding: 0.5rem;
            border-radius: 8px;
            transition: all 0.2s;
        }
        
        .stRadio > div > label:hover {
            background: rgba(100, 120, 150, 0.1);
        }
        
        .stRadio > div > label[data-baseweb="radio"] {
            background: rgba(100, 120, 150, 0.15) !important;
            border: 1px solid rgba(100, 120, 150, 0.3) !important;
        }
        
        /* ========== Checkbox样式 ========== */
        .stCheckbox > label {
            color: #e4e6eb !important;
            font-weight: 500;
        }
        
        .stCheckbox > label > div[data-baseweb="checkbox"] {
            background: rgba(100, 120, 150, 0.1) !important;
            border: 1.5px solid rgba(100, 120, 150, 0.3) !important;
        }
        
        /* ========== 指标卡片样式 ========== */
        [data-testid="stMetricValue"] {
            color: #4a9eff !important;
            font-weight: 700;
            font-size: 1.2rem;
            word-wrap: break-word;
            overflow-wrap: break-word;
            white-space: normal;
        }
        
        [data-testid="stMetricLabel"] {
            color: #b8bcc8 !important;
            font-weight: 500;
            font-size: 0.85rem;
        }
        
        [data-testid="stMetricDelta"] {
            color: #4ade80 !important;
        }
        
        /* 调整metric容器，确保内容完全显示 */
        [data-testid="stMetric"] {
            padding: 0.5rem;
            min-height: auto;
        }
        
        /* 确保metric容器内的文本可以换行 */
        [data-testid="stMetric"] > div {
            word-wrap: break-word;
            overflow-wrap: break-word;
            white-space: normal;
        }
        
        /* ========== 按钮样式 ========== */
        .stButton > button {
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
            color: #ffffff !important;
            border: none;
            border-radius: 8px;
            padding: 0.6rem 1.5rem;
            font-weight: 600;
            transition: all 0.2s;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
        }
        
        .stButton > button:hover {
            background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
            box-shadow: 0 6px 16px rgba(37, 99, 235, 0.4);
            transform: translateY(-1px);
        }
        
        /* ========== 输入框样式 ========== */
        .stNumberInput > div > div > input {
            background: rgba(20, 25, 35, 0.8) !important;
            border: 1.5px solid rgba(100, 120, 150, 0.3) !important;
            border-radius: 8px;
            color: #e4e6eb !important;
            font-weight: 500;
        }
        
        .stNumberInput > div > div > input:focus {
            border-color: #4a9eff !important;
            box-shadow: 0 0 0 3px rgba(74, 158, 255, 0.1) !important;
        }
        
        /* ========== 信息框样式 ========== */
        .stInfo {
            background: rgba(37, 99, 235, 0.1) !important;
            border-left: 4px solid #4a9eff !important;
            border-radius: 8px;
            color: #b8d4ff !important;
            padding: 1rem;
        }
        
        .stWarning {
            background: rgba(245, 158, 11, 0.1) !important;
            border-left: 4px solid #f59e0b !important;
            border-radius: 8px;
            color: #fde68a !important;
            padding: 1rem;
        }
        
        .stSuccess {
            background: rgba(34, 197, 94, 0.1) !important;
            border-left: 4px solid #22c55e !important;
            border-radius: 8px;
            color: #bbf7d0 !important;
            padding: 1rem;
        }
        
        .stError {
            background: rgba(239, 68, 68, 0.1) !important;
            border-left: 4px solid #ef4444 !important;
            border-radius: 8px;
            color: #fecaca !important;
            padding: 1rem;
        }
        
        /* ========== 分隔线样式 ========== */
        hr {
            border: none;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(100, 120, 150, 0.3), transparent);
            margin: 2rem 0;
        }
        
        /* ========== 代码块样式 ========== */
        code {
            background: rgba(20, 25, 35, 0.8) !important;
            color: #4ade80 !important;
            border: 1px solid rgba(74, 222, 128, 0.2);
            border-radius: 4px;
            padding: 0.2rem 0.4rem;
        }
        
        /* ========== 表格样式 ========== */
        .dataframe {
            background: rgba(20, 25, 35, 0.8) !important;
            color: #e4e6eb !important;
            border: 1px solid rgba(100, 120, 150, 0.2);
            border-radius: 8px;
        }
        
        .dataframe thead {
            background: rgba(100, 120, 150, 0.15) !important;
            color: #ffffff !important;
        }
        
        .dataframe tbody tr:hover {
            background: rgba(100, 120, 150, 0.1) !important;
        }
        
        /* ========== Expander样式 ========== */
        .streamlit-expanderHeader {
            background: rgba(100, 120, 150, 0.08) !important;
            border: 1px solid rgba(100, 120, 150, 0.2) !important;
            color: #e4e6eb !important;
            border-radius: 8px;
            font-weight: 600;
        }
        
        .streamlit-expanderHeader:hover {
            background: rgba(100, 120, 150, 0.15) !important;
        }
        
        /* ========== 滚动条样式 ========== */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(0, 0, 0, 0.3);
        }
        
        ::-webkit-scrollbar-thumb {
            background: rgba(100, 120, 150, 0.5);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(100, 120, 150, 0.7);
        }
        
        /* ========== Plotly图表容器 ========== */
        .js-plotly-plot {
            background: rgba(20, 25, 35, 0.6) !important;
            border-radius: 12px;
            padding: 15px;
            border: 1px solid rgba(100, 120, 150, 0.15);
        }
        
        /* ========== Spinner样式 ========== */
        .stSpinner > div {
            border-color: #4a9eff transparent transparent transparent !important;
        }
        
        /* ========== 链接样式 ========== */
        a {
            color: #4a9eff !important;
        }
        
        a:hover {
            color: #60a5fa !important;
        }
    </style>
    """, unsafe_allow_html=True)

# 写死的基准数据（从data.py获取后固定）
# 这是根据jiaoyi.csv的日期范围（2024/1/4到2025/4/28）获取的黄金期货基准收益率数据
# 数据格式：与交易日期对齐的日收益率序列（小数形式）
BENCHMARK_DATA_HARDCODED = None  # 将在运行时从data.py获取并缓存

@st.cache_data
def get_hardcoded_benchmark_data(trade_dates):
    """
    获取写死的基准数据（如果data.py可用则获取，否则返回None）
    实际使用中，可以将获取到的数据直接写死在代码中
    """
    try:
        from data import get_benchmark_daily_returns_aligned
        benchmark_returns = get_benchmark_daily_returns_aligned(trade_dates)
        return benchmark_returns
    except:
        return None

# 加载交易数据（使用GBK编码）
@st.cache_data
def load_trade_data(filename='jiaoyi.csv'):
    """加载交易数据"""
    try:
        df = pd.read_csv(filename, encoding='gbk')
        return df
    except Exception as e:
        st.error(f"读取文件失败: {e}")
        return None

# 数据清洗和预处理
def preprocess_data(df):
    """预处理交易数据"""
    df = df.copy()
    
    # 合并日期和时间
    try:
        df['日期时间'] = pd.to_datetime(df['日期'].astype(str) + ' ' + df['委托时间'].astype(str), 
                                      format='%Y/%m/%d %H:%M:%S', errors='coerce')
    except:
        df['日期时间'] = pd.to_datetime(df['日期'].astype(str) + ' ' + df['委托时间'].astype(str), errors='coerce')
    
    df = df.sort_values('日期时间').reset_index(drop=True)
    
    # 转换数据类型
    # 成交数量（处理"手"单位）
    if '成交数量' in df.columns:
        if df['成交数量'].dtype == 'object':
            df['成交数量'] = df['成交数量'].astype(str).str.replace('手', '').str.replace(',', '').str.strip()
            df['成交数量'] = pd.to_numeric(df['成交数量'], errors='coerce')
        else:
            df['成交数量'] = pd.to_numeric(df['成交数量'], errors='coerce')
    
    # 成交价格（列名可能是'成交价'）
    price_col = '成交价' if '成交价' in df.columns else '成交价格'
    if price_col in df.columns:
        if df[price_col].dtype == 'object':
            df[price_col] = df[price_col].astype(str).str.replace(',', '').str.strip()
        df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
        # 统一列名为'成交价格'
        if price_col != '成交价格':
            df['成交价格'] = df[price_col]
    
    # 成交金额（列名可能是'成交额'）
    amount_col = '成交额' if '成交额' in df.columns else '成交金额'
    if amount_col in df.columns:
        if df[amount_col].dtype == 'object':
            df[amount_col] = df[amount_col].astype(str).str.replace(',', '').str.strip()
        df[amount_col] = pd.to_numeric(df[amount_col], errors='coerce')
        # 统一列名为'成交金额'
        if amount_col != '成交金额':
            df['成交金额'] = df[amount_col]
    
    # 平仓盈亏（处理"-"表示0的情况）
    if df['平仓盈亏'].dtype == 'object':
        df['平仓盈亏'] = df['平仓盈亏'].astype(str).str.replace(',', '').str.strip()
        df['平仓盈亏'] = df['平仓盈亏'].replace(['-', ''], '0')
    df['平仓盈亏'] = pd.to_numeric(df['平仓盈亏'], errors='coerce').fillna(0)
    
    # 手续费
    if df['手续费'].dtype == 'object':
        df['手续费'] = df['手续费'].astype(str).str.replace(',', '').str.strip()
        df['手续费'] = df['手续费'].replace(['-', ''], '0')
    df['手续费'] = pd.to_numeric(df['手续费'], errors='coerce').fillna(0)
    
    # 计算每笔交易的净盈亏（平仓盈亏 - 手续费）
    df['净盈亏'] = df['平仓盈亏'] - df['手续费']
    
    return df

# 计算风险指标
def calculate_risk_metrics(daily_returns, total_returns, initial_capital, daily_pnl, benchmark_returns=None):
    """计算各种风险指标"""
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
    
    # Sharpe 夏普比率（假设无风险利率为0）
    if metrics['Algorithm Volatility'] > 0:
        metrics['Sharpe'] = metrics['Total Annualized Returns'] / metrics['Algorithm Volatility']
    else:
        metrics['Sharpe'] = 0
    
    # Sortino 索提诺比率（只考虑下行波动）
    downside_returns = daily_returns[daily_returns < 0]
    if len(downside_returns) > 1:
        downside_std = np.std(downside_returns) * np.sqrt(252) * 100
        metrics['Downside Risk'] = downside_std
        if downside_std > 0:
            metrics['Sortino'] = metrics['Total Annualized Returns'] / downside_std
        else:
            metrics['Sortino'] = 0
    else:
        metrics['Downside Risk'] = 0
        metrics['Sortino'] = 0
    
    # Max Drawdown 最大回撤
    cumulative = np.cumsum(daily_returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative - running_max
    metrics['Max Drawdown'] = np.min(drawdown) if len(drawdown) > 0 else 0
    
    # Alpha 和 Beta（如果有基准数据）
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
            # 确保两个数组长度一致
            final_len = min(len(daily_returns_clean), len(benchmark_returns_clean))
            daily_returns_final = daily_returns_clean[:final_len]
            benchmark_returns_final = benchmark_returns_clean[:final_len]
            
            # 计算协方差和方差
            covariance = np.cov(daily_returns_final, benchmark_returns_final)[0, 1]
            benchmark_variance = np.var(benchmark_returns_final, ddof=0)
            
            if benchmark_variance > 1e-10:  # 避免除零
                metrics['Beta'] = covariance / benchmark_variance
                
                # Alpha = 策略平均日收益率 - Beta * 基准平均日收益率，然后年化
                # 注意：daily_returns和benchmark_returns都应该是小数形式（0.01表示1%）
                strategy_mean_daily = np.mean(daily_returns_final)
                benchmark_mean_daily = np.mean(benchmark_returns_final)
                alpha_daily = strategy_mean_daily - metrics['Beta'] * benchmark_mean_daily
                # 年化Alpha（转换为百分比）
                metrics['Alpha'] = alpha_daily * 252 * 100
            else:
                metrics['Beta'] = 0
                metrics['Alpha'] = 0
            
            # Information Ratio 信息比率
            excess_returns = daily_returns_final - benchmark_returns_final
            if len(excess_returns) > 1:
                tracking_error = np.std(excess_returns, ddof=0) * np.sqrt(252)  # 保持小数形式
                excess_return_mean = np.mean(excess_returns)
                excess_return_annual = excess_return_mean * 252  # 年化超额收益（小数形式）
                
                if tracking_error > 1e-10:
                    # Information Ratio = 年化超额收益 / 年化跟踪误差（都是小数形式）
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
    
    # 日胜率（按日统计）
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
        min_len = min(len(daily_returns), len(benchmark_returns))
        daily_returns_aligned = daily_returns[:min_len]
        benchmark_returns_aligned = benchmark_returns[:min_len]
        valid_mask = ~(np.isnan(daily_returns_aligned) | np.isnan(benchmark_returns_aligned) | 
                      np.isinf(daily_returns_aligned) | np.isinf(benchmark_returns_aligned))
        daily_clean = daily_returns_aligned[valid_mask]
        benchmark_clean = benchmark_returns_aligned[valid_mask]
        if len(daily_clean) > 0 and len(benchmark_clean) > 0:
            final_len = min(len(daily_clean), len(benchmark_clean))
            excess_returns = daily_clean[:final_len] - benchmark_clean[:final_len]
            metrics['AEI'] = np.mean(excess_returns) * 100
        else:
            metrics['AEI'] = 0
    else:
        metrics['AEI'] = 0
    
    # 超额收益最大回撤
    if benchmark_returns is not None and len(benchmark_returns) > 0:
        min_len = min(len(daily_returns), len(benchmark_returns))
        daily_returns_aligned = daily_returns[:min_len]
        benchmark_returns_aligned = benchmark_returns[:min_len]
        valid_mask = ~(np.isnan(daily_returns_aligned) | np.isnan(benchmark_returns_aligned) | 
                      np.isinf(daily_returns_aligned) | np.isinf(benchmark_returns_aligned))
        daily_clean = daily_returns_aligned[valid_mask]
        benchmark_clean = benchmark_returns_aligned[valid_mask]
        if len(daily_clean) > 0 and len(benchmark_clean) > 0:
            final_len = min(len(daily_clean), len(benchmark_clean))
            excess_returns = daily_clean[:final_len] - benchmark_clean[:final_len]
            excess_cumulative = np.cumsum(excess_returns)
            excess_running_max = np.maximum.accumulate(excess_cumulative)
            excess_drawdown = excess_cumulative - excess_running_max
            metrics['超额收益最大回撤'] = np.min(excess_drawdown) if len(excess_drawdown) > 0 else 0
        else:
            metrics['超额收益最大回撤'] = 0
    else:
        metrics['超额收益最大回撤'] = 0
    
    # 超额收益夏普比率
    if benchmark_returns is not None and len(benchmark_returns) > 0:
        min_len = min(len(daily_returns), len(benchmark_returns))
        daily_returns_aligned = daily_returns[:min_len]
        benchmark_returns_aligned = benchmark_returns[:min_len]
        valid_mask = ~(np.isnan(daily_returns_aligned) | np.isnan(benchmark_returns_aligned) | 
                      np.isinf(daily_returns_aligned) | np.isinf(benchmark_returns_aligned))
        daily_clean = daily_returns_aligned[valid_mask]
        benchmark_clean = benchmark_returns_aligned[valid_mask]
        if len(daily_clean) > 1 and len(benchmark_clean) > 1:
            final_len = min(len(daily_clean), len(benchmark_clean))
            excess_returns = daily_clean[:final_len] - benchmark_clean[:final_len]
            if len(excess_returns) > 1:
                excess_vol = np.std(excess_returns, ddof=0) * np.sqrt(252)  # 小数形式
                excess_mean = np.mean(excess_returns) * 252  # 年化超额收益（小数形式）
                metrics['超额收益夏普比率'] = excess_mean / excess_vol if excess_vol > 1e-10 else 0
            else:
                metrics['超额收益夏普比率'] = 0
        else:
            metrics['超额收益夏普比率'] = 0
    else:
        metrics['超额收益夏普比率'] = 0
    
    # 超额收益（除法版）- 策略收益 / 基准收益 - 1
    if benchmark_returns is not None and len(benchmark_returns) > 0:
        min_len = min(len(daily_returns), len(benchmark_returns))
        daily_returns_aligned = daily_returns[:min_len]
        benchmark_returns_aligned = benchmark_returns[:min_len]
        valid_mask = ~(np.isnan(daily_returns_aligned) | np.isnan(benchmark_returns_aligned) | 
                      np.isinf(daily_returns_aligned) | np.isinf(benchmark_returns_aligned))
        daily_clean = daily_returns_aligned[valid_mask]
        benchmark_clean = benchmark_returns_aligned[valid_mask]
        if len(daily_clean) > 0 and len(benchmark_clean) > 0:
            final_len = min(len(daily_clean), len(benchmark_clean))
            strategy_total = np.sum(daily_clean[:final_len])
            benchmark_total = np.sum(benchmark_clean[:final_len])
            if abs(benchmark_total) > 1e-10:
                metrics['超额收益'] = (strategy_total / benchmark_total - 1) * 100
            else:
                metrics['超额收益'] = 0
        else:
            metrics['超额收益'] = 0
    else:
        metrics['超额收益'] = 0
    
    # 对数轴上的超额收益
    # 计算方法：log(1 + 策略累计收益) - log(1 + 基准累计收益)
    if benchmark_returns is not None and len(benchmark_returns) > 0:
        min_len = min(len(daily_returns), len(benchmark_returns))
        daily_returns_aligned = daily_returns[:min_len]
        benchmark_returns_aligned = benchmark_returns[:min_len]
        valid_mask = ~(np.isnan(daily_returns_aligned) | np.isnan(benchmark_returns_aligned) | 
                      np.isinf(daily_returns_aligned) | np.isinf(benchmark_returns_aligned))
        daily_clean = daily_returns_aligned[valid_mask]
        benchmark_clean = benchmark_returns_aligned[valid_mask]
        if len(daily_clean) > 0 and len(benchmark_clean) > 0:
            final_len = min(len(daily_clean), len(benchmark_clean))
            strategy_cumulative = np.cumsum(daily_clean[:final_len])
            benchmark_cumulative = np.cumsum(benchmark_clean[:final_len])
            # 使用log1p避免log(0)的问题
            log_excess = np.log1p(strategy_cumulative[-1]) - np.log1p(benchmark_cumulative[-1])
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
            y=df_sorted['成交价格'],
            mode='lines',
            name='成交价格',
            line=dict(color='#4a9eff', width=2),
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
                y=buy_signals['成交价格'],
                mode='markers',
                name='买入信号（开多）',
                marker=dict(
                    symbol='triangle-up',
                    size=12,
                    color='#22c55e',
                    line=dict(width=1.5, color='#16a34a')
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
                y=sell_signals['成交价格'],
                mode='markers',
                name='卖出信号（平多）',
                marker=dict(
                    symbol='triangle-down',
                    size=12,
                    color='#ef4444',
                    line=dict(width=1.5, color='#dc2626')
                ),
                hovertemplate='卖出时间: %{x}<br>价格: %{y:.2f}<br>盈亏: %{customdata:.0f}<extra></extra>',
                customdata=sell_signals['平仓盈亏']
            ),
            row=1, col=1
        )
    
    # 2. 累计收益曲线（副图）
    df_sorted['累计收益'] = df_sorted['净盈亏'].cumsum()
    colors_profit = ['#22c55e' if x >= 0 else '#ef4444' for x in df_sorted['净盈亏']]
    
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
            line=dict(color='#4a9eff', width=2.5),
            hovertemplate='时间: %{x}<br>累计收益: %{y:,.0f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # 添加零线
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(184, 188, 200, 0.5)", row=2, col=1)
    
    # 更新布局 - 现代化深色风格
    fig.update_layout(
        height=800,
        title_text="交易信号图",
        title_x=0.5,
        title_font=dict(size=22, color='#ffffff'),
        showlegend=True,
        hovermode='x unified',
        template='plotly_dark',
        plot_bgcolor='rgba(15, 20, 25, 0.95)',
        paper_bgcolor='rgba(15, 20, 25, 0.95)',
        font=dict(color='#b8bcc8', size=12),
        legend=dict(
            bgcolor='rgba(20, 25, 35, 0.8)',
            bordercolor='rgba(100, 120, 150, 0.3)',
            borderwidth=1
        )
    )
    
    # 更新Y轴标签
    fig.update_yaxes(
        title_text="价格", 
        row=1, col=1, 
        title_font=dict(color='#e4e6eb', size=14),
        gridcolor='rgba(100, 120, 150, 0.15)',
        linecolor='rgba(100, 120, 150, 0.3)'
    )
    fig.update_yaxes(
        title_text="盈亏", 
        row=2, col=1, 
        title_font=dict(color='#e4e6eb', size=14),
        gridcolor='rgba(100, 120, 150, 0.15)',
        linecolor='rgba(100, 120, 150, 0.3)'
    )
    
    # 更新X轴
    fig.update_xaxes(
        title_text="时间", 
        row=2, col=1, 
        title_font=dict(color='#e4e6eb', size=14),
        gridcolor='rgba(100, 120, 150, 0.15)',
        linecolor='rgba(100, 120, 150, 0.3)'
    )
    
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
            line=dict(color='#4a9eff', width=2.5),
            hovertemplate='日期: %{x}<br>累计收益: %{y:,.0f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 添加零线
    if not use_log_scale:
        fig.add_hline(y=0, line_dash="dash", line_color="rgba(184, 188, 200, 0.5)", row=1, col=1)
    
    # 2. 日盈亏分布
    colors = ['#22c55e' if x > 0 else '#ef4444' for x in daily_pnl['日盈亏']]
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
            fillcolor='rgba(239, 68, 68, 0.2)',
            line=dict(color='#ef4444', width=2),
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
            line=dict(color='#22c55e', width=2.5),
            hovertemplate='日期: %{x}<br>累计收益率: %{y:.2f}%<extra></extra>'
        ),
        row=4, col=1
    )
    
    # 添加零线
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(184, 188, 200, 0.5)", row=4, col=1)
    
    # 更新布局 - 现代化深色风格
    fig.update_layout(
        height=1200,
        title_text="策略风险分析图表",
        title_x=0.5,
        title_font=dict(size=22, color='#ffffff'),
        showlegend=True,
        hovermode='x unified',
        template='plotly_dark',
        plot_bgcolor='rgba(15, 20, 25, 0.95)',
        paper_bgcolor='rgba(15, 20, 25, 0.95)',
        font=dict(color='#b8bcc8', size=12),
        legend=dict(
            bgcolor='rgba(20, 25, 35, 0.8)',
            bordercolor='rgba(100, 120, 150, 0.3)',
            borderwidth=1
        )
    )
    
    # 更新Y轴标签
    fig.update_yaxes(
        title_text=yaxis_title, 
        row=1, col=1, 
        title_font=dict(color='#e4e6eb', size=14),
        gridcolor='rgba(100, 120, 150, 0.15)',
        linecolor='rgba(100, 120, 150, 0.3)'
    )
    fig.update_yaxes(
        title_text="日盈亏", 
        row=2, col=1, 
        title_font=dict(color='#e4e6eb', size=14),
        gridcolor='rgba(100, 120, 150, 0.15)',
        linecolor='rgba(100, 120, 150, 0.3)'
    )
    fig.update_yaxes(
        title_text="回撤", 
        row=3, col=1, 
        title_font=dict(color='#e4e6eb', size=14),
        gridcolor='rgba(100, 120, 150, 0.15)',
        linecolor='rgba(100, 120, 150, 0.3)'
    )
    fig.update_yaxes(
        title_text="累计收益率 (%)", 
        row=4, col=1, 
        title_font=dict(color='#e4e6eb', size=14),
        gridcolor='rgba(100, 120, 150, 0.15)',
        linecolor='rgba(100, 120, 150, 0.3)'
    )
    
    # 更新X轴
    fig.update_xaxes(
        title_text="日期", 
        row=4, col=1, 
        title_font=dict(color='#e4e6eb', size=14),
        gridcolor='rgba(100, 120, 150, 0.15)',
        linecolor='rgba(100, 120, 150, 0.3)'
    )
    
    # 如果使用对数轴，设置y轴类型
    if use_log_scale:
        fig.update_yaxes(type="log", row=1, col=1)
    
    return fig

# 风险指标说明函数
def show_risk_metrics_explanation(metrics):
    """显示风险指标的详细说明"""
    st.markdown("### 📚 风险指标详解")
    
    # Total Returns
    total_returns = metrics.get('Total Returns', 0)
    st.markdown("#### 💰 Total Returns (策略收益)")
    st.markdown(f"**当前值**: {total_returns:.4f}%")
    st.markdown("""
    **概念**: 策略在整个回测期间的总收益率。
    
    **数值范围**: 无上限，通常为 -100% 到 +∞
    
    **信号解读**:
    - **> 0%**: 策略盈利，数值越大越好
    - **< 0%**: 策略亏损，需要优化策略
    - **> 50%**: 表现优秀
    - **> 100%**: 表现卓越
    """)
    st.divider()
    
    # Total Annualized Returns
    annual_returns = metrics.get('Total Annualized Returns', 0)
    st.markdown("#### 📈 Total Annualized Returns (策略年化收益)")
    st.markdown(f"**当前值**: {annual_returns:.4f}%")
    st.markdown("""
    **概念**: 将总收益率年化后的指标，便于比较不同时间跨度的策略。
    
    **数值范围**: 无上限，通常为 -100% 到 +∞
    
    **信号解读**:
    - **> 10%**: 表现良好
    - **> 20%**: 表现优秀
    - **> 30%**: 表现卓越
    - **< 5%**: 表现一般，可能需要优化
    """)
    st.divider()
    
    # Sharpe Ratio
    sharpe = metrics.get('Sharpe', 0)
    st.markdown("#### ⚡ Sharpe (夏普比率)")
    st.markdown(f"**当前值**: {sharpe:.4f}")
    st.markdown("""
    **概念**: 衡量每承担一单位风险所获得的超额收益，是风险调整后收益的重要指标。
    
    **数值范围**: 通常为 -∞ 到 +∞，实际中常见为 -2 到 5
    
    **信号解读**:
    - **> 2**: 表现优秀，风险调整后收益很高
    - **1-2**: 表现良好，风险收益比合理
    - **0-1**: 表现一般，风险收益比偏低
    - **< 0**: 表现较差，风险调整后收益为负
    """)
    st.divider()
    
    # Sortino Ratio
    sortino = metrics.get('Sortino', 0)
    st.markdown("#### 📉 Sortino (索提诺比率)")
    st.markdown(f"**当前值**: {sortino:.4f}")
    st.markdown("""
    **概念**: 类似夏普比率，但只考虑下行波动（亏损波动），更关注下行风险。
    
    **数值范围**: 通常为 -∞ 到 +∞，实际中常见为 -2 到 5
    
    **信号解读**:
    - **> 2**: 下行风险控制优秀
    - **1-2**: 下行风险控制良好
    - **0-1**: 下行风险控制一般
    - **< 0**: 下行风险较大
    """)
    st.divider()
    
    # Max Drawdown
    max_dd = metrics.get('Max Drawdown', 0)
    st.markdown("#### ⬇️ Max Drawdown (最大回撤)")
    st.markdown(f"**当前值**: {max_dd:.4f}%")
    st.markdown("""
    **概念**: 从历史最高点到最低点的最大跌幅，反映策略的最大亏损幅度。
    
    **数值范围**: 通常为 -100% 到 0%
    
    **信号解读**:
    - **> -10%**: 回撤控制优秀
    - **-10% 到 -20%**: 回撤控制良好
    - **-20% 到 -30%**: 回撤控制一般
    - **< -30%**: 回撤较大，风险较高
    """)
    st.divider()
    
    # Alpha
    alpha = metrics.get('Alpha', 0)
    st.markdown("#### 🎯 Alpha (阿尔法)")
    st.markdown(f"**当前值**: {alpha:.4f}%")
    
    # 根据Alpha值给出更准确的解读
    if alpha > 20:
        alpha_status = "⚠️ 异常高"
        alpha_interpretation = "Alpha值异常高，可能表示：1) 策略表现极佳；2) 数据对齐或计算有问题；3) 基准数据选择不当。建议检查数据。"
    elif alpha > 10:
        alpha_status = "✅ 表现卓越"
        alpha_interpretation = "策略显著跑赢基准，选股/择时能力非常强。"
    elif alpha > 5:
        alpha_status = "✅ 表现优秀"
        alpha_interpretation = "策略显著跑赢基准，选股/择时能力强。"
    elif alpha > 0:
        alpha_status = "✅ 表现良好"
        alpha_interpretation = "策略略优于基准。"
    elif alpha > -5:
        alpha_status = "⚠️ 表现一般"
        alpha_interpretation = "策略略低于基准，可能需要优化。"
    else:
        alpha_status = "❌ 表现较差"
        alpha_interpretation = "策略显著跑输基准，需要重新评估策略。"
    
    st.markdown(f"**状态**: {alpha_status}")
    st.markdown(f"**解读**: {alpha_interpretation}")
    st.markdown("""
    **概念**: 策略相对于基准的年化超额收益，衡量策略的选股/择时能力。
    
    **数值范围**: 理论上无限制，实际中常见为 -50% 到 +50%
    
    **计算公式**: Alpha = (策略平均日收益率 - Beta × 基准平均日收益率) × 252 × 100%
    """)
    st.divider()
    
    # Beta
    beta = metrics.get('Beta', 0)
    st.markdown("#### 📊 Beta (贝塔)")
    st.markdown(f"**当前值**: {beta:.4f}")
    
    # 根据Beta值给出更准确的解读
    if abs(beta) < 0.1:
        beta_status = "ℹ️ 与市场相关性极低"
        beta_interpretation = "策略与基准市场几乎无关，可能是独立策略或数据问题。"
    elif beta < 0:
        beta_status = "⚠️ 负相关"
        beta_interpretation = "策略与市场负相关，这在某些对冲策略中是正常的，但需要确认数据是否正确。"
    elif beta > 1.5:
        beta_status = "⚠️ 高风险"
        beta_interpretation = "策略波动显著大于市场，风险较高。"
    elif beta > 1:
        beta_status = "ℹ️ 波动大于市场"
        beta_interpretation = "策略波动大于市场，风险较高。"
    elif beta > 0.5:
        beta_status = "✅ 波动适中"
        beta_interpretation = "策略波动与市场接近，风险适中。"
    else:
        beta_status = "✅ 低风险"
        beta_interpretation = "策略波动小于市场，风险较低。"
    
    st.markdown(f"**状态**: {beta_status}")
    st.markdown(f"**解读**: {beta_interpretation}")
    st.markdown("""
    **概念**: 策略相对于基准的系统性风险系数，衡量策略与市场的相关性。
    
    **数值范围**: 理论上无限制，实际中常见为 -1 到 2
    
    **计算公式**: Beta = Cov(策略收益率, 基准收益率) / Var(基准收益率)
    
    **信号解读**:
    - **> 1**: 策略波动大于市场，风险较高
    - **≈ 1**: 策略波动与市场一致
    - **0 到 1**: 策略波动小于市场，风险较低
    - **< 0**: 策略与市场负相关（较少见）
    - **≈ 0**: 策略与市场相关性低
    """)
    st.divider()
    
    # Information Ratio
    ir = metrics.get('Information Ratio', 0)
    st.markdown("#### 📈 Information Ratio (信息比率)")
    st.markdown(f"**当前值**: {ir:.4f}")
    
    # 根据IR值给出更准确的解读
    if abs(ir) > 3:
        ir_status = "⚠️ 异常值"
        ir_interpretation = "Information Ratio值异常，可能表示：1) 策略表现极佳且跟踪误差很小；2) 数据对齐或计算有问题。建议检查数据。"
    elif ir > 1:
        ir_status = "✅ 表现卓越"
        ir_interpretation = "主动管理能力非常强，超额收益显著且稳定。"
    elif ir > 0.5:
        ir_status = "✅ 表现优秀"
        ir_interpretation = "主动管理能力强，超额收益稳定。"
    elif ir > 0:
        ir_status = "✅ 表现良好"
        ir_interpretation = "主动管理能力一般，有超额收益但不够稳定。"
    else:
        ir_status = "❌ 表现较差"
        ir_interpretation = "主动管理能力差，未能产生稳定的超额收益。"
    
    st.markdown(f"**状态**: {ir_status}")
    st.markdown(f"**解读**: {ir_interpretation}")
    st.markdown("""
    **概念**: 超额收益与跟踪误差的比值，衡量主动管理的能力。
    
    **数值范围**: 理论上无限制，实际中常见为 -2 到 2，优秀策略可能达到 1-3
    
    **计算公式**: IR = 年化超额收益 / 年化跟踪误差
    
    **信号解读**:
    - **> 1**: 主动管理能力非常强
    - **0.5 到 1**: 主动管理能力强
    - **0 到 0.5**: 主动管理能力一般
    - **< 0**: 主动管理能力差
    """)
    st.divider()
    
    # Volatility
    vol = metrics.get('Algorithm Volatility', 0)
    st.markdown("#### 📊 Algorithm Volatility (策略波动率)")
    st.markdown(f"**当前值**: {vol:.4f}%")
    st.markdown("""
    **概念**: 策略收益率的年化标准差，衡量策略的风险水平。
    
    **数值范围**: 通常为 0% 到 100%
    
    **信号解读**:
    - **< 15%**: 波动率低，风险较小
    - **15% 到 30%**: 波动率中等
    - **> 30%**: 波动率高，风险较大
    """)
    st.divider()
    
    # Win Rate
    win_rate = metrics.get('胜率', 0)
    st.markdown("#### 🎲 胜率")
    st.markdown(f"**当前值**: {win_rate:.4f}%")
    st.markdown("""
    **概念**: 盈利交易占总交易的比例。
    
    **数值范围**: 0% 到 100%
    
    **信号解读**:
    - **> 60%**: 胜率很高，策略稳定性好
    - **50% 到 60%**: 胜率良好
    - **40% 到 50%**: 胜率一般
    - **< 40%**: 胜率较低，需要优化
    """)
    st.divider()
    
    # Profit/Loss Ratio
    pl_ratio = metrics.get('盈亏比', 0)
    st.markdown("#### 💎 盈亏比")
    st.markdown(f"**当前值**: {pl_ratio:.4f}")
    st.markdown("""
    **概念**: 平均盈利与平均亏损的比值，衡量策略的风险收益特征。
    
    **数值范围**: 通常为 0 到 10
    
    **信号解读**:
    - **> 2**: 盈亏比优秀，盈利交易收益显著大于亏损
    - **1 到 2**: 盈亏比良好
    - **< 1**: 盈亏比偏低，平均亏损大于平均盈利
    """)
    st.divider()

# Streamlit主界面
def main():
    # 应用自定义样式
    apply_custom_styles()
    
    # 侧边栏配置
    with st.sidebar:
        st.markdown("### 📊 导航")
        st.markdown("---")
        
        # 使用radio button切换视图
        view_mode = st.radio(
            "选择视图",
            ["📈 风险指标", "📊 图表"],
            key="view_mode"
        )
        
        st.markdown("---")
        
        if view_mode == "📈 风险指标":
            st.markdown("### 📚 风险指标说明")
            st.markdown("""
            本页面提供详细的风险指标解释，包括：
            - 指标概念
            - 数值范围
            - 信号解读
            
            指标值会根据当前策略数据自动计算。
            """)
        else:
            st.markdown("### 📊 图表选项")
            
            # 对数轴选项
            use_log_scale = st.checkbox(
                "使用对数轴显示累计收益",
                value=False,
                key="use_log_scale",
                help="在对数轴上，相同的百分比变化显示为相同的距离，便于比较不同规模的投资表现"
            )
            
            st.markdown("---")
            st.markdown("### ℹ️ 说明")
            st.info("""
            **基准数据说明：**
            
            基准数据已从akshare获取并缓存，用于计算Alpha、Beta等相对指标。
            
            如果基准数据获取失败，相关指标将显示为0。
            """)
    
    # 主内容区域
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
        initial_capital = abs(df['成交金额'].iloc[0]) if len(df) > 0 and df['成交金额'].iloc[0] != 0 else 1000000
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
        
        # 获取基准数据（写死的数据，从缓存获取）
        benchmark_returns = None
        try:
            with st.spinner("正在获取基准数据..."):
                benchmark_returns = get_hardcoded_benchmark_data(daily_pnl['日期'])
            if benchmark_returns is not None and len(benchmark_returns) > 0:
                # 确保基准数据长度与策略数据一致
                if len(benchmark_returns) != len(daily_returns_pct):
                    # 如果长度不一致，尝试对齐
                    min_len = min(len(benchmark_returns), len(daily_returns_pct))
                    benchmark_returns = benchmark_returns[:min_len]
                    daily_returns_pct_aligned = daily_returns_pct[:min_len]
                else:
                    daily_returns_pct_aligned = daily_returns_pct
                st.success(f"✅ 成功获取基准数据，共 {len(benchmark_returns)} 个交易日")
            else:
                st.warning("⚠️ 未能获取基准数据，基准相关指标将显示为0")
                benchmark_returns = None
                daily_returns_pct_aligned = daily_returns_pct
        except Exception as e:
            st.warning(f"⚠️ 获取基准数据时出错: {str(e)[:100]}，基准相关指标将显示为0")
            benchmark_returns = None
            daily_returns_pct_aligned = daily_returns_pct
        
        # 计算风险指标
        metrics = calculate_risk_metrics(daily_returns_pct_aligned, total_returns_pct, initial_capital, daily_pnl, benchmark_returns)
    
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
    
    # 根据侧边栏选择显示不同内容
    view_mode = st.session_state.get('view_mode', '📈 风险指标')
    
    if view_mode == "📈 风险指标":
        # 风险指标详细说明页面
        st.header("📈 风险指标详解")
        
        # 显示风险指标说明
        show_risk_metrics_explanation(metrics)
        
    else:
        # 图表页面
        # 交易信号图（主要图表）
        st.header("📊 交易信号图")
        st.info("💡 **交易信号图说明**：上图显示价格走势，绿色▲表示买入信号（开多），红色▼表示卖出信号（平多）。下图显示每笔交易的盈亏和累计收益。")
        
        use_log_scale = st.session_state.get('use_log_scale', False)
        fig_signals = plot_trading_signals(df)
        st.plotly_chart(fig_signals, use_container_width=True)
        
        st.divider()
        
        # 风险分析图表
        st.header("📈 风险分析图表")
        
        # 绘制图表（对数轴选项在侧边栏）
        fig = plot_risk_charts(df, daily_pnl, metrics, use_log_scale=use_log_scale)
        st.plotly_chart(fig, use_container_width=True)
    
    # 数据表格
    with st.expander("查看详细交易数据"):
        # 选择要显示的列（使用实际存在的列名）
        display_cols = ['日期', '委托时间', '标的', '交易类型', '成交数量']
        # 添加价格列（可能是'成交价'或'成交价格'）
        if '成交价格' in df.columns:
            display_cols.append('成交价格')
        elif '成交价' in df.columns:
            display_cols.append('成交价')
        # 添加金额列（可能是'成交额'或'成交金额'）
        if '成交金额' in df.columns:
            display_cols.append('成交金额')
        elif '成交额' in df.columns:
            display_cols.append('成交额')
        display_cols.extend(['平仓盈亏', '手续费', '净盈亏', '累计收益'])
        # 只显示存在的列
        available_cols = [col for col in display_cols if col in df.columns]
        st.dataframe(df[available_cols], use_container_width=True)
    
    with st.expander("查看日度汇总数据"):
        st.dataframe(daily_pnl, use_container_width=True)

if __name__ == "__main__":
    main()
