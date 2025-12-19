import pandas as pd 
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

import akshare as ak  # 获取AU0黄金主力连续合约

# 常量

Rf = 0.04                    # 无风险利率
TRADING_DAYS_PER_YEAR = 250  # 每年交易日数，由聚宽API公式得到


#公共风险指标函数

def get_total_returns(Pend, Pstart):
    """Total Returns 策略收益 - Total Returns=(Pend−Pstart)/Pstart∗100%"""
    if Pstart == 0:
        return 0.0
    return (Pend - Pstart) / Pstart * 100


def get_total_annualized_returns(P, n):
    """Total Annualized Returns 策略年化收益 - Rp=((1+P)的250/n次方−1)∗100%"""
    if n <= 0:
        return 0.0
    return ((1 + P) ** (TRADING_DAYS_PER_YEAR / n) - 1) * 100


def get_beta(Dp, Dm):
    """Beta 贝塔 - Beta=Cov(Dp,Dm)/Var(Dm)"""
    if Dm is None:
        return 0.0
    min_len = min(len(Dp), len(Dm))
    Dp_clean = np.asarray(Dp[:min_len])
    Dm_clean = np.asarray(Dm[:min_len])
    mask = ~(np.isnan(Dp_clean) | np.isnan(Dm_clean) |
             np.isinf(Dp_clean) | np.isinf(Dm_clean))
    Dp_clean = Dp_clean[mask]
    Dm_clean = Dm_clean[mask]
    if len(Dp_clean) < 2:
        return 0.0
    cov = np.cov(Dp_clean, Dm_clean)[0, 1]
    var_dm = np.var(Dm_clean, ddof=0)
    return cov / var_dm if var_dm > 1e-10 else 0.0


def get_alpha(Rp, Rm, βp):
    """Alpha 阿尔法 - Alpha=Rp-[Rf+βp(Rm-Rf)]，这里 Rp/Rm 传入为小数形式"""
    if Rm is None:
        return 0.0
    α = Rp - (Rf + βp * (Rm - Rf))
    return α * 100


def get_sharpe(Rp, σp):
    """Sharpe 夏普比率 - Sharpe=(Rp-Rf)/σp，Rp和σp都是小数形式"""
    if σp is None or σp == 0 or σp < 1e-10:
        return 0.0
    return (Rp - Rf) / σp


def get_sortino(Rp, σpd):
    """Sortino 索提诺比率 - Sortino=(Rp-Rf)/σpd，Rp和σpd都是小数形式"""
    if σpd is None or σpd == 0 or σpd < 1e-10:
        return 0.0
    return (Rp - Rf) / σpd


def get_information_ratio(Rp, Rm, Dp, Dm):
    """Information Ratio 信息比率 - IR=(Rp-Rm)/σt，Rp和Rm是百分比形式"""
    if Dm is None or Rm is None or len(Dp) < 2 or len(Dm) < 2:
        return 0.0
    
    n = min(len(Dp), len(Dm))
    excess = Dp[:n] - Dm[:n]  # 策略与基准每日收益差值
    σt = np.std(excess, ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR) * 100  # 年化标准差（%）
    
    return (Rp - Rm) / σt if σt > 1e-10 else 0.0


def get_algorithm_volatility(Dp):
    """Algorithm Volatility 策略波动率 - σp=根号下250/(n−1)*∑(rp−rp均值)^2"""
    Dp_clean = np.asarray(Dp)
    Dp_clean = Dp_clean[~np.isnan(Dp_clean)]
    Dp_clean = Dp_clean[~np.isinf(Dp_clean)]
    if len(Dp_clean) < 2:
        return 0.0
    σp = np.std(Dp_clean, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR) * 100
    return σp


def get_benchmark_volatility(Dm):
    """Benchmark Volatility 基准波动率 - σm=根号下250/(n−1)*∑(rm−rm均值)^2"""
    if Dm is None:
        return 0.0
    Dm_clean = np.asarray(Dm)
    Dm_clean = Dm_clean[~np.isnan(Dm_clean)]
    Dm_clean = Dm_clean[~np.isinf(Dm_clean)]
    if len(Dm_clean) < 2:
        return 0.0
    σm = np.std(Dm_clean, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR) * 100
    return σm


def get_max_drawdown(Dp):
    """Max Drawdown 最大回撤 - 基于权益曲线"""
    r = np.asarray(Dp)
    r = r[~np.isnan(r)]
    r = r[~np.isinf(r)]
    if len(r) == 0:
        return 0.0
    cum = np.cumprod(1 + r) - 1
    running_max = np.maximum.accumulate(cum)
    dd = cum - running_max
    return np.min(dd) * 100


def get_downside_risk(Dp):
    """
    Downside Risk 下行波动率
    σpd=根号下(250/n)∑(rp−△rpi均值)2f(t)
    rp=策略每日收益率, △rpi均值=策略至第i日平均收益率=(1/i)∑(j从1到i)rj
    f(t)=1 if rp<△rpi, f(t)=0 if rp>=△rpi
    """
    r = np.asarray(Dp)
    r = r[~np.isnan(r)]
    r = r[~np.isinf(r)]
    n = len(r)
    if n < 2:
        return 0.0
    downside_sum = 0.0
    for i in range(1, n + 1):  # i从1到n
        rpi_mean = np.mean(r[0:i])  # 至第i日平均收益率
        rp_i = r[i - 1]  # 第i日收益率
        if rp_i < rpi_mean:
            downside_sum += (rp_i - rpi_mean) ** 2
    σpd = np.sqrt((TRADING_DAYS_PER_YEAR / n) * downside_sum) * 100
    return σpd

def get_win_rate(win_trades, loss_trades):
    """胜率 = 盈利交易次数/总交易次数"""
    total = win_trades + loss_trades
    return (win_trades / total * 100) if total > 0 else 0.0


def get_daily_win_rate(Dp, Dm):
    """日胜率 = 当日策略收益跑赢当日基准收益的天数/总交易日数"""
    if Dm is None:
        return 0.0
    min_len = min(len(Dp), len(Dm))
    sp = np.asarray(Dp[:min_len])
    bm = np.asarray(Dm[:min_len])
    mask = ~(np.isnan(sp) | np.isnan(bm) | np.isinf(sp) | np.isinf(bm))
    sp = sp[mask]
    bm = bm[mask]
    if len(sp) == 0:
        return 0.0
    return (sp > bm).sum() / len(sp) * 100


def get_profit_loss_ratio(total_profit, total_loss):
    """盈亏比 = 总盈利额/总亏损额"""
    return (total_profit / total_loss) if total_loss > 0 else 0.0


def get_aei(Dp, Dm):
    """AEI 日均超额收益"""
    if Dm is None:
        return 0.0
    min_len = min(len(Dp), len(Dm))
    sp = np.asarray(Dp[:min_len])
    bm = np.asarray(Dm[:min_len])
    mask = ~(np.isnan(sp) | np.isnan(bm) | np.isinf(sp) | np.isinf(bm))
    sp = sp[mask]
    bm = bm[mask]
    if len(sp) == 0:
        return 0.0
    sp_cum = np.cumprod(1 + sp) - 1
    bm_cum = np.cumprod(1 + bm) - 1
    ei = (1 + sp_cum) / (1 + bm_cum) - 1
    dei = np.diff(ei, prepend=0)
    return dei.mean() * 100


def get_excess_return_max_drawdown(Dp, Dm):
    """超额收益最大回撤"""
    if Dm is None:
        return 0.0
    min_len = min(len(Dp), len(Dm))
    sp = np.asarray(Dp[:min_len])
    bm = np.asarray(Dm[:min_len])
    mask = ~(np.isnan(sp) | np.isnan(bm) | np.isinf(sp) | np.isinf(bm))
    sp = sp[mask]
    bm = bm[mask]
    if len(sp) == 0:
        return 0.0
    sp_cum = np.cumprod(1 + sp) - 1
    bm_cum = np.cumprod(1 + bm) - 1
    ei = (1 + sp_cum) / (1 + bm_cum) - 1
    ei_running_max = np.maximum.accumulate(ei)
    dd = ei - ei_running_max
    return np.min(dd) * 100


def get_excess_return_sharpe(Dp, Dm):
    """超额收益夏普比率 EI Sharpe Ratio=(RpEI-Rf)/σpEI"""
    if Dm is None:
        return 0.0
    min_len = min(len(Dp), len(Dm))
    sp = np.asarray(Dp[:min_len])
    bm = np.asarray(Dm[:min_len])
    mask = ~(np.isnan(sp) | np.isnan(bm) | np.isinf(sp) | np.isinf(bm))
    sp = sp[mask]
    bm = bm[mask]
    if len(sp) < 2:
        return 0.0
    excess = sp - bm
    mean_excess = excess.mean()
    RpEI = mean_excess * TRADING_DAYS_PER_YEAR * 100  # 年化超额收益率（%）
    σpEI = np.std(excess, ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR) * 100
    if σpEI == 0:
        return 0.0
    return (RpEI - Rf * 100) / σpEI

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

def calculate_risk_metrics(daily_returns, total_returns, initial_capital, daily_pnl, benchmark_returns=None, trade_df=None):
    """
    计算各种风险指标
    参数:
        daily_returns: 策略日收益率（小数形式）
        total_returns: 策略总收益率（百分比）
        initial_capital: 初始资金
        daily_pnl: DataFrame，包含日期和累计收益
        benchmark_returns: 基准日收益率（小数形式）
        trade_df: DataFrame，原始交易数据，用于计算胜率和盈亏比
    返回:
        dict: 包含所有风险指标的字典
    """
    # 转为 numpy 并清理数据
    if isinstance(daily_returns, pd.Series):
        daily_returns = daily_returns.values
    
    Dp = np.asarray(daily_returns)  # 策略每日收益 Dp
    Dp = Dp[~np.isnan(Dp)]
    Dp = Dp[~np.isinf(Dp)]
    if len(Dp) == 0:
        return {}
    
    # 策略收益 P, Rp, Pstart, Pend
    Pstart = float(initial_capital)
    Pend = Pstart + float(daily_pnl['累计收益'].iloc[-1]) if len(daily_pnl) > 0 else Pstart
    P = (Pend - Pstart) / Pstart if Pstart > 0 else 0.0  # 策略收益（小数）
    n = len(daily_pnl)  # 策略执行天数

    # 基准收益 Rm, Dm
    if benchmark_returns is not None:
        Dm = np.asarray(benchmark_returns)
        Dm = Dm[~np.isnan(Dm)]
        Dm = Dm[~np.isinf(Dm)]
        if len(Dm) == 0:
            Dm = None
    else:
        Dm = None

    if Dm is not None and len(Dm) > 0:
        bm_total = np.prod(1 + Dm) - 1
        n_bm = len(Dm)
        Rm = get_total_annualized_returns(bm_total, n_bm)  # 基准年化收益（%）
    else:
        Rm = None

    # 计算波动率（百分比形式）
    σp_pct = get_algorithm_volatility(Dp)  # % 形式
    σm_pct = get_benchmark_volatility(Dm)  # % 形式
    σpd_pct = get_downside_risk(Dp)  # % 形式
    
    # 转换为小数形式（用于 Sharpe/Sortino 计算）
    σp = σp_pct / 100 if σp_pct > 0 else 0.0
    σpd = σpd_pct / 100 if σpd_pct > 0 else 0.0

    # 计算风险指标
    Rp_pct = get_total_annualized_returns(P, n)  # 策略年化收益 Rp（%）
    Rp = Rp_pct / 100  # 转换为小数形式
    
    # 基准年化收益（小数形式）
    if Rm is not None:
        Rm_decimal = Rm / 100  # Rm 是百分比，转换为小数
    else:
        Rm_decimal = None
    
    βp = get_beta(Dp, Dm)  # Beta
    alpha = get_alpha(Rp, Rm_decimal, βp) if Rm_decimal is not None else 0.0
    sharpe = get_sharpe(Rp, σp) if σp > 1e-10 else 0.0
    sortino = get_sortino(Rp, σpd) if σpd > 1e-10 else 0.0
    info_ratio = get_information_ratio(Rp_pct, Rm, Dp, Dm) if Rm is not None else 0.0

    # 胜率和盈亏比（按实际交易计算，每次卖出记为一次交易）
    if trade_df is not None and '交易类型' in trade_df.columns and '平仓盈亏' in trade_df.columns:
        # 筛选卖出交易（平仓交易）
        close_mask = trade_df['交易类型'].astype(str).str.contains('平', na=False)
        df_close = trade_df[close_mask].copy()
        
        if not df_close.empty:
            # 计算胜率：盈利交易次数 / 总交易次数（包括所有平仓交易，含盈亏为0的）
            win_mask = df_close['平仓盈亏'] > 0
            loss_mask = df_close['平仓盈亏'] < 0
            win_trades = win_mask.sum()
            loss_trades = loss_mask.sum()
            total_trades = len(df_close)  # 所有平仓交易次数，包括盈亏为0的
            win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0.0
            
            # 计算盈亏比：总盈利额 / 总亏损额（用平仓盈亏，不含手续费）
            total_profit = df_close[win_mask]['平仓盈亏'].sum()  # 总盈利额（平仓盈亏，正数）
            total_loss = abs(df_close[loss_mask]['平仓盈亏'].sum())  # 总亏损额（平仓盈亏，绝对值）
            pl_ratio = (total_profit / total_loss) if total_loss > 0 else 0.0
        else:
            win_rate = 0.0
            pl_ratio = 0.0
    else:
        # 如果没有交易数据，回退到按日收益计算
        winning_trades = (Dp > 0).sum()
        total_trades = len(Dp)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
        
        if winning_trades > 0 and (total_trades - winning_trades) > 0:
            avg_win = Dp[Dp > 0].mean()
            avg_loss = abs(Dp[Dp < 0].mean())
            pl_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0
        else:
            pl_ratio = 0.0
    
    # 日胜率和其他超额指标
    daily_win_rate = get_daily_win_rate(Dp, Dm)
    aei = get_aei(Dp, Dm)
    excess_mdd = get_excess_return_max_drawdown(Dp, Dm)
    excess_sharpe = get_excess_return_sharpe(Dp, Dm)

    # 超额收益（除法版）与对数轴超额收益
    if Dm is not None and len(Dm) > 0:
        min_len = min(len(Dp), len(Dm))
        sp = Dp[:min_len]
        bm = Dm[:min_len]
        mask = ~(np.isnan(sp) | np.isnan(bm) | np.isinf(sp) | np.isinf(bm))
        sp = sp[mask]
        bm = bm[mask]
        if len(sp) > 0:
            sp_total = np.prod(1 + sp) - 1
            bm_total = np.prod(1 + bm) - 1
            if abs(bm_total) > 1e-10:
                excess_div = (sp_total / bm_total - 1) * 100
                log_excess = (np.log1p(sp_total) - np.log1p(bm_total)) * 100
            else:
                excess_div = 0.0
                log_excess = 0.0
        else:
            excess_div = 0.0
            log_excess = 0.0
    else:
        excess_div = 0.0
        log_excess = 0.0

    metrics = {
        'Total Returns': get_total_returns(Pend, Pstart),
        'Total Annualized Returns': Rp_pct,
        'Alpha': alpha,
        'Beta': βp,
        'Sharpe': sharpe,
        'Sortino': sortino,
        'Information Ratio': info_ratio,
        'Algorithm Volatility': σp_pct,
        'Benchmark Volatility': σm_pct,
        'Max Drawdown': get_max_drawdown(Dp),
        'Downside Risk': σpd_pct,
        '胜率': win_rate,
        '日胜率': daily_win_rate,
        '盈亏比': pl_ratio,
        'AEI': aei,
        '超额收益最大回撤': excess_mdd,
        '超额收益夏普比率': excess_sharpe,
        '超额收益': excess_div,
        '对数轴上的超额收益': log_excess,
    }

    return metrics


@st.cache_data
def get_benchmark_daily_returns_aligned(trade_dates, symbol: str = "AU0"):
    """
    使用 akshare 动态获取基准（AU0 主力连续合约）并对齐到交易日期，返回基准日收益率序列 Dm（小数）。
    """
    try:
        benchmark_df = ak.futures_zh_daily_sina(symbol=symbol)
    except Exception:
        return None

    if benchmark_df is None or benchmark_df.empty:
        return None

    # 处理日期列
    if isinstance(benchmark_df.index, pd.DatetimeIndex):
        benchmark_df['日期'] = pd.to_datetime(benchmark_df.index)
    elif 'date' in benchmark_df.columns:
        benchmark_df['日期'] = pd.to_datetime(benchmark_df['date'], errors='coerce')
    else:
        first_col = benchmark_df.columns[0]
        benchmark_df['日期'] = pd.to_datetime(benchmark_df[first_col], errors='coerce')

    benchmark_df['日期'] = pd.to_datetime(benchmark_df['日期']).dt.normalize()

    # 筛选交易日期范围
    start_date = pd.to_datetime(trade_dates.min()).normalize()
    end_date = pd.to_datetime(trade_dates.max()).normalize()
    benchmark_df = benchmark_df[(benchmark_df['日期'] >= start_date) & (benchmark_df['日期'] <= end_date)].copy()
    benchmark_df = benchmark_df.sort_values('日期').reset_index(drop=True)

    if 'close' in benchmark_df.columns:
        prices = pd.to_numeric(benchmark_df['close'], errors='coerce').values
    else:
        return None

    valid_mask = ~np.isnan(prices)
    prices_clean = prices[valid_mask]
    dates_filtered = benchmark_df.loc[valid_mask, '日期'].values

    if len(prices_clean) < 2:
        return None

    benchmark_returns_raw = np.diff(prices_clean) / prices_clean[:-1]
    benchmark_dates_raw = dates_filtered[1:]

    benchmark_df_aligned = pd.DataFrame({
        '日期': pd.to_datetime(benchmark_dates_raw).astype('datetime64[ns]').astype('datetime64[ns]').astype('datetime64[ns]').astype('datetime64[ns]'),
        '收益率': benchmark_returns_raw
    }).drop_duplicates(subset=['日期']).sort_values('日期').reset_index(drop=True)

    trade_df = pd.DataFrame({'日期': pd.to_datetime(trade_dates).dt.normalize()})

    merged = trade_df.merge(benchmark_df_aligned, on='日期', how='left')
    if merged['收益率'].isna().all():
        merged['收益率'] = 0.0
    else:
        if pd.isna(merged.loc[0, '收益率']):
            merged.loc[0, '收益率'] = 0.0
        merged['收益率'] = merged['收益率'].ffill().fillna(0.0)

    return merged['收益率'].values

# 获取 AU0 主力合约收盘价数据,根据收盘价绘制交易信号图，信号图画在收盘价曲线上
@st.cache_data
def get_au0_close_prices(start_date=None, end_date=None):
    """
    从 akshare 获取 AU0 主力合约的收盘价数据
    
    参数:
    start_date: 开始日期，格式 'YYYY-MM-DD' 或 datetime，或 None
    end_date: 结束日期，格式 'YYYY-MM-DD' 或 datetime，或 None
    
    返回:
    price_df: DataFrame，包含 '日期' 和 '收盘价' 列，如果失败返回 None
    """
    try:
        # AU0 主力合约数据作为基准
        df = ak.futures_zh_daily_sina(symbol="AU0")

        if df is None or df.empty:
            return None
        
        # 处理日期列
        if isinstance(df.index, pd.DatetimeIndex):
            df['日期'] = pd.to_datetime(df.index)
        elif 'date' in df.columns:
            df['日期'] = pd.to_datetime(df['date'])
        elif '日期' in df.columns:
            df['日期'] = pd.to_datetime(df['日期'])
        else:
            # 尝试第一列作为日期
            first_col = df.columns[0]
            try:
                df['日期'] = pd.to_datetime(df[first_col])
            except:
                return None
        
        # 获取收盘价列
        if 'close' in df.columns:
            df['收盘价'] = pd.to_numeric(df['close'], errors='coerce')
        elif '收盘' in df.columns:
            df['收盘价'] = pd.to_numeric(df['收盘'], errors='coerce')
        elif '收盘价' in df.columns:
            df['收盘价'] = pd.to_numeric(df['收盘价'], errors='coerce')
        else:
            # 尝试找到包含'close'或'收盘'的列
            close_cols = [col for col in df.columns if 'close' in col.lower() or '收盘' in col]
            if close_cols:
                df['收盘价'] = pd.to_numeric(df[close_cols[0]], errors='coerce')
            else:
                return None
        
        # 确保日期列为 datetime 类型，只保留日期部分
        df['日期'] = pd.to_datetime(df['日期']).dt.normalize()
        
        # 筛选日期范围
        if start_date is not None:
            start_date_dt = pd.to_datetime(start_date)
            df = df[df['日期'] >= start_date_dt]
        if end_date is not None:
            end_date_dt = pd.to_datetime(end_date)
            df = df[df['日期'] <= end_date_dt]
        
        # 排序并清理数据
        df = df.sort_values('日期').reset_index(drop=True)
        df = df.dropna(subset=['日期', '收盘价'])
        
        # 返回日期和收盘价
        result_df = df[['日期', '收盘价']].copy()
        result_df = result_df.sort_values('日期').reset_index(drop=True)
        
        return result_df if len(result_df) > 0 else None
        
    except Exception as e:
        return None

# 绘制交易信号图
def plot_trading_signals(df):
    """
    绘制交易信号图，显示价格走势和买卖信号点
    信号图绘制在 AU0 主力合约的收盘价曲线上，而非折线图
    """
    # 创建子图：仅价格走势（信号图单独一栏）
    fig = make_subplots(
        rows=1, cols=1,
        
    )
    
    # 按日期时间排序
    df_sorted = df.sort_values('日期时间').reset_index(drop=True)
    
    # 获取交易日期范围
    start_date = df_sorted['日期时间'].min()
    end_date = df_sorted['日期时间'].max()
    
    # 从 akshare 获取 AU0 主力合约收盘价数据
    price_df = get_au0_close_prices(start_date=start_date, end_date=end_date)
    
    if price_df is not None and len(price_df) > 0:
        # 使用收盘价数据绘制价格曲线
        fig.add_trace(
            go.Scatter(
                x=price_df['日期'],
                y=price_df['收盘价'],
                mode='lines',
                name='AU0收盘价',
                line=dict(color='#1f77b4', width=2),
                hovertemplate='日期: %{x|%Y-%m-%d}<br>收盘价: %{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 为交易信号找到对应的收盘价
        # 将交易日期时间转换为日期（只保留日期部分）
        df_sorted['交易日期'] = pd.to_datetime(df_sorted['日期时间']).dt.normalize()#normalize()将时间部分设为00:00:00
        
        # 提取买入和卖出信号
        buy_signals = df_sorted[df_sorted['交易类型'].str.contains('开多', na=False)].copy()
        sell_signals = df_sorted[df_sorted['交易类型'].str.contains('平多', na=False)].copy()
        
        # 标记买入信号（开多）- 使用收盘价
        if len(buy_signals) > 0:
            # 合并收盘价数据
            buy_signals = buy_signals.merge(
                price_df[['日期', '收盘价']], 
                left_on='交易日期', 
                right_on='日期', 
                how='left'
            )
            
            # 如果某天没有收盘价，使用前向填充或后向填充
            buy_signals['收盘价'] = buy_signals['收盘价'].ffill().bfill()
            
            # 过滤掉仍然没有收盘价的数据
            buy_signals = buy_signals[buy_signals['收盘价'].notna()]#notna()用于过滤非空值
            
            if len(buy_signals) > 0:#只有在有有效数据时才添加标记
                fig.add_trace(#add_trace用于向图表中添加数据
                    go.Scatter(
                        x=buy_signals['日期时间'],
                        y=buy_signals['收盘价'],
                        mode='markers',
                        name='买入信号（开多）',
                        marker=dict(
                            symbol='triangle-up',
                            size=14,
                            color='green',
                            line=dict(width=2, color='darkgreen')
                        ),
                        hovertemplate=(
                            '买入时间: %{x|%Y-%m-%d %H:%M:%S}<br>'
                            '收盘价: %{y:.2f}<br>'
                            '成交价: %{customdata[0]:.2f}<br>'
                            '数量: %{customdata[1]}手<extra></extra>'
                        ),
                        customdata=buy_signals[['成交价', '成交数量']].values
                    ),
                    row=1, col=1
                )
        
        # 标记卖出信号（平多）- 使用收盘价
        if len(sell_signals) > 0:
            # 合并收盘价数据
            sell_signals = sell_signals.merge(
                price_df[['日期', '收盘价']], 
                left_on='交易日期', 
                right_on='日期', 
                how='left'
            )
            
            # 如果某天没有收盘价，使用前向填充或后向填充
            sell_signals['收盘价'] = sell_signals['收盘价'].ffill().bfill()
            
            # 过滤掉仍然没有收盘价的数据
            sell_signals = sell_signals[sell_signals['收盘价'].notna()]
            
            if len(sell_signals) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=sell_signals['日期时间'],
                        y=sell_signals['收盘价'],
                        mode='markers',
                        name='卖出信号（平多）',
                        marker=dict(
                            symbol='triangle-down',
                            size=14,
                            color='red',
                            line=dict(width=2, color='darkred')
                        ),
                        hovertemplate=(
                            '卖出时间: %{x|%Y-%m-%d %H:%M:%S}<br>'
                            '收盘价: %{y:.2f}<br>'
                            '成交价: %{customdata[0]:.2f}<br>'
                            '盈亏: %{customdata[1]:.0f}<extra></extra>'
                        ),
                        customdata=sell_signals[['成交价', '平仓盈亏']].values
                    ),
                    row=1, col=1
                )
        
        # 注意：上面已经基于收盘价画过一次买入/卖出信号，这里不再重复画基于成交价的信号，避免图上出现重复三角形
    
    # 更新布局
    fig.update_layout(
        height=800,
        title_text="交易信号图（基于收盘价曲线）",
        title_x=0.5,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white'
    )
    
    # 更新Y轴标签
    if price_df is not None and len(price_df) > 0:
        fig.update_yaxes(title_text="AU0收盘价", row=1, col=1)
    else:
        fig.update_yaxes(title_text="成交价", row=1, col=1)
    
    # 更新X轴日期格式（不显示标签，只设置日期格式）
    fig.update_xaxes(tickformat="%Y-%m-%d", row=1, col=1)
    
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
    y_data = daily_pnl['累计收益'].values
    yaxis_title = "累计收益"
    
    if use_log_scale:
        # 对于对数轴，需要确保所有值为正数
        y_min = y_data.min()
        if y_min <= 0:
            # 如果有负值或0值，需要偏移使所有值为正
            # 偏移量 = abs(最小值) + 1，确保最小值为1
            offset = abs(y_min) + 1
            y_data_log = y_data + offset
        else:
            # 如果所有值都为正，使用原始值
            offset = 0
            y_data_log = y_data
        
        # 使用对数轴显示
        y_data = y_data_log
        yaxis_title = "累计收益 (对数轴)"
        
        # 计算Y轴范围，确保能显示所有数据
        y_max = y_data.max()
        y_min_log = y_data.min()
        # 设置Y轴范围，留一些边距
        # 对于对数轴，最小值至少为1（因为log(1)=0），但可以更小以显示更多细节
        if y_min_log > 0:
            # 使用最小值的较小比例，但至少为1
            y_range_min = max(1, y_min_log * 0.3)  # 至少从1开始，或者最小值的30%
        else:
            y_range_min = 1  # 如果最小值异常，至少从1开始
        y_range_max = y_max * 1.5  # 最大值增加50%的边距，确保能看到所有数据
    else:
        offset = 0
        y_range_min = None
        y_range_max = None
    
    fig.add_trace(
        go.Scatter(
            x=daily_pnl['日期'],
            y=y_data,
            mode='lines+markers',
            name='累计收益',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=3),
            hovertemplate='日期: %{x}<br>累计收益: %{customdata:,.0f}<extra></extra>',
            customdata=daily_pnl['累计收益'].values  # 显示原始值
        ),
        row=1, col=1
    )
    
    # 添加零线
    if not use_log_scale:
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
    
    # 2. 日盈亏曲线（改为连续折线而不是柱状，避免“离散点”的感觉）
    colors = ['green' if x > 0 else 'red' for x in daily_pnl['日盈亏']]
    fig.add_trace(
        go.Scatter(
            x=daily_pnl['日期'],
            y=daily_pnl['日盈亏'],
            mode='lines+markers',
            name='日盈亏',
            line=dict(color='gray', width=1),
            marker=dict(color=colors, size=4),
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
            mode='lines+markers',
            name='回撤',
            fill='tozeroy',
            fillcolor='rgba(255,0,0,0.3)',
            line=dict(color='red', width=1),
            marker=dict(size=3, color='red'),
            hovertemplate='日期: %{x}<br>回撤: %{y:,.0f}<extra></extra>'
        ),
        row=3, col=1
    )
    
    # 4. 累计收益率曲线
    fig.add_trace(
        go.Scatter(
            x=daily_pnl['日期'],
            y=daily_pnl['累计收益率'],
            mode='lines+markers',
            name='累计收益率',
            line=dict(color='green', width=2),
            marker=dict(size=3, color='green'),
            hovertemplate='日期: %{x}<br>累计收益率: %{y:.2f}%<extra></extra>'
        ),
        row=4, col=1
    )
    
    # 添加零线
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=4, col=1)
    
    # 更新布局
    fig.update_layout(
        height=1200,
        title_text="策略风险分析",
        title_x=0.5,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white'
    )
    
    # 更新Y轴标签
    if use_log_scale:
        # 设置对数轴类型和范围
        # Plotly的对数轴range参数使用对数空间的值 [log10(min), log10(max)]
        fig.update_yaxes(
            title_text=yaxis_title,
            type="log",
            range=[np.log10(y_range_min), np.log10(y_range_max)],
            row=1, col=1
        )
    else:
        fig.update_yaxes(title_text=yaxis_title, row=1, col=1)
    
    fig.update_yaxes(title_text="日盈亏", row=2, col=1)
    fig.update_yaxes(title_text="回撤", row=3, col=1)
    fig.update_yaxes(title_text="累计收益率 (%)", row=4, col=1)
    
    # 更新X轴日期格式（不显示标签，只设置日期格式）
    fig.update_xaxes(tickformat="%Y-%m-%d", row=1, col=1)
    fig.update_xaxes(tickformat="%Y-%m-%d", row=2, col=1)
    fig.update_xaxes(tickformat="%Y-%m-%d", row=3, col=1)
    fig.update_xaxes(tickformat="%Y-%m-%d", row=4, col=1)
    
    return fig

# Streamlit主界面
def main():
    st.title("📊 风险指标与交易信号图")
    
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
        
        # 按日期聚合（基于成交记录先得到“有交易日”的日盈亏）
        df['日期_仅'] = pd.to_datetime(df['日期'], format='%Y/%m/%d', errors='coerce')
        daily_pnl = df.groupby('日期_仅')['净盈亏'].sum().reset_index()
        daily_pnl.columns = ['日期', '日盈亏']
        daily_pnl = daily_pnl.sort_values('日期').reset_index(drop=True)

        # 使用真实期货交易日历补全“无成交日”（逐日盯市：无成交日的日盈亏=0，权益横盘）
        price_calendar = get_au0_close_prices(
            start_date=daily_pnl['日期'].min(),
            end_date=daily_pnl['日期'].max()
        )
        if price_calendar is not None and len(price_calendar) > 0:
            all_days = price_calendar[['日期']].drop_duplicates().sort_values('日期')
            daily_pnl = all_days.merge(daily_pnl, on='日期', how='left')
        # 无成交日填0
        daily_pnl['日盈亏'] = daily_pnl['日盈亏'].fillna(0.0)
        daily_pnl['累计收益'] = daily_pnl['日盈亏'].cumsum()

        # 按 need.txt 逻辑，用“前一日权益”计算日收益率，使权益曲线和图表在所有交易日连续
        equity_prev = initial_capital + daily_pnl['累计收益'].shift(1).fillna(0.0)
        equity_prev = equity_prev.replace(0, np.nan)
        daily_ret = daily_pnl['日盈亏'] / equity_prev
        if np.isnan(daily_ret.iloc[0]):
            daily_ret.iloc[0] = daily_pnl['日盈亏'].iloc[0] / initial_capital
        daily_ret = daily_ret.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        daily_pnl['日收益率'] = daily_ret * 100
        daily_pnl['累计收益率'] = (daily_pnl['累计收益'] / initial_capital) * 100
        
        # 计算日收益率序列（小数形式）
        daily_returns_pct = daily_ret.values
        total_returns_pct = daily_pnl['累计收益'].iloc[-1] / initial_capital * 100 if len(daily_pnl) > 0 else 0
        
        # 使用 akshare 动态获取基准数据并对齐到交易日期
        benchmark_returns = get_benchmark_daily_returns_aligned(daily_pnl['日期'])
        
        # 确保数据长度一致
        if benchmark_returns is not None and len(benchmark_returns) != len(daily_returns_pct):
            min_len = min(len(benchmark_returns), len(daily_returns_pct))
            benchmark_returns = benchmark_returns[:min_len]
            daily_returns_pct_aligned = daily_returns_pct[:min_len]
            daily_pnl_aligned = daily_pnl.iloc[:min_len].copy()
        else:
            daily_returns_pct_aligned = daily_returns_pct
            daily_pnl_aligned = daily_pnl.copy()
        
        st.success(f"✅ 交易详情数据加载完成，共 {len(daily_pnl_aligned)} 个交易日")
        
        # 计算风险指标（使用对齐后的 daily_pnl 和基准，传入原始交易数据用于计算胜率和盈亏比）
        metrics = calculate_risk_metrics(daily_returns_pct_aligned, total_returns_pct, initial_capital, daily_pnl_aligned, benchmark_returns, trade_df=df)
    
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
    
    # 将指标分为多列显示
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("收益指标")
        st.write(f"**Total Returns (策略收益)**: {metrics.get('Total Returns', 0):.4f}%")
        st.write(f"**Total Annualized Returns (策略年化收益)**: {metrics.get('Total Annualized Returns', 0):.4f}%")
        st.write(f"**Alpha (阿尔法)**: {metrics.get('Alpha', 0):.4f}%")
        st.write(f"**Beta (贝塔)**: {metrics.get('Beta', 0):.4f}")
        st.write(f"**AEI (日均超额收益)**: {metrics.get('AEI', 0):.4f}%")
        st.write(f"**超额收益**: {metrics.get('超额收益', 0):.4f}%")
        st.write(f"**对数轴上的超额收益**: {metrics.get('对数轴上的超额收益', 0):.4f}%")
    
    with col2:
        st.subheader("风险指标")
        st.write(f"**Sharpe (夏普比率)**: {metrics.get('Sharpe', 0):.4f}")
        st.write(f"**Sortino (索提诺比率)**: {metrics.get('Sortino', 0):.4f}")
        st.write(f"**Information Ratio (信息比率)**: {metrics.get('Information Ratio', 0):.4f}")
        st.write(f"**Algorithm Volatility (策略波动率)**: {metrics.get('Algorithm Volatility', 0):.4f}%")
        st.write(f"**Benchmark Volatility (基准波动率)**: {metrics.get('Benchmark Volatility', 0):.4f}%")
        st.write(f"**Max Drawdown (最大回撤)**: {metrics.get('Max Drawdown', 0):.4f}%")
        st.write(f"**Downside Risk (下行波动率)**: {metrics.get('Downside Risk', 0):.4f}%")
    
    with col3:
        st.subheader("交易统计")
        st.write(f"**胜率**: {metrics.get('胜率', 0):.4f}%")
        st.write(f"**日胜率**: {metrics.get('日胜率', 0):.4f}%")
        st.write(f"**盈亏比**: {metrics.get('盈亏比', 0):.4f}")
        st.write(f"**超额收益最大回撤**: {metrics.get('超额收益最大回撤', 0):.4f}%")
        st.write(f"**超额收益夏普比率**: {metrics.get('超额收益夏普比率', 0):.4f}")
    
    st.divider()
    
    st.header("风险指标总结")
    
    # 动态生成总结，使用实际计算出的指标
    total_returns = metrics.get('Total Returns', 0)
    annualized_returns = metrics.get('Total Annualized Returns', 0)
    alpha = metrics.get('Alpha', 0)
    excess_return = metrics.get('超额收益', 0)
    beta = metrics.get('Beta', 0)
    sharpe = metrics.get('Sharpe', 0)
    sortino = metrics.get('Sortino', 0)
    max_drawdown = metrics.get('Max Drawdown', 0)
    algo_vol = metrics.get('Algorithm Volatility', 0)
    bench_vol = metrics.get('Benchmark Volatility', 0)
    win_rate = metrics.get('胜率', 0)
    pl_ratio = metrics.get('盈亏比', 0)
    daily_win_rate = metrics.get('日胜率', 0)
    
    # 生成评价文本
    alpha_desc = "相对基准跑赢" if alpha > 0 else "相对基准跑输"
    excess_desc = "相对基准跑赢" if excess_return > 0 else "相对基准跑输"
    beta_desc = "策略波动性接近市场" if abs(beta - 1.0) < 0.2 else ("策略波动性高于市场" if beta > 1.0 else "策略波动性低于市场")
    sortino_desc = "下行风险控制优于总风险控制" if sortino > sharpe else "下行风险控制一般"
    volatility_desc = "策略波动略高于市场" if algo_vol > bench_vol else "策略波动低于市场"
    win_rate_desc = "择时准确率" + ("较高" if win_rate > 50 else "偏低" if win_rate < 30 else "中等")
    pl_ratio_desc = "盈亏结构优秀，亏损小额，盈利大幅" if pl_ratio > 2.0 else "盈亏结构一般"
    daily_win_rate_desc = "日度表现优于基准" if daily_win_rate > 50 else "日度表现略低于基准"
    
    st.markdown(f"""
        - **总收益**: {total_returns:.2f}% | **年化收益**: {annualized_returns:.2f}% 
        - **Alpha**: {alpha:.2f}% ({alpha_desc}) | **超额收益**: {excess_return:.2f}% ({excess_desc})
        - **Beta**: {beta:.2f} ({beta_desc})

        **风险调整后收益**
        - **夏普比率**: {sharpe:.2f} | **索提诺比率**: {sortino:.2f} ({sortino_desc})
        - **最大回撤**: {max_drawdown:.2f}% (风险控制能力{'较强' if abs(max_drawdown) < 10 else '一般'})
        - **波动率**: {algo_vol:.2f}% vs 基准 {bench_vol:.2f}% ({volatility_desc})

        **交易特征**
        - **胜率**: {win_rate:.2f}% ({win_rate_desc})
        - **盈亏比**: {pl_ratio:.2f} ({pl_ratio_desc})
        - **日胜率**: {daily_win_rate:.2f}% ({daily_win_rate_desc})
            """)
    
    # 交易信号图（主要图表）
    st.divider()
    st.header("📈 交易信号图")
    fig = plot_trading_signals(df)
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("查看详细交易数据"):
        display_cols = ['日期', '委托时间', '标的', '交易类型', '成交数量']
        display_cols.append('成交价')
        display_cols.append('成交额')
        display_cols.extend(['平仓盈亏', '手续费', '净盈亏', '累计收益'])
        available_cols = [col for col in display_cols if col in df.columns]
        st.dataframe(df[available_cols], use_container_width=True)
    with st.expander("查看日度汇总数据"):
        st.dataframe(daily_pnl, use_container_width=True)

if __name__ == "__main__":
    main()
