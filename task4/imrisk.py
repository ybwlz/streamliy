import akshare as ak
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# 页面配置
st.set_page_config(
    page_title="IM期货策略风险分析",
    page_icon="📊",
    layout="wide"
)

Rf = 0.04  # 无风险利率
TRADING_DAYS_PER_YEAR = 250  # 每年交易日数


#缓存基准数据
@st.cache_data
def get_benchmark_data(daily_pnl):
    """
    获取沪深300指数基准数据并计算收益率
    
    参数:
    --------
    daily_pnl: DataFrame，包含策略的日期和日盈亏数据
    
    返回:
    --------
    dict: 包含基准日收益率、总收益率、年化收益率的字典
    """
    start_date = daily_pnl['日期'].min().strftime('%Y%m%d')#strftime('%Y%m%d') 将日期转换为字符串，格式为YYYYMMDD
    end_date = daily_pnl['日期'].max().strftime('%Y%m%d')
    
    # 获取更早的数据，以便计算第一个交易日的收益率
    start_date_dt = pd.to_datetime(start_date, format='%Y%m%d')
    start_date_early = (start_date_dt - pd.Timedelta(days=30)).strftime('%Y%m%d')
    
    # 获取沪深300指数数据（指数代码：000300）
    try:
        benchmark_df = ak.index_zh_a_hist(symbol="000300", period="daily", start_date=start_date_early, end_date=end_date)
        if benchmark_df is None or benchmark_df.empty:
            raise ValueError("指数数据为空")
    except Exception as e:
        print(f"获取沪深300指数数据失败: {e}")
        return {'benchmark_daily_returns': None, 'benchmark_total_return': None, 'benchmark_annualized_return': None}
        
    # 处理日期列
    if '日期' in benchmark_df.columns:
        benchmark_df['date'] = pd.to_datetime(benchmark_df['日期'], errors='coerce')
    elif 'date' in benchmark_df.columns:
        benchmark_df['date'] = pd.to_datetime(benchmark_df['date'], errors='coerce')
    else:
        benchmark_df['date'] = pd.to_datetime(benchmark_df.iloc[:, 0], errors='coerce')
    
    benchmark_df = benchmark_df.sort_values('date').reset_index(drop=True)
    
    # 获取收盘价
    if '收盘' in benchmark_df.columns:
        prices = pd.to_numeric(benchmark_df['收盘'], errors='coerce').values
    elif '收盘价' in benchmark_df.columns:
        prices = pd.to_numeric(benchmark_df['收盘价'], errors='coerce').values
    elif 'close' in benchmark_df.columns:
        prices = pd.to_numeric(benchmark_df['close'], errors='coerce').values
    else:
        #防御性编程，遍历各种可能的列名
        price_col = None
        for col in benchmark_df.columns:
            #'收盘' in str(col) → 包含中文"收盘"
            if '收盘' in str(col) or 'close' in str(col).lower():#str(col).lower() 将字符串转换为小写
                price_col = col
                break
        prices = pd.to_numeric(benchmark_df[price_col], errors='coerce').values if price_col else None
    
    if prices is None or len(prices) < 2:
        return {'benchmark_daily_returns': None, 'benchmark_total_return': None, 'benchmark_annualized_return': None}
    
    # 构建基准数据DataFrame（包含所有获取到的数据，用于计算收益率）
    benchmark_price_df = pd.DataFrame({
        '日期': pd.to_datetime(benchmark_df['date']).dt.normalize().values,
        '收盘价': prices
    }).drop_duplicates(subset=['日期']).sort_values('日期').reset_index(drop=True)#drop_duplicates(subset=['日期']) 
    
    # 先计算基准数据的日收益率（基于完整数据，确保第一个交易日有前一日数据）
    benchmark_price_df['基准日收益率'] = benchmark_price_df['收盘价'].pct_change()#pct_change() 计算收益率，相当于每日收益/前一日收盘价
    
    # 筛选到策略日期范围并对齐
    start_date_dt = pd.to_datetime(start_date, format='%Y%m%d')
    end_date_dt = pd.to_datetime(end_date, format='%Y%m%d')
    benchmark_price_df_filtered = benchmark_price_df[
        (benchmark_price_df['日期'] >= start_date_dt) & 
        (benchmark_price_df['日期'] <= end_date_dt)
    ].copy()
    
    trade_df = pd.DataFrame({
        '日期': pd.to_datetime(daily_pnl['日期']).dt.normalize()
    })
    #merge(benchmark_price_df_filtered[['日期', '基准日收益率']], on='日期', how='left') 将交易数据和基准数据合并，on='日期' 按日期合并，how='left' 左连接
    merged = trade_df.merge(benchmark_price_df_filtered[['日期', '基准日收益率']], on='日期', how='left')
    
    merged['基准日收益率'] = merged['基准日收益率'].fillna(0.0)
    
    benchmark_daily_returns = merged['基准日收益率'].values
    
    # 计算基准总收益和年化收益（复利计算）
    valid_benchmark = benchmark_daily_returns[~np.isnan(benchmark_daily_returns)]
    if len(valid_benchmark) > 0 and valid_benchmark[0] == 0:
        valid_benchmark = valid_benchmark[1:]  # 排除第一个0收益率
    
    if len(valid_benchmark) > 0:
        benchmark_total_return = np.prod(1 + valid_benchmark) - 1
        benchmark_trading_days = len(valid_benchmark)
        years_benchmark = benchmark_trading_days / TRADING_DAYS_PER_YEAR
        benchmark_annualized_return = ((1 + benchmark_total_return) ** (1 / years_benchmark)) - 1 if years_benchmark > 0 else 0.0
    else:
        benchmark_total_return = 0.0
        benchmark_annualized_return = 0.0
    print(benchmark_df.head(5))
    return {
        'benchmark_daily_returns': benchmark_daily_returns,
        'benchmark_total_return': benchmark_total_return,
        'benchmark_annualized_return': benchmark_annualized_return
    }


@st.cache_data
def load_and_process_data(transaction_file="transaction.csv", benchmark_symbol="IF0"):
    """
    加载并处理所有数据
    
    返回:
    --------
    dict: 包含策略和基准数据的字典
    """
    # 1.加载交易数据
    import os
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))#获取当前脚本的绝对路径，os.path.dirname()获取其父目录
    # 拼接完整路径
    csv_path = os.path.join(script_dir,transaction_file)
    # 读取CSV文件
    df = pd.read_csv(csv_path, encoding='gbk')
    
    # 2. 预处理
    df['日期时间'] = pd.to_datetime(df['日期'] + ' ' + df['委托时间'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    df = df.sort_values('日期时间').reset_index(drop=True)
    df['成交数量'] = pd.to_numeric(df['成交数量'].astype(str).str.replace('手', ''), errors='coerce')
    df['成交价'] = pd.to_numeric(df['成交价'], errors='coerce')
    df['成交额'] = pd.to_numeric(df['成交额'], errors='coerce')
    df['平仓盈亏'] = pd.to_numeric(df['平仓盈亏'], errors='coerce').fillna(0)#fillna(0) 将缺失值填充为0
    df['手续费'] = pd.to_numeric(df['手续费'], errors='coerce').fillna(0)
    df['净盈亏'] = df['平仓盈亏'] - df['手续费']
    
    # 3. 初始资金
    initial_capital = 1000000
    
    # 4. 按日期聚合,pnl是profit and loss的缩写，表示盈利和亏损
    df['日期_仅'] = pd.to_datetime(df['日期'], format='%Y-%m-%d', errors='coerce')
    daily_pnl = df.groupby('日期_仅')['净盈亏'].sum().reset_index()#groupby('日期_仅') 按日期聚合，['净盈亏'].sum() 求和，reset_index() 将索引转换为列
    daily_pnl.columns = ['日期', '日盈亏']#columns = ['日期', '日盈亏'] 将列名设置为日期和日盈亏
    daily_pnl = daily_pnl.sort_values('日期').reset_index(drop=True)
    daily_pnl['累计收益'] = daily_pnl['日盈亏'].cumsum()#cumsum() 计算累计和
    
    # 5. 计算策略日收益率（用前一日权益做分母）ll
    equity_prev = initial_capital + daily_pnl['累计收益'].shift(1).fillna(0.0)#shift(1) 将数据向下移动1行，fillna(0.0) 将缺失值填充为0.0
    equity_prev = equity_prev.replace(0, np.nan)
    daily_returns = daily_pnl['日盈亏'] / equity_prev
    if np.isnan(daily_returns.iloc[0]):
        daily_returns.iloc[0] = daily_pnl['日盈亏'].iloc[0] / initial_capital
    daily_returns = daily_returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    strategy_daily_returns = daily_returns.values
    
    # 6. 策略总收益率和年化收益率
    # Total Returns=(Pend−Pstart)/Pstart∗100%
    # Pstart=策略开始股票和现金的总价值（初始资金）
    # Pend=策略最终股票和现金的总价值（初始资金+累计收益）
    Pstart = initial_capital  # 初始权益
    Pend = initial_capital + daily_pnl['累计收益'].iloc[-1] if len(daily_pnl) > 0 else initial_capital  # 最终权益
    strategy_total_return = (Pend - Pstart) / Pstart if Pstart > 0 else 0.0  # 总收益率（小数形式）
    
    trading_days = len(daily_pnl)
    date_range = (daily_pnl['日期'].max() - daily_pnl['日期'].min()).days
    years = date_range / 365.25 if date_range > 0 else trading_days / TRADING_DAYS_PER_YEAR
    # Total Annualized Returns=Rp=((1+P)的250/n次方−1)∗100%
    # P=策略收益（小数形式），n=策略执行天数
    P = strategy_total_return  # 策略收益（小数形式）
    n = trading_days  # 策略执行天数
    strategy_annualized_return = ((1 + P) ** (TRADING_DAYS_PER_YEAR / n) - 1) if n > 0 else 0.0
    
    # 7. 交易统计（平仓维度）- 盈亏比=总盈利额/总亏损额（用平仓盈亏，不含手续费）
    close_mask = df['交易类型'].astype(str).str.contains('平')#str.contains('平') 判断交易类型是否包含平
    df_close = df[close_mask].copy()#df[close_mask].copy() 复制一份数据，用于计算盈亏比
    if not df_close.empty:
        # 盈亏比用平仓盈亏计算（不含手续费）
        win_mask = df_close['平仓盈亏'] > 0#win_mask 盈利交易
        loss_mask = (df_close['平仓盈亏'] < 0) | (df_close['平仓盈亏'] == 0)  # 盈亏为0也算亏损（与聚宽一致）
        win_trades = win_mask.sum()#盈利交易次数
        loss_trades = loss_mask.sum()#亏损交易次数
        total_profit = df_close[win_mask]['平仓盈亏'].sum()  # 总盈利额（平仓盈亏，正数）
        total_loss = abs(df_close[df_close['平仓盈亏'] < 0]['平仓盈亏'].sum())  # 总亏损额（平仓盈亏，绝对值，不包括盈亏为0的）
    else:
        win_trades = loss_trades = 0
        total_profit = total_loss = 0.0
    
    # 8. 获取基准数据（沪深300指数 000300）
    # 根据聚宽策略代码：set_benchmark('000300.XSHG')，基准是沪深300指数
    benchmark_result = get_benchmark_data(daily_pnl)
    benchmark_daily_returns = benchmark_result['benchmark_daily_returns']#基准每日收益
    benchmark_total_return = benchmark_result['benchmark_total_return']#基准总收益率
    benchmark_annualized_return = benchmark_result['benchmark_annualized_return']#基准年化收益率
    return {
        'df': df,  # 原始交易数据（用于绘图）
        'strategy_daily_returns': strategy_daily_returns,
        'strategy_total_return': strategy_total_return,
        'strategy_annualized_return': strategy_annualized_return,
        'Pstart': Pstart,  # 初始权益
        'Pend': Pend,  # 最终权益
        'P': P,  # 策略收益（小数形式）
        'n': n,  # 策略执行天数
        'initial_capital': initial_capital,#初始资金
        'daily_pnl': daily_pnl,#每日盈亏
        'trading_days': trading_days,#交易天数
        'win_trades': win_trades,#盈利交易次数
        'loss_trades': loss_trades,#亏损交易次数
        'total_profit': total_profit,#总盈利额
        'total_loss': total_loss,#总亏损额
        'benchmark_daily_returns': benchmark_daily_returns,#基准每日收益
        'benchmark_total_return': benchmark_total_return,#基准总收益率
        'benchmark_annualized_return': benchmark_annualized_return,#基准年化收益率
    }


#风险指标
#Total Returns 策略收益 - Total Returns=(Pend−Pstart)/Pstart∗100%
"""

"""
def get_total_returns(Pend, Pstart):
    
    if Pstart == 0:
        return 0.0
    return ((Pend - Pstart) / Pstart) * 100

#Total Annualized Returns 策略年化收益 - Rp=((1+P)的250/n次方−1)∗100%
def get_total_annualized_returns(P, n):
    
    if n == 0:
        return 0.0
    Rp = ((1 + P) ** (TRADING_DAYS_PER_YEAR / n) - 1) * 100
    return Rp

#Beta 贝塔 - βp=Cov(Dp,Dm)/Var(Dm)
def get_beta(Dp, Dm):
    #Cov(Dp,Dm) 策略每日收益与基准每日收益的协方差，Var(Dm) 基准每日收益的方差
    if Dm is None or len(Dp) < 2 or len(Dm) < 2:#如果基准每日收益为空或策略每日收益小于2或基准每日收益小于2，则贝塔为0
        return 0.0
    
    n = min(len(Dp), len(Dm))#n 策略每日收益和基准每日收益的最小长度
    Dp = Dp[:n]
    Dm = Dm[:n]
    
    # 过滤掉 NaN 和 inf 值
    valid_mask = ~(np.isnan(Dp) | np.isnan(Dm) | np.isinf(Dp) | np.isinf(Dm))
    Dp_clean = Dp[valid_mask]
    Dm_clean = Dm[valid_mask]
    
    if len(Dp_clean) < 2 or len(Dm_clean) < 2:
        return 0.0
    
    # 统一使用样本统计量（ddof=1），这是金融中的标准做法
    Cov = np.cov(Dp_clean, Dm_clean, ddof=1)[0, 1]#np.cov()[0, 1] 获取协方差，使用样本统计量
    Var_Dm = np.var(Dm_clean, ddof=1)#np.var() 计算方差，使用样本统计量，与协方差保持一致
    
    return Cov / Var_Dm if Var_Dm > 1e-10 else 0.0

#Alpha 阿尔法 - α=Rp-[Rf+βp(Rm-Rf)]
def get_alpha(Rp, Rm, βp):
    if Rm is None:
        return 0.0
    α = Rp - (Rf + βp * (Rm - Rf))
    return α * 100

#Sharpe 夏普比率 - Sharpe=(Rp-Rf)/σp
def get_sharpe(Rp, σp):
    return (Rp - Rf) / σp if σp > 1e-10 else 0.0

#Sortino 索提诺比率 - Sortino=(Rp-Rf)/σpd
def get_sortino(Rp, σpd):
    return (Rp - Rf) / σpd if σpd > 1e-10 else 0.0

#Information Ratio 信息比率 - IR=(Rp-Rm)/σt  σt=策略与基准每日收益差值的年化标准差
def get_information_ratio(Rp, Rm, Dp, Dm):
    #如果基准每日收益为空或策略每日收益小于2或基准每日收益小于2，则信息比率为0
    if Dm is None or Rm is None or len(Dp) < 2 or len(Dm) < 2:
        return 0.0
    
    n = min(len(Dp), len(Dm))#n 策略每日收益和基准每日收益的最小长度
    Dp_aligned = Dp[:n]
    Dm_aligned = Dm[:n]
    
    # 过滤掉 NaN 和 inf 值
    valid_mask = ~(np.isnan(Dp_aligned) | np.isnan(Dm_aligned) | 
                   np.isinf(Dp_aligned) | np.isinf(Dm_aligned))
    Dp_clean = Dp_aligned[valid_mask]
    Dm_clean = Dm_aligned[valid_mask]
    
    if len(Dp_clean) < 2 or len(Dm_clean) < 2:
        return 0.0
    
    excess = Dp_clean - Dm_clean  # excess 策略每日收益与基准每日收益的差值
    # 使用样本标准差（ddof=1），与策略波动率保持一致
    σt = np.std(excess, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
    
    return (Rp - Rm) / σt if σt > 1e-10 else 0.0


#Algorithm Volatility 策略波动率 - σp=sqrt(250/(n-1))*sqrt(Σ(rp-rp_mean)^2)
def get_algorithm_volatility(Dp):
    if len(Dp) < 2:
        return 0.0
    σp = np.std(Dp, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR) * 100#np.std() 计算标准差,ddof=1 表示分母为n-1
    return σp


#Benchmark Volatility 基准波动率 - σm=sqrt(250/(n-1))*sqrt(Σ(rm-rm_mean)^2)
def get_benchmark_volatility(Dm):
    if Dm is None or len(Dm) < 2:
        return 0.0
    
    Dm_valid = Dm[Dm != 0]  # 排除第一个交易日
    
    if len(Dm_valid) < 2:
        return 0.0
    
    σm = np.std(Dm_valid, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR) * 100
    return σm


#Max Drawdown 最大回撤
def get_max_drawdown(Dp):
    """Max Drawdown=Max((Px−Py)/Px),Px,Py=策略某日股票和现金的总价值，y>x"""
    if len(Dp) == 0:
        return 0.0
    
    # 计算净值序列（总价值相对于初始值的倍数）
    # net_value[i] = 初始资金 * (1 + Dp[0]) * (1 + Dp[1]) * ... * (1 + Dp[i])
    net_value = np.cumprod(1 + Dp)  # 净值，例如：[1.0, 1.05, 1.08, 1.02, ...]
    
    # 计算运行最大值（历史最高净值）
    running_max = np.maximum.accumulate(net_value)
    
    # 计算回撤：(历史最高净值 - 当前净值) / 历史最高净值
    # 公式：Max((Px−Py)/Px)，其中 Px 是历史最高净值，Py 是当前净值
    drawdown = (running_max - net_value) / running_max
    
    # 返回最大回撤（正数，表示回撤幅度）
    return np.max(drawdown) * 100


#Downside Risk 下行波动率 - σpd=根号下(250/n)∑(rp−△rpi均值)2f(t)
def get_downside_risk(Dp):
    """
    计算下行波动率（年化）- 最终优化版
    
    参数:
    --------
    Dp: array-like，策略日收益率序列
    
    返回:
    --------
    float: 下行波动率（百分比形式）
    
    公式:
    --------
    1. 计算整体平均收益率 mean_return = mean(Dp)
    2. 筛选低于均值的日收益 downside_returns = Dp[Dp < mean_return]
    3. 计算总体标准差并年化 σpd = std(downside_returns, ddof=0) × sqrt(250) × 100
    
    关键点：
    - 基准：整体均值（不是累积均值）
    - 标准差：总体标准差 ddof=0（不是样本标准差 ddof=1）
    - 年化：标准 sqrt(250)
    
    注意：此方法与聚宽平台高度匹配
    """
    if len(Dp) < 2:
        return 0.0
    
    # 计算整体平均收益率
    mean_return = np.mean(Dp)
    
    # 筛选低于均值的日收益
    downside_returns = Dp[Dp < mean_return]
    
    # 如果下行数据不足，返回0
    if len(downside_returns) < 2:
        return 0.0
    
    # 计算总体标准差并年化（转为百分比）
    # 使用 ddof=0（总体标准差）而非 ddof=1（样本标准差）
    σpd = np.std(downside_returns, ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR) * 100
    
    return σpd

#胜率 =盈利交易次数/总交易次数  #赚了多少次
def get_win_rate(win_trades, loss_trades):
    total_trades = win_trades + loss_trades
    return (win_trades / total_trades * 100) if total_trades > 0 else 0.0


def get_daily_win_rate(Dp, Dm):
    """日胜率=当日策略收益跑赢当日基准收益的天数/总交易日数"""
    if Dm is None or len(Dp) == 0 or len(Dm) == 0:
        return 0.0
    
    n = min(len(Dp), len(Dm))
    winning_days = (Dp[:n] > Dm[:n]).sum()
    return (winning_days / n * 100) if n > 0 else 0.0


def get_profit_loss_ratio(total_profit, total_loss):
    #盈亏比=总盈利额/总亏损额
    return (total_profit / total_loss) if total_loss > 0 else 0.0


def get_aei(Dp, Dm):
    """AEI=∑(i从1到n)(△EIi−△EI(i−1))/n, EI=(策略收益+100%)/(基准收益+100%)−100%"""
    if Dm is None or len(Dp) == 0 or len(Dm) == 0:
        return 0.0
    
    n = min(len(Dp), len(Dm))
    sp_cum = np.cumprod(1 + Dp[:n]) - 1
    bm_cum = np.cumprod(1 + Dm[:n]) - 1
    
    EI = (1 + sp_cum) / (1 + bm_cum) - 1
    EI_changes = np.diff(EI, prepend=0)
    
    return np.mean(EI_changes) * 100


def get_excess_return_max_drawdown(Dp, Dm):
    """
    超额收益最大回撤=Max((EIx−EIy)/EIx), 其中y>x
    EI=(策略收益+100%)/(基准收益+100%)−100%
    注意：使用净值方式计算，EI净值 = 1 + EI，然后计算净值回撤
    """
    if Dm is None or len(Dp) == 0 or len(Dm) == 0:
        return 0.0
    
    n = min(len(Dp), len(Dm))
    sp_cum = np.cumprod(1 + Dp[:n])  # 策略累计净值
    bm_cum = np.cumprod(1 + Dm[:n])  # 基准累计净值
    
    # EI=(策略收益+100%)/(基准收益+100%)−100% = sp_cum/bm_cum - 1
    EI = sp_cum / bm_cum - 1
    
    # 计算Max((EIx−EIy)/EIx)，其中y>x
    # 使用净值方式：EI净值 = 1 + EI
    EI_net_value = 1 + EI  # EI净值
    
    # 类似最大回撤的计算：找到运行最大值，然后计算回撤
    EI_running_max = np.maximum.accumulate(EI_net_value)  # EI净值的运行最大值
    EI_drawdown = (EI_running_max - EI_net_value) / EI_running_max  # 回撤比例
    
    # 返回最大回撤
    max_drawdown = np.max(EI_drawdown) if len(EI_drawdown) > 0 else 0.0
    
    return max_drawdown * 100


def get_excess_return_sharpe(Dp, Dm):
    """
    超额收益夏普比率=(RpEI-Rf)/σpEI
    RpEI=年化超额收益率（基于EI净值计算年化收益率）
    σpEI=超额收益波动率（基于EI净值日收益率的标准差，年化）
    """
    if Dm is None or len(Dp) < 2 or len(Dm) < 2:
        return 0.0
    
    n = min(len(Dp), len(Dm))
    Dp_aligned = Dp[:n]
    Dm_aligned = Dm[:n]
    
    # 只过滤掉 inf 值，保留 NaN（让 numpy 处理）
    valid_mask = ~(np.isinf(Dp_aligned) | np.isinf(Dm_aligned))
    Dp_clean = Dp_aligned[valid_mask]
    Dm_clean = Dm_aligned[valid_mask]
    
    if len(Dp_clean) < 2 or len(Dm_clean) < 2:
        return 0.0
    
    # 计算EI日收益率
    # EI = (策略收益+100%)/(基准收益+100%) - 100% = sp_cum/bm_cum - 1
    # EI净值 = 1 + EI = sp_cum/bm_cum
    # EI日收益率 = EI净值[i] / EI净值[i-1] - 1 = (sp_cum[i]/bm_cum[i]) / (sp_cum[i-1]/bm_cum[i-1]) - 1
    # = (sp_cum[i]/sp_cum[i-1]) / (bm_cum[i]/bm_cum[i-1]) - 1
    # = (1 + Dp[i]) / (1 + Dm[i]) - 1
    EI_daily_returns = (1 + Dp_clean) / (1 + Dm_clean) - 1  # EI日收益率
    
    # 过滤掉 NaN 和 inf（在计算标准差时）
    EI_daily_returns_valid = EI_daily_returns[~(np.isnan(EI_daily_returns) | np.isinf(EI_daily_returns))]
    
    n_clean = len(EI_daily_returns_valid)
    if n_clean < 2:
        return 0.0
    
    # 计算RpEI：基于EI日收益率的年化收益率（复利方式）
    # 使用所有有效数据计算总收益率
    EI_total_return = np.prod(1 + EI_daily_returns_valid) - 1  # EI的总收益率
    RpEI = ((1 + EI_total_return) ** (TRADING_DAYS_PER_YEAR / n_clean) - 1) if n_clean > 0 else 0.0
    
    # 计算EI日收益率的标准差（年化），使用样本标准差（ddof=1），与策略波动率保持一致
    σpEI = np.std(EI_daily_returns_valid, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
    
    return (RpEI - Rf) / σpEI if σpEI > 1e-10 else 0.0 


def calculate_all_metrics(data):
    """计算所有风险指标（按need.txt变量命名）"""
    # 提取变量（按need.txt命名）
    Rp = data['strategy_annualized_return']  # 策略年化收益率
    Rm = data['benchmark_annualized_return']  # 基准年化收益率
    Dp = data['strategy_daily_returns']  # 策略每日收益
    Dm = data['benchmark_daily_returns']  # 基准每日收益
    
    # 计算中间变量
    σp = get_algorithm_volatility(Dp) / 100  # 策略波动率（小数形式）
    σm = get_benchmark_volatility(Dm) / 100  # 基准波动率（小数形式）
    σpd = get_downside_risk(Dp) / 100  # 下行波动率（小数形式）
    βp = get_beta(Dp, Dm)  # Beta
    
    # 计算超额收益（除法版本，与聚宽一致）
    # 超额收益 = (策略收益+100%)/(基准收益+100%) - 100% = (1+策略收益)/(1+基准收益) - 1
    strategy_total_return = data['strategy_total_return']  # 小数形式
    strategy_total_return_pct = get_total_returns(data['Pend'], data['Pstart'])  # 百分比形式
    if data['benchmark_total_return'] is not None:
        benchmark_total_return = data['benchmark_total_return']  # 小数形式
        if abs(1 + benchmark_total_return) > 1e-10:
            excess_return = ((1 + strategy_total_return) / (1 + benchmark_total_return) - 1) * 100
        else:
            excess_return = (strategy_total_return - benchmark_total_return) * 100
    else:
        excess_return = 0.0
    
    metrics = {
        'total_returns': strategy_total_return_pct,
        'total_annualized_returns': get_total_annualized_returns(data['P'], data['n']),
        'alpha': get_alpha(Rp, Rm, βp),
        'beta': βp,
        'sharpe_ratio': get_sharpe(Rp, σp),
        'sortino_ratio': get_sortino(Rp, σpd),
        'information_ratio': get_information_ratio(Rp, Rm, Dp, Dm),
        'strategy_volatility': get_algorithm_volatility(Dp),
        'benchmark_volatility': get_benchmark_volatility(Dm),
        'max_drawdown': get_max_drawdown(Dp),
        'downside_risk': get_downside_risk(Dp),
        'win_rate': get_win_rate(data['win_trades'], data['loss_trades']),
        'daily_win_rate': get_daily_win_rate(Dp, Dm),
        'profit_loss_ratio': get_profit_loss_ratio(data['total_profit'], data['total_loss']),
        'aei': get_aei(Dp, Dm),
        'excess_max_drawdown': get_excess_return_max_drawdown(Dp, Dm),
        'excess_sharpe_ratio': get_excess_return_sharpe(Dp, Dm),
        'excess_return': excess_return,
        'win_trades': data['win_trades'],
        'loss_trades': data['loss_trades'],
    }
    
    return metrics

# ==================计算阶段使用的计算指标，目前没有使用==================
def print_metrics(metrics, data=None):
    #打印所有风险指标
    print("策略收益")
    print(f"{metrics['Total Returns']:.2f}%")
    print("策略年化收益")
    print(f"{metrics['Total Annualized Returns']:.2f}%")
    print("超额收益")
    print(f"{metrics.get('超额收益', 0):.2f}%")
    print("基准收益")
    if data and 'benchmark_total_return' in data and data['benchmark_total_return'] is not None:
        benchmark_pct = data['benchmark_total_return'] * 100
        print(f"{benchmark_pct:.2f}%")
    else:
        print("0.00%")
    print("阿尔法")
    print(f"{metrics['alpha'] / 100:.3f}")
    print("贝塔")
    print(f"{metrics['beta']:.3f}")
    print("夏普比率")
    print(f"{metrics['sharpe_ratio']:.3f}")
    print("胜率")
    print(f"{metrics['win_rate'] / 100:.3f}")
    print("盈亏比")
    print(f"{metrics['profit_loss_ratio']:.3f}")
    print("最大回撤")
    print(f"{metrics['max_drawdown']:.2f}%")
    print("索提诺比率")
    print(f"{metrics['sortino_ratio']:.3f}")
    print("日均超额收益")
    print(f"{metrics['aei']:.2f}%")
    print("超额收益最大回撤")
    print(f"{metrics['excess_max_drawdown']:.2f}%")
    print("超额收益夏普比率")
    print(f"{metrics['excess_sharpe_ratio']:.3f}")
    print("日胜率")
    print(f"{metrics['daily_win_rate'] / 100:.3f}")
    print("盈利次数")
    print(f"{metrics['win_trades']}")
    print("亏损次数")
    print(f"{metrics['loss_trades']}")
    print("信息比率")
    print(f"{metrics['information_ratio']:.3f}")
    print("策略波动率")
    print(f"{metrics['strategy_volatility'] / 100:.3f}")
    print("基准波动率")
    print(f"{metrics['benchmark_volatility'] / 100:.3f}")


def main():
    #主函数
    print("="*60)#分割线
    print("🚀 IM期货策略风险指标计算")
    print("="*60)
    
    # 加载数据
    data = load_and_process_data(transaction_file="transaction.csv", benchmark_symbol="IF0")
    
    print("\n" + "="*60)
    print("📊 数据加载完成")
    print("="*60)
    print(f"策略交易日数: {data['trading_days']}")
    print(f"策略总收益率: {data['strategy_total_return']*100:.4f}%")
    print(f"策略年化收益率: {data['strategy_annualized_return']*100:.4f}%")
    if data['benchmark_daily_returns'] is not None:
        print(f"基准总收益率: {data['benchmark_total_return']*100:.4f}%")
        print(f"基准年化收益率: {data['benchmark_annualized_return']*100:.4f}%")
    print("="*60)
    
    # 计算所有指标
    metrics = calculate_all_metrics(data)
    
    # 打印结果
    print_metrics(metrics, data)
    
    print("✅ 所有风险指标计算完成！")


# ================== Streamlit可视化 ==================
def plot_signal_chart(data, df_raw):
    """
    绘制交易信号图 - 收盘价曲线 + 交易信号标注（主图，放大显示）
    """
    # 获取每日收盘价（连续曲线）
    df_daily_price = df_raw.groupby('日期').agg({
        '成交价': 'last',
    }).reset_index()
    df_daily_price['日期'] = pd.to_datetime(df_daily_price['日期'])
    
    # 创建图表
    fig = go.Figure()
    
    # 1. 先画收盘价线（连续）
    fig.add_trace(
        go.Scatter(
            x=df_daily_price['日期'],
            y=df_daily_price['成交价'],
            mode='lines',
            name='收盘价',
            line=dict(color='#1f77b4', width=2),
            hovertemplate='收盘价: %{y:.2f}<extra></extra>'
        )
    )
    
    # 2. 在收盘价线上标注交易信号
    df_raw_copy = df_raw.copy()
    df_raw_copy['日期'] = pd.to_datetime(df_raw_copy['日期'])
    
    # 开多信号（绿色向上三角形）
    open_long = df_raw_copy[df_raw_copy['交易类型'].str.contains('开多', na=False)]
    if len(open_long) > 0:
        fig.add_trace(
            go.Scatter(
                x=open_long['日期'],
                y=open_long['成交价'],
                mode='markers',
                name='开多',
                marker=dict(symbol='triangle-up', size=7, color='green', line=dict(width=1, color='darkgreen')),
                hovertemplate='开多<br>日期: %{x}<br>价格: %{y:.2f}<extra></extra>'
            )
        )
    
    # 开空信号（红色向下三角形）
    open_short = df_raw_copy[df_raw_copy['交易类型'].str.contains('开空', na=False)]
    if len(open_short) > 0:
        fig.add_trace(
            go.Scatter(
                x=open_short['日期'],
                y=open_short['成交价'],
                mode='markers',
                name='开空',
                marker=dict(symbol='triangle-down', size=7, color='red', line=dict(width=1, color='darkred')),
                hovertemplate='开空<br>日期: %{x}<br>价格: %{y:.2f}<extra></extra>'
            )
        )
    
    # 平多信号（红色向下三角形）
    close_long = df_raw_copy[df_raw_copy['交易类型'].str.contains('平多', na=False)]
    if len(close_long) > 0:
        fig.add_trace(
            go.Scatter(
                x=close_long['日期'],
                y=close_long['成交价'],
                mode='markers',
                name='平多',
                marker=dict(symbol='triangle-down', size=7, color='red', line=dict(width=1, color='darkred')),
                hovertemplate='平多<br>日期: %{x}<br>价格: %{y:.2f}<br>盈亏: %{text}<extra></extra>',
                text=close_long['平仓盈亏'].apply(lambda x: f'{x:.2f}')
            )
        )
    
    # 平空信号（绿色向上三角形）
    close_short = df_raw_copy[df_raw_copy['交易类型'].str.contains('平空', na=False)]
    if len(close_short) > 0:
        fig.add_trace(
            go.Scatter(
                x=close_short['日期'],
                y=close_short['成交价'],
                mode='markers',
                name='平空',
                marker=dict(symbol='triangle-up', size=7, color='green', line=dict(width=1, color='darkgreen')),
                hovertemplate='平空<br>日期: %{x}<br>价格: %{y:.2f}<br>盈亏: %{text}<extra></extra>',
                text=close_short['平仓盈亏'].apply(lambda x: f'{x:.2f}')
            )
        )
    
    # 更新布局（网格背景，主图放大）
    fig.update_layout(
        height=700,
        showlegend=True,
        hovermode='x unified',
        xaxis=dict(
            title="日期",
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            tickformat='%Y-%m-%d'
        ),
        yaxis=dict(
            title="收盘价",
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12)
    )
    
    return fig


def plot_returns_chart(data, use_log_scale=False, show_strategy=True, show_benchmark=True, show_excess=False):
    """
    绘制收益曲线图（支持对数轴、多曲线选择）+ 日盈亏柱状图（双子图）
    """
    daily_pnl = data['daily_pnl']
    Dp = data['strategy_daily_returns']
    Dm = data['benchmark_daily_returns']
    
    # 计算策略净值和最大回撤
    net_value = data['initial_capital'] + daily_pnl['累计收益']
    strategy_nav = net_value / data['initial_capital']
    
    # 策略最大回撤
    strategy_running_max = np.maximum.accumulate(strategy_nav)
    strategy_drawdown = (strategy_running_max - strategy_nav) / strategy_running_max * 100
    strategy_max_dd_idx = strategy_drawdown.argmax()
    strategy_max_dd_date = daily_pnl['日期'].iloc[strategy_max_dd_idx]
    
    # 基准最大回撤
    benchmark_max_dd_idx = None
    benchmark_max_dd_date = None
    if Dm is not None and len(Dm) > 0:
        benchmark_nav = np.cumprod(1 + Dm)
        benchmark_running_max = np.maximum.accumulate(benchmark_nav)
        benchmark_drawdown = (benchmark_running_max - benchmark_nav) / benchmark_running_max * 100
        benchmark_max_dd_idx = benchmark_drawdown.argmax()
        benchmark_max_dd_date = daily_pnl['日期'].iloc[benchmark_max_dd_idx]
    
    # 超额收益最大回撤
    excess_max_dd_idx = None
    excess_max_dd_date = None
    if Dm is not None and len(Dm) > 0:
        excess_nav = strategy_nav / np.cumprod(1 + Dm)
        excess_running_max = np.maximum.accumulate(excess_nav)
        excess_drawdown = (excess_running_max - excess_nav) / excess_running_max * 100
        excess_max_dd_idx = excess_drawdown.argmax()
        excess_max_dd_date = daily_pnl['日期'].iloc[excess_max_dd_idx]
    
    # 创建双子图：上：收益曲线，下：日盈亏柱状图（分开显示，共享X轴）
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,  # 增加间距，让两个图分开
        row_heights=[0.6, 0.4],  # 日盈亏图更大
        subplot_titles=('', '')
    )
    
    # 策略收益率
    if use_log_scale:
        # 对数轴：log(1 + 累计收益率)
        y_strategy = np.log(strategy_nav)
        y_title = "收益率（对数）"
        strategy_max_dd_y = y_strategy[strategy_max_dd_idx]
    else:
        y_strategy = (strategy_nav - 1) * 100
        y_title = "收益率（%）"
        strategy_max_dd_y = y_strategy[strategy_max_dd_idx]
    
    # 基准收益率
    y_benchmark = None
    benchmark_max_dd_y = None
    if Dm is not None and len(Dm) > 0 and show_benchmark:
        benchmark_nav = np.cumprod(1 + Dm)
        if use_log_scale:
            y_benchmark = np.log(benchmark_nav)
            benchmark_pct = (benchmark_nav - 1) * 100  # 用于显示百分比
            benchmark_max_dd_y = y_benchmark[benchmark_max_dd_idx]
        else:
            y_benchmark = (benchmark_nav - 1) * 100
            benchmark_pct = y_benchmark
            benchmark_max_dd_y = y_benchmark[benchmark_max_dd_idx]
        
        # 基准线（不带阴影）
        fig.add_trace(
            go.Scatter(
                x=daily_pnl['日期'],
                y=y_benchmark,
                mode='lines',
                name='基准收益率',
                line=dict(color='#d62728', width=2),
                customdata=benchmark_pct,
                hovertemplate='基准收益: %{customdata:.2f}%<extra></extra>'
            ),
            row=1, col=1
        )
    
    # 超额收益率
    y_excess = None
    excess_max_dd_y = None
    if Dm is not None and len(Dm) > 0 and show_excess:
        benchmark_nav = np.cumprod(1 + Dm)
        # 确保数组长度匹配
        n = min(len(strategy_nav), len(benchmark_nav))
        excess_nav = strategy_nav[:n] / benchmark_nav[:n]
        
        if use_log_scale:
            # 对数轴：log(策略净值/基准净值) = log(策略净值) - log(基准净值)
            # 利用对数性质：log(x/y) = log(x) - log(y)
            y_excess = np.log(excess_nav)  # 等价于 np.log(strategy_nav[:n]) - np.log(benchmark_nav[:n])
            excess_pct = (excess_nav - 1) * 100  # 用于悬停显示百分比
            if excess_max_dd_idx is not None and excess_max_dd_idx < len(y_excess):
                excess_max_dd_y = y_excess[excess_max_dd_idx]
        else:
            y_excess = (excess_nav - 1) * 100
            excess_pct = y_excess
            if excess_max_dd_idx is not None and excess_max_dd_idx < len(y_excess):
                excess_max_dd_y = y_excess[excess_max_dd_idx]
        
        fig.add_trace(
            go.Scatter(
                x=daily_pnl['日期'][:n],
                y=y_excess,
                mode='lines',
                name='超额收益率',
                line=dict(color='#ff7f0e', width=2),
                customdata=excess_pct,
                hovertemplate='超额收益: %{customdata:.2f}%<extra></extra>'
            ),
            row=1, col=1
        )
    
    # 策略收益率（带阴影到0线）
    if show_strategy:
        # 准备 customdata：策略收益、日盈亏、累计收益、当日净值
        hover_data = []
        for i in range(len(daily_pnl)):
            hover_data.append([
                y_strategy[i],  # 策略收益率
                daily_pnl['日盈亏'].iloc[i],  # 日盈亏
                daily_pnl['累计收益'].iloc[i],  # 累计收益
                net_value.iloc[i]  # 当日净值
            ])
        
        fig.add_trace(
            go.Scatter(
                x=daily_pnl['日期'],
                y=y_strategy,
                mode='lines',
                name='策略收益率',
                line=dict(color='#1f77b4', width=2),
                fill='tozeroy',
                fillcolor='rgba(173, 216, 230, 0.3)',
                customdata=hover_data,
                hovertemplate=(
                    '策略收益: %{customdata[0]:.2f}%<br>'
                    '日盈亏: %{customdata[1]:,.2f}<br>'
                    '累计收益: %{customdata[2]:,.2f}<br>'
                    '当日净值: %{customdata[3]:,.2f}<br>'
                    '<extra></extra>'
                )
            ),
            row=1, col=1
        )
    
    # 标注各条曲线的最大回撤点（绿色圆点）
    
    # 策略最大回撤点
    if show_strategy:
        fig.add_trace(
            go.Scatter(
                x=[strategy_max_dd_date],
                y=[strategy_max_dd_y],
                mode='markers',
                name='策略最大回撤',
                marker=dict(symbol='circle', size=14, color='green', line=dict(width=2, color='darkgreen')),
                showlegend=False,
                hovertemplate=f'策略最大回撤<br>日期: {strategy_max_dd_date.strftime("%Y-%m-%d")}<br>回撤: {strategy_drawdown[strategy_max_dd_idx]:.2f}%<extra></extra>'
            ),
            row=1, col=1
        )
    
    # 超额收益最大回撤点
    if show_excess and excess_max_dd_idx is not None and excess_max_dd_y is not None:
        fig.add_trace(
            go.Scatter(
                x=[excess_max_dd_date],
                y=[excess_max_dd_y],
                mode='markers',
                name='超额最大回撤',
                marker=dict(symbol='circle', size=14, color='green', line=dict(width=2, color='darkgreen')),
                showlegend=False,
                hovertemplate=f'超额最大回撤<br>日期: {excess_max_dd_date.strftime("%Y-%m-%d")}<br>回撤: {excess_drawdown[excess_max_dd_idx]:.2f}%<extra></extra>'
            ),
            row=1, col=1
        )
    
    # 0线（黑色实线）
    fig.add_hline(y=0, line=dict(color='black', width=2, dash='solid'), row=1, col=1)
    
    # 添加日盈亏柱状图（第2个子图）- 使用高级配色
    colors = ['#7cb342' if x >= 0 else '#9c27b0' for x in daily_pnl['日盈亏']]  # 盈利：亮绿色，亏损：紫色
    fig.add_trace(
        go.Bar(
            x=daily_pnl['日期'],
            y=daily_pnl['日盈亏'],
            name='日盈亏',
            marker=dict(
                color=colors,
                line=dict(width=0)  # 无边框，更干净
            ),
            hovertemplate='日期: %{x}<br>日盈亏: %{y:,.2f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # 更新布局（网格背景）
    fig.update_layout(
        height=720,  # 缩小10%（800 * 0.9 = 720）
        showlegend=True,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12)
    )
    
    # 更新子图1（收益曲线）的Y轴
    # 对数轴：显示 log(净值)，范围扩大，Y轴标签显示为百分比
    if use_log_scale:
        # 范围：净值从0.65到1.25（-35%到+25%），确保超额收益能显示
        #就是相对于基准的收益，如果基准是1，那么策略是1.25，那么相对于基准的收益就是25%
        nav_min = 0.65   # -35%的净值
        nav_max = 1.25  # +25%的净值
        log_min = np.log(nav_min)
        log_max = np.log(nav_max)
        
        # 生成百分比刻度标签（-30%, -25%, ..., 0%, ..., 20%, 25%），每5%一个刻度
        pct_ticks = np.arange(-35, 25, 5)  # -30%到25%，每5%一个刻度
        nav_ticks = 1 + pct_ticks / 100  # 转换为净值
        log_ticks = np.log(nav_ticks)  # 转换为对数值
        
        fig.update_yaxes(
            title_text="收益率（对数，%）",
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            range=[log_min, log_max],
            tickvals=log_ticks,  # 设置刻度位置（对数值）
            ticktext=[f'{pct:.0f}%' for pct in pct_ticks],  # 显示百分比标签
            showline=True,
            linewidth=1,
            linecolor='lightgray',
            mirror=True,
            row=1, col=1
        )
    else:
        fig.update_yaxes(
            title_text=y_title,
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            dtick=5,  # 每5%一个刻度
            showline=True,
            linewidth=1,
            linecolor='lightgray',
            mirror=True,
            row=1, col=1
        )
    
    # 更新子图2（日盈亏）的Y轴
    fig.update_yaxes(
        title_text="日盈亏",
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        row=2, col=1
    )
    
    # 更新X轴（只在底部显示）
    fig.update_xaxes(
        title_text="",
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        tickformat='%Y-%m-%d',
        row=1, col=1
    )
    
    fig.update_xaxes(
        title_text="日期",
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        tickformat='%Y-%m-%d',
        row=2, col=1
    )
    
    return fig


def main():
    """Streamlit主界面"""
    st.title("📊 IM期货策略风险分析系统")
    
    # 加载数据
    with st.spinner("🔄 正在加载数据..."):
        data = load_and_process_data()
        df_raw = data['df']
    
    st.success(f"✅ 交易详情数据加载完成，共 {data['trading_days']} 个交易日")

    # 计算所有指标
    metrics = calculate_all_metrics(data)
    
    # ===== 风险指标（仿照聚宽格式）=====
    st.header("风险指标")
    
    # CSS: 文字省略号 + 字体放大125% + 加粗
    st.markdown("""
        <style>
        .metric-box {
            width: 100%;
            padding: 5px;
        }
        .metric-label {
            color: gray;
            font-size: 15px;
            margin-bottom: 4px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .metric-value {
            font-size: 23px;
            font-weight: 700;
            margin-top: 0;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # 计算策略最大回撤区间（最大回撤发生的起止日期）
    daily_pnl = data['daily_pnl']
    net_value = data['initial_capital'] + daily_pnl['累计收益']
    strategy_nav = net_value / data['initial_capital']
    strategy_running_max = np.maximum.accumulate(strategy_nav)
    strategy_drawdown = (strategy_running_max - strategy_nav) / strategy_running_max * 100
    
    # 找到策略最大回撤点
    strategy_max_dd_idx = strategy_drawdown.argmax()
    strategy_max_dd_date = daily_pnl['日期'].iloc[strategy_max_dd_idx]
    
    # 找到策略最大回撤的起点（峰值点）
    strategy_max_dd_start_idx = strategy_nav[:strategy_max_dd_idx+1].argmax()
    strategy_max_dd_start_date = daily_pnl['日期'].iloc[strategy_max_dd_start_idx]
    
    # 格式化日期
    start_date = strategy_max_dd_start_date.strftime('%Y/%m/%d')
    end_date = strategy_max_dd_date.strftime('%Y/%m/%d')
    
    # 使用22列布局（两行共用，确保垂直对齐）
    # 第一行：11个指标
    col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11 = st.columns(11)
    
    with col1:
        st.markdown(f'<div class="metric-box"><div class="metric-label">策略收益</div><div class="metric-value" style="color: green;">{data["strategy_total_return"]*100:.2f}%</div></div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'<div class="metric-box"><div class="metric-label">策略年化收益</div><div class="metric-value" style="color: green;">{data["strategy_annualized_return"]*100:.2f}%</div></div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'<div class="metric-box"><div class="metric-label">超额收益</div><div class="metric-value" style="color: green;">{metrics["excess_return"]:.2f}%</div></div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown(f'<div class="metric-box"><div class="metric-label">基准收益</div><div class="metric-value" style="color: red;">{data["benchmark_total_return"]*100:.2f}%</div></div>', unsafe_allow_html=True)
    
    with col5:
        st.markdown(f'<div class="metric-box"><div class="metric-label">阿尔法</div><div class="metric-value">{metrics["alpha"]/100:.3f}</div></div>', unsafe_allow_html=True)
    
    with col6:
        st.markdown(f'<div class="metric-box"><div class="metric-label">贝塔</div><div class="metric-value">{metrics["beta"]:.3f}</div></div>', unsafe_allow_html=True)
    
    with col7:
        st.markdown(f'<div class="metric-box"><div class="metric-label">夏普比率</div><div class="metric-value">{metrics["sharpe_ratio"]:.3f}</div></div>', unsafe_allow_html=True)
    
    with col8:
        st.markdown(f'<div class="metric-box"><div class="metric-label">胜率</div><div class="metric-value">{metrics["win_rate"]/100:.3f}</div></div>', unsafe_allow_html=True)
    
    with col9:
        st.markdown(f'<div class="metric-box"><div class="metric-label">盈亏比</div><div class="metric-value">{metrics["profit_loss_ratio"]:.3f}</div></div>', unsafe_allow_html=True)
    
    with col10:
        st.markdown(f'<div class="metric-box"><div class="metric-label">最大回撤</div><div class="metric-value">{metrics["max_drawdown"]:.2f}%</div></div>', unsafe_allow_html=True)
    
    with col11:
        st.markdown(f'<div class="metric-box"><div class="metric-label">索提诺比率</div><div class="metric-value">{metrics["sortino_ratio"]:.3f}</div></div>', unsafe_allow_html=True)
    
    # 第二行：11列布局（10个指标 + 1个空列，确保与第一行垂直对齐）
    col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11 = st.columns(11)
    
    with col1:
        st.markdown(f'<div class="metric-box"><div class="metric-label">日均超额收益</div><div class="metric-value">{metrics["aei"]:.2f}%</div></div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'<div class="metric-box"><div class="metric-label">超额收益最大回撤</div><div class="metric-value">{metrics["excess_max_drawdown"]:.2f}%</div></div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'<div class="metric-box"><div class="metric-label">超额收益夏普比率</div><div class="metric-value">{metrics["excess_sharpe_ratio"]:.3f}</div></div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown(f'<div class="metric-box"><div class="metric-label">日胜率</div><div class="metric-value">{metrics["daily_win_rate"]/100:.3f}</div></div>', unsafe_allow_html=True)
    
    with col5:
        st.markdown(f'<div class="metric-box"><div class="metric-label">盈利次数</div><div class="metric-value">{int(metrics["win_trades"])}</div></div>', unsafe_allow_html=True)
    
    with col6:
        st.markdown(f'<div class="metric-box"><div class="metric-label">亏损次数</div><div class="metric-value">{int(metrics["loss_trades"])}</div></div>', unsafe_allow_html=True)
    
    with col7:
        st.markdown(f'<div class="metric-box"><div class="metric-label">信息比率</div><div class="metric-value">{metrics["information_ratio"]:.3f}</div></div>', unsafe_allow_html=True)
    
    with col8:
        st.markdown(f'<div class="metric-box"><div class="metric-label">策略波动率</div><div class="metric-value">{metrics["strategy_volatility"]/100:.3f}</div></div>', unsafe_allow_html=True)
    
    with col9:
        st.markdown(f'<div class="metric-box"><div class="metric-label">基准波动率</div><div class="metric-value">{metrics["benchmark_volatility"]/100:.3f}</div></div>', unsafe_allow_html=True)
    
    # 最大回撤区间占两列（col10 + col11）
    with col10:
        st.markdown(f"""
            <div class="metric-box" style="width: 200%; box-sizing: border-box;">
                <div class="metric-label">最大回撤区间</div>
                <div class="metric-value" style="font-size: 18px;">{start_date} - {end_date}</div>
            </div>
        """, unsafe_allow_html=True)
    
    # col11 被col10占用
    with col11:
        pass
    
    st.divider()
    
    # ===== 交易信号图 =====
    st.header("📈 交易信号图")
    with st.spinner("🎨 正在绘制交易信号图..."):
        fig_signal = plot_signal_chart(data, df_raw)
        st.plotly_chart(fig_signal, use_container_width=True)
    
    st.divider()
    
    # ===== 收益曲线图 =====
    st.header("📈 收益曲线")
    
    # 曲线选择和坐标轴选项（参考聚宽布局）
    col_option1, col_option2, col_option3, col_option4 = st.columns(4)
    
    with col_option1:
        show_strategy = st.checkbox("策略收益率", value=True)
    with col_option2:
        show_benchmark = st.checkbox("基准收益率", value=True)
    with col_option3:
        show_excess = st.checkbox("超额收益率", value=False)
    with col_option4:
        use_log_scale = st.checkbox("对数轴", value=False, help="对数轴可以更清晰地展示收益率变化")
    
    with st.spinner("🎨 正在绘制收益曲线..."):
        fig_returns = plot_returns_chart(data, use_log_scale, show_strategy, show_benchmark, show_excess)
        st.plotly_chart(fig_returns, use_container_width=True)
    
    st.divider()#分割线
    
    # ===== 数据查看 =====
    with st.expander("📋 查看详细交易数据"):
        st.dataframe(df_raw, use_container_width=True)#use_container_width=True用于使数据框适应容器宽度
    with st.expander("📋 查看日度汇总数据"):
        st.dataframe(data['daily_pnl'], use_container_width=True)


if __name__ == "__main__":
    main()  
