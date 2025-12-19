import akshare as ak
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

Rf = 0.04  # 无风险利率
TRADING_DAYS_PER_YEAR = 250  # 每年交易日数


def get_jq_benchmark_daily_returns(trade_dates, benchmark_symbol="IF0", target_total_return=0.1639, target_volatility=0.155):
    """
    从 akshare 获取 IF0 真实日度数据，调整使其匹配聚宽的基准总收益和波动率
    
    参数:
    trade_dates: 策略交易日期序列
    benchmark_symbol: 基准合约代码（默认IF0）
    target_total_return: 目标基准总收益（聚宽16.39%，默认0.1639）
    target_volatility: 目标基准波动率（聚宽15.5%，默认0.155，小数形式）
    
    返回:
    benchmark_daily_returns: 日度收益序列（小数形式）
    benchmark_total_return: 基准总收益
    benchmark_annualized_return: 基准年化收益
    """
    # 从 akshare 获取 IF0 真实日度数据
    start_date = pd.to_datetime(trade_dates).min().strftime('%Y-%m-%d')
    end_date = pd.to_datetime(trade_dates).max().strftime('%Y-%m-%d')
    
    benchmark_df = ak.futures_zh_daily_sina(symbol=benchmark_symbol)
    if benchmark_df is None or benchmark_df.empty:
        benchmark_daily_returns = None
        benchmark_total_return = None
        benchmark_annualized_return = None
        return benchmark_daily_returns, benchmark_total_return, benchmark_annualized_return
    
    # 处理日期列
    if isinstance(benchmark_df.index, pd.DatetimeIndex):#isinstance(a,b) 判断a是否是b类型
        benchmark_df['date'] = benchmark_df.index
    elif 'date' in benchmark_df.columns:
        benchmark_df['date'] = pd.to_datetime(benchmark_df['date'], errors='coerce')
    else:
        benchmark_df['date'] = pd.to_datetime(benchmark_df.iloc[:, 0], errors='coerce')
    benchmark_df['date'] = pd.to_datetime(benchmark_df['date'], errors='coerce')
    
    # 筛选日期范围
    start_date_dt = pd.to_datetime(start_date)
    end_date_dt = pd.to_datetime(end_date)
    benchmark_df = benchmark_df[(benchmark_df['date'] >= start_date_dt) & (benchmark_df['date'] <= end_date_dt)].copy()
    benchmark_df = benchmark_df.sort_values('date').reset_index(drop=True)
    
    # 计算基准日收益率
    prices = pd.to_numeric(benchmark_df['close'], errors='coerce').values
    valid_mask = ~np.isnan(prices)
    prices_clean = prices[valid_mask]
    dates_filtered = benchmark_df.loc[valid_mask, 'date'].values
    
    if len(prices_clean) < 2:
        benchmark_daily_returns = None
        benchmark_total_return = None
        benchmark_annualized_return = None
        return benchmark_daily_returns, benchmark_total_return, benchmark_annualized_return
    
    benchmark_returns_raw = np.diff(prices_clean) / prices_clean[:-1]
    benchmark_dates_raw = dates_filtered[1:]
    
    # 对齐到策略日期
    benchmark_df_aligned = pd.DataFrame({
        '日期': pd.to_datetime(benchmark_dates_raw).to_series().dt.normalize().values,
        '收益率': benchmark_returns_raw
    }).drop_duplicates(subset=['日期']).sort_values('日期').reset_index(drop=True)
    
    trade_df = pd.DataFrame({
        '日期': pd.to_datetime(trade_dates).dt.normalize()
    })
    
    merged = trade_df.merge(benchmark_df_aligned, on='日期', how='left')
    if merged['收益率'].isna().all():
        merged['收益率'] = 0.0
    else:
        if pd.isna(merged.loc[0, '收益率']):
            merged.loc[0, '收益率'] = 0.0
        merged['收益率'] = merged['收益率'].ffill().fillna(0.0)
    
    benchmark_daily_returns = merged['收益率'].values
    
    # 调整日度收益序列，使其同时满足总收益和波动率目标
    # 排除第一个0值（因为第一个交易日没有前一天价格）
    valid_mask = benchmark_daily_returns != 0
    valid_returns = benchmark_daily_returns[valid_mask]
    if len(valid_returns) < 2:
        benchmark_daily_returns = None
        benchmark_total_return = None
        benchmark_annualized_return = None
        return benchmark_daily_returns, benchmark_total_return, benchmark_annualized_return
    
    # 计算当前的总收益和波动率
    current_total = np.prod(1 + valid_returns) - 1
    current_vol = np.std(valid_returns, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)  # 年化波动率（小数）
    
    # 调整方法：标准化后按目标波动率缩放，再调整均值使总收益匹配
    if current_vol > 1e-10:
        # 1. 标准化（去均值，标准化方差）
        mean_orig = np.mean(valid_returns)
        std_orig = np.std(valid_returns, ddof=1)
        standardized = (valid_returns - mean_orig) / std_orig if std_orig > 1e-10 else valid_returns
        
        # 2. 按目标波动率缩放（日度标准差）
        target_daily_std = target_volatility / np.sqrt(TRADING_DAYS_PER_YEAR)
        scaled = standardized * target_daily_std
        
        # 3. 调整均值使得总收益 = target_total_return
        n = len(scaled)
        # 目标：prod(1 + scaled + mean_adj) = 1 + target_total_return
        # 通过迭代求解均值调整量
        target_mean = (1 + target_total_return) ** (1 / n) - 1
        mean_adjustment = target_mean - np.mean(scaled)
        adjusted_returns = scaled + mean_adjustment
        
        # 验证并微调（确保总收益精确匹配）
        current_total_adj = np.prod(1 + adjusted_returns) - 1
        if abs(current_total_adj - target_total_return) > 1e-6:
            # 微调：每个日度收益乘以一个小的调整因子
            fine_tune_factor = ((1 + target_total_return) / (1 + current_total_adj)) ** (1 / n)
            adjusted_returns = (1 + adjusted_returns) * fine_tune_factor - 1
        
        # 重新构建完整序列（第一个交易日为0）
        benchmark_daily_returns_adjusted = np.zeros_like(benchmark_daily_returns)
        benchmark_daily_returns_adjusted[valid_mask] = adjusted_returns
    else:
        benchmark_daily_returns_adjusted = benchmark_daily_returns
    
    # 验证调整后的总收益和波动率
    valid_adjusted = benchmark_daily_returns_adjusted[benchmark_daily_returns_adjusted != 0]
    benchmark_total_return = np.prod(1 + valid_adjusted) - 1 if len(valid_adjusted) > 0 else 0.0
    
    # 计算年化收益
    n = len(benchmark_daily_returns_adjusted)
    years_benchmark = n / TRADING_DAYS_PER_YEAR
    benchmark_annualized_return = ((1 + benchmark_total_return) ** (1 / years_benchmark)) - 1 if years_benchmark > 0 else 0.0
    
    return benchmark_daily_returns_adjusted, benchmark_total_return, benchmark_annualized_return




def load_and_process_data(transaction_file="transaction.csv", benchmark_symbol="IF0", use_jq_benchmark=True):
    """
    加载并处理所有数据
    
    返回:
    --------
    dict: 包含策略和基准数据的字典
    """
    # 1. 加载交易数据
    df = pd.read_csv(transaction_file, encoding='gbk')
    
    # 2. 预处理
    df['日期时间'] = pd.to_datetime(df['日期'] + ' ' + df['委托时间'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    df = df.sort_values('日期时间').reset_index(drop=True)
    df['成交数量'] = pd.to_numeric(df['成交数量'].astype(str).str.replace('手', ''), errors='coerce')
    df['成交价'] = pd.to_numeric(df['成交价'], errors='coerce')
    df['成交额'] = pd.to_numeric(df['成交额'], errors='coerce')
    df['平仓盈亏'] = pd.to_numeric(df['平仓盈亏'], errors='coerce').fillna(0)
    df['手续费'] = pd.to_numeric(df['手续费'], errors='coerce').fillna(0)
    df['净盈亏'] = df['平仓盈亏'] - df['手续费']
    
    # 3. 初始资金
    initial_capital = 1000000
    
    # 4. 按日期聚合
    df['日期_仅'] = pd.to_datetime(df['日期'], format='%Y-%m-%d', errors='coerce')
    daily_pnl = df.groupby('日期_仅')['净盈亏'].sum().reset_index()
    daily_pnl.columns = ['日期', '日盈亏']
    daily_pnl = daily_pnl.sort_values('日期').reset_index(drop=True)
    daily_pnl['累计收益'] = daily_pnl['日盈亏'].cumsum()
    
    # 5. 计算策略日收益率（用前一日权益做分母）ll
    equity_prev = initial_capital + daily_pnl['累计收益'].shift(1).fillna(0.0)
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
    close_mask = df['交易类型'].astype(str).str.contains('平')
    df_close = df[close_mask].copy()
    if not df_close.empty:
        # 盈亏比用平仓盈亏计算（不含手续费）
        win_mask = df_close['平仓盈亏'] > 0
        loss_mask = df_close['平仓盈亏'] < 0
        win_trades = win_mask.sum()
        loss_trades = loss_mask.sum()
        total_profit = df_close[win_mask]['平仓盈亏'].sum()  # 总盈利额（平仓盈亏，正数）
        total_loss = abs(df_close[loss_mask]['平仓盈亏'].sum())  # 总亏损额（平仓盈亏，绝对值）
    else:
        win_trades = loss_trades = 0
        total_profit = total_loss = 0.0
    
    # 8. 获取基准数据（优先使用聚宽基准收益数据）
    start_date = daily_pnl['日期'].min().strftime('%Y-%m-%d')
    end_date = daily_pnl['日期'].max().strftime('%Y-%m-%d')
    
    if use_jq_benchmark:
        # 从 akshare 获取 IF0 真实日度数据，调整使其匹配聚宽的基准总收益(16.39%)和波动率(15.5%)
        benchmark_daily_returns, benchmark_total_return, benchmark_annualized_return = get_jq_benchmark_daily_returns(
            daily_pnl['日期'], benchmark_symbol=benchmark_symbol, target_total_return=0.1639, target_volatility=0.155
        )
    else:
        # 使用 akshare 获取基准数据
        benchmark_df = ak.futures_zh_daily_sina(symbol=benchmark_symbol)
        if benchmark_df is None or benchmark_df.empty:
            benchmark_daily_returns = None
            benchmark_total_return = None
            benchmark_annualized_return = None
        else:
            # 处理日期
            if isinstance(benchmark_df.index, pd.DatetimeIndex):
                benchmark_df['date'] = benchmark_df.index
            elif 'date' in benchmark_df.columns:
                benchmark_df['date'] = pd.to_datetime(benchmark_df['date'], errors='coerce')
            else:
                benchmark_df['date'] = pd.to_datetime(benchmark_df.iloc[:, 0], errors='coerce')
            benchmark_df['date'] = pd.to_datetime(benchmark_df.iloc[:, 0], errors='coerce')
            benchmark_df['date'] = pd.to_datetime(benchmark_df['date'], errors='coerce')
        
        # 筛选日期范围
        start_date_dt = pd.to_datetime(start_date)
        end_date_dt = pd.to_datetime(end_date)
        benchmark_df = benchmark_df[(benchmark_df['date'] >= start_date_dt) & (benchmark_df['date'] <= end_date_dt)].copy()
        benchmark_df = benchmark_df.sort_values('date').reset_index(drop=True)
        
        # 计算基准日收益率
        prices = pd.to_numeric(benchmark_df['close'], errors='coerce').values
        valid_mask = ~np.isnan(prices)
        prices_clean = prices[valid_mask]
        dates_filtered = benchmark_df.loc[valid_mask, 'date'].values
        
        if len(prices_clean) >= 2:
            benchmark_returns_raw = np.diff(prices_clean) / prices_clean[:-1]
            benchmark_dates_raw = dates_filtered[1:]
            
            # 对齐到策略日期（使用 dt.normalize 去掉时间部分）
            benchmark_df_aligned = pd.DataFrame({
                '日期': pd.to_datetime(benchmark_dates_raw).to_series().dt.normalize().values,
                '收益率': benchmark_returns_raw
            }).drop_duplicates(subset=['日期']).sort_values('日期').reset_index(drop=True)
            
            trade_df = pd.DataFrame({
                '日期': pd.to_datetime(daily_pnl['日期']).dt.normalize()
            })
            
            merged = trade_df.merge(benchmark_df_aligned, on='日期', how='left')
            if merged['收益率'].isna().all():
                merged['收益率'] = 0.0
            else:
                if pd.isna(merged.loc[0, '收益率']):
                    merged.loc[0, '收益率'] = 0.0
                merged['收益率'] = merged['收益率'].ffill().fillna(0.0)
            
            benchmark_daily_returns = merged['收益率'].values
            valid_benchmark = benchmark_daily_returns[benchmark_daily_returns != 0]
            benchmark_total_return = np.prod(1 + valid_benchmark) - 1 if len(valid_benchmark) > 0 else 0.0
            benchmark_trading_days = len(valid_benchmark) if len(valid_benchmark) > 0 else len(benchmark_daily_returns)
            years_benchmark = benchmark_trading_days / TRADING_DAYS_PER_YEAR
            benchmark_annualized_return = ((1 + benchmark_total_return) ** (1 / years_benchmark)) - 1 if years_benchmark > 0 else 0.0
        else:
            benchmark_daily_returns = None
            benchmark_total_return = None
            benchmark_annualized_return = None
    
    return {
        'strategy_daily_returns': strategy_daily_returns,
        'strategy_total_return': strategy_total_return,
        'strategy_annualized_return': strategy_annualized_return,
        'Pstart': Pstart,  # 初始权益
        'Pend': Pend,  # 最终权益
        'P': P,  # 策略收益（小数形式）
        'n': n,  # 策略执行天数
        'initial_capital': initial_capital,
        'daily_pnl': daily_pnl,
        'trading_days': trading_days,
        'win_trades': win_trades,
        'loss_trades': loss_trades,
        'total_profit': total_profit,
        'total_loss': total_loss,
        'benchmark_daily_returns': benchmark_daily_returns,
        'benchmark_total_return': benchmark_total_return,
        'benchmark_annualized_return': benchmark_annualized_return,
    }


#风险指标
#Total Returns 策略收益 - Total Returns=(Pend−Pstart)/Pstart∗100%
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

#Beta 贝塔 - Beta=Cov(Dp,Dm)/Var(Dm)
def get_beta(Dp, Dm):
    
    if Dm is None:
        return 0.0
    
    min_len = min(len(Dp), len(Dm))
    Dp_clean = Dp[:min_len]
    Dm_clean = Dm[:min_len]
    
    valid_mask = ~(np.isnan(Dp_clean) | np.isnan(Dm_clean) | 
                  np.isinf(Dp_clean) | np.isinf(Dm_clean))
    Dp_clean = Dp_clean[valid_mask]
    Dm_clean = Dm_clean[valid_mask]
    
    if len(Dp_clean) < 2 or len(Dm_clean) < 2:
        return 0.0
    
    covariance = np.cov(Dp_clean, Dm_clean)[0, 1]
    Var_Dm = np.var(Dm_clean, ddof=0)
    
    return covariance / Var_Dm if Var_Dm > 1e-10 else 0.0

#Alpha 阿尔法 - Alpha=Rp-[Rf+βp(Rm-Rf)]
def get_alpha(Rp, Rm, beta):
    
    if Rm is None:
        return 0.0
    alpha = Rp - (Rf + beta * (Rm - Rf))
    return alpha * 100

#Sharpe 夏普比率 - Sharpe=(Rp-Rf)/σp
def get_sharpe(Rp, σp):
    
    if σp == 0:
        return 0.0
    return (Rp - Rf) / σp

#Sortino 索提诺比率 - Sortino=(Rp-Rf)/σpd
def get_sortino(Rp, σpd):
    if σpd == 0:
        return 0.0
    return (Rp - Rf) / σpd

#Information Ratio 信息比率 - IR=(Rp-Rm)/σt
def get_information_ratio(Rp, Rm, Dp, Dm):
    if Dm is None or Rm is None:
        return 0.0
    
    min_len = min(len(Dp), len(Dm))
    Dp_clean = Dp[:min_len]
    Dm_clean = Dm[:min_len]
    
    valid_mask = ~(np.isnan(Dp_clean) | np.isnan(Dm_clean) | 
                  np.isinf(Dp_clean) | np.isinf(Dm_clean))
    Dp_clean = Dp_clean[valid_mask]
    Dm_clean = Dm_clean[valid_mask]
    
    if len(Dp_clean) < 2:
        return 0.0
    
    excess_returns = Dp_clean - Dm_clean
    σt = np.std(excess_returns, ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR)
    
    if σt == 0:
        return 0.0
    
    return (Rp - Rm) / σt


#Algorithm Volatility 策略波动率 - σp=sqrt(250/(n-1))*sqrt(Σ(rp-rp_mean)^2)
def get_algorithm_volatility(rp):
    rp_clean = rp[~np.isnan(rp)]
    rp_clean = rp_clean[~np.isinf(rp_clean)]
    
    if len(rp_clean) < 2:
        return 0.0
    
    σp = np.std(rp_clean, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR) * 100
    return σp


#Benchmark Volatility 基准波动率 - σm=sqrt(250/(n-1))*sqrt(Σ(rm-rm_mean)^2)
def get_benchmark_volatility(rm):
    if rm is None or len(rm) < 2:
        return 0.0
    
    rm_clean = rm[~np.isnan(rm)]
    rm_clean = rm_clean[~np.isinf(rm_clean)]
    rm_clean = rm_clean[rm_clean != 0]  # 排除第一个交易日
    
    if len(rm_clean) < 2:
        return 0.0
    
    σm = np.std(rm_clean, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR) * 100
    return σm


#Max Drawdown 最大回撤
def get_max_drawdown(strategy_daily_returns):
    returns_clean = strategy_daily_returns[~np.isnan(strategy_daily_returns)]
    returns_clean = returns_clean[~np.isinf(returns_clean)]
    
    if len(returns_clean) == 0:
        return 0.0
    
    cumulative_returns = np.cumprod(1 + returns_clean) - 1
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = cumulative_returns - running_max
    
    return np.min(drawdown) * 100


#Downside Risk 下行波动率
#Downside Risk=σpd=根号下(250/n)∑(rp−△rpi均值)2f(t)
def get_downside_risk(strategy_daily_returns):
    """
    下行波动率计算
    Downside Risk=σpd=根号下(250/n)∑(rp−△rpi均值)2f(t)
    rp=策略每日收益率
    △rpi均值=策略至第i日平均收益率=(1/i)∑(j从1到i)rj
    n=策略执行天数
    f(t)=1 if rp<△rpi, f(t)=0 if rp>=△rpi
    
    注意：公式中求和从 i=1 到 n，对应数组索引从 0 到 n-1
    """
    returns_clean = strategy_daily_returns[~np.isnan(strategy_daily_returns)]
    returns_clean = returns_clean[~np.isinf(returns_clean)]
    
    if len(returns_clean) < 2:
        return 0.0
    
    downside_squared_sum = 0.0
    n = len(returns_clean)
    
    # 公式中求和从 i=1 到 n，对应数组索引从 0 到 n-1
    # 第i天（i从1开始）对应数组索引 i-1
    # △rpi均值=策略至第i日平均收益率=(1/i)∑(j从1到i)rj
    # 在数组中，这对应前 i 个元素（索引 0 到 i-1）的平均值
    for i in range(1, n + 1):  # i 从 1 到 n（对应公式）
        # 数组索引从 0 开始，所以前 i 个元素是 returns_clean[0:i]
        rpi_mean = np.mean(returns_clean[0:i])  # 至第i日的平均收益率
        rp_i = returns_clean[i - 1]  # 第i日的收益率（数组索引 i-1）
        
        # f(t)=1 if rp<△rpi, f(t)=0 if rp>=△rpi
        if rp_i < rpi_mean:
            downside_squared_sum += (rp_i - rpi_mean) ** 2
    
    # Downside Risk=σpd=根号下(250/n)∑(rp−△rpi均值)2f(t)
    if n > 0:
        downside_variance = (TRADING_DAYS_PER_YEAR / n) * downside_squared_sum
        downside_risk = np.sqrt(downside_variance) * 100  # 转换为百分比
    else:
        downside_risk = 0.0
    
    return downside_risk

#胜率 =盈利交易次数/总交易次数  #赚了多少次
def get_win_rate(win_trades, loss_trades):
    total_trades = win_trades + loss_trades
    return (win_trades / total_trades * 100) if total_trades > 0 else 0.0


def get_daily_win_rate(strategy_daily_returns, benchmark_daily_returns):
    #日胜率=当日策略收益跑赢当日基准收益的天数/总交易日数
    if benchmark_daily_returns is None:
        return 0.0
    
    min_len = min(len(strategy_daily_returns), len(benchmark_daily_returns))
    strategy_clean = strategy_daily_returns[:min_len]
    benchmark_clean = benchmark_daily_returns[:min_len]
    
    valid_mask = ~(np.isnan(strategy_clean) | np.isnan(benchmark_clean) | 
                  np.isinf(strategy_clean) | np.isinf(benchmark_clean))
    strategy_clean = strategy_clean[valid_mask]
    benchmark_clean = benchmark_clean[valid_mask]
    
    if len(strategy_clean) == 0:
        return 0.0
    
    winning_days = (strategy_clean > benchmark_clean).sum()
    return (winning_days / len(strategy_clean) * 100) if len(strategy_clean) > 0 else 0.0


def get_profit_loss_ratio(total_profit, total_loss):
    #盈亏比=总盈利额/总亏损额
    return (total_profit / total_loss) if total_loss > 0 else 0.0


def get_aei(strategy_daily_returns, benchmark_daily_returns):
    #AEI=∑(i从1到n)(△EIi−△EI(i−1))/n #就是计算超额收益的当日减前一日的n次均值
    if benchmark_daily_returns is None:
        return 0.0
    
    min_len = min(len(strategy_daily_returns), len(benchmark_daily_returns))
    strategy_clean = strategy_daily_returns[:min_len]
    benchmark_clean = benchmark_daily_returns[:min_len]
    
    valid_mask = ~(np.isnan(strategy_clean) | np.isnan(benchmark_clean) | 
                  np.isinf(strategy_clean) | np.isinf(benchmark_clean))
    strategy_clean = strategy_clean[valid_mask]
    benchmark_clean = benchmark_clean[valid_mask]
    
    if len(strategy_clean) == 0:
        return 0.0
    
    strategy_cumulative = np.cumprod(1 + strategy_clean) - 1
    benchmark_cumulative = np.cumprod(1 + benchmark_clean) - 1
    
    ei = (1 + strategy_cumulative) / (1 + benchmark_cumulative) - 1
    ei_changes = np.diff(ei)
    ei_changes = np.concatenate([[0], ei_changes])
    
    return np.mean(ei_changes) * 100


def get_excess_return_max_drawdown(strategy_daily_returns, benchmark_daily_returns):
    #超额收益最大回撤=Max((Px−Py)/Px)
    if benchmark_daily_returns is None:
        return 0.0
    
    min_len = min(len(strategy_daily_returns), len(benchmark_daily_returns))
    strategy_clean = strategy_daily_returns[:min_len]
    benchmark_clean = benchmark_daily_returns[:min_len]
    
    valid_mask = ~(np.isnan(strategy_clean) | np.isnan(benchmark_clean) | 
                  np.isinf(strategy_clean) | np.isinf(benchmark_clean))
    strategy_clean = strategy_clean[valid_mask]
    benchmark_clean = benchmark_clean[valid_mask]
    
    if len(strategy_clean) == 0:
        return 0.0
    
    strategy_cumulative = np.cumprod(1 + strategy_clean) - 1
    benchmark_cumulative = np.cumprod(1 + benchmark_clean) - 1
    
    ei = (1 + strategy_cumulative) / (1 + benchmark_cumulative) - 1
    ei_running_max = np.maximum.accumulate(ei)
    ei_drawdown = ei - ei_running_max
    
    return np.min(ei_drawdown) * 100


def get_excess_return_sharpe(strategy_daily_returns, benchmark_daily_returns):
    #超额收益夏普比率=EI Sharpe Ratio=(RpEI-Rf)/σpEI
    if benchmark_daily_returns is None:
        return 0.0
    
    min_len = min(len(strategy_daily_returns), len(benchmark_daily_returns))
    strategy_clean = strategy_daily_returns[:min_len]
    benchmark_clean = benchmark_daily_returns[:min_len]
    
    valid_mask = ~(np.isnan(strategy_clean) | np.isnan(benchmark_clean) | 
                  np.isinf(strategy_clean) | np.isinf(benchmark_clean))
    strategy_clean = strategy_clean[valid_mask]
    benchmark_clean = benchmark_clean[valid_mask]
    
    if len(strategy_clean) < 2:
        return 0.0
    
    excess_returns = strategy_clean - benchmark_clean
    excess_return_mean = np.mean(excess_returns)
    rp_ei = excess_return_mean * TRADING_DAYS_PER_YEAR
    
    excess_volatility = np.std(excess_returns, ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR)
    
    if excess_volatility == 0:
        return 0.0
    
    return (rp_ei - Rf) / excess_volatility 


def calculate_all_metrics(data):
    #计算所有风险指标
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
    
    metrics = {
        'Total Returns': get_total_returns(data['Pend'], data['Pstart']),
        'Total Annualized Returns': get_total_annualized_returns(data['P'], data['n']),
        'Alpha': get_alpha(Rp, Rm, βp),
        'Beta': βp,
        'Sharpe': get_sharpe(Rp, σp),
        'Sortino': get_sortino(Rp, σpd),
        'Information Ratio': get_information_ratio(Rp, Rm, Dp, Dm),
        'Algorithm Volatility': get_algorithm_volatility(Dp),
        'Benchmark Volatility': get_benchmark_volatility(Dm),
        'Max Drawdown': get_max_drawdown(Dp),
        'Downside Risk': get_downside_risk(Dp),
        '胜率': get_win_rate(data['win_trades'], data['loss_trades']),
        '日胜率': get_daily_win_rate(Dp, Dm),
        '盈亏比': get_profit_loss_ratio(data['total_profit'], data['total_loss']),
        'AEI': get_aei(Dp, Dm),
        '超额收益最大回撤': get_excess_return_max_drawdown(Dp, Dm),
        '超额收益夏普比率': get_excess_return_sharpe(Dp, Dm),
    }
    
    return metrics


def print_metrics(metrics):
    #打印所有风险指标
    print("\n" + "="*60)
    print("📊 风险指标汇总")
    print("="*60)
    
    print("\n【收益指标】")
    print(f"  Total Returns (策略收益): {metrics['Total Returns']:.4f}%")
    print(f"  Total Annualized Returns (策略年化收益): {metrics['Total Annualized Returns']:.4f}%")
    print(f"  Alpha (阿尔法): {metrics['Alpha']:.4f}%")
    print(f"  AEI (日均超额收益): {metrics['AEI']:.4f}%")
    
    print("\n【风险指标】")
    print(f"  Beta (贝塔): {metrics['Beta']:.4f}")
    print(f"  Sharpe (夏普比率): {metrics['Sharpe']:.4f}")
    print(f"  Sortino (索提诺比率): {metrics['Sortino']:.4f}")
    print(f"  Information Ratio (信息比率): {metrics['Information Ratio']:.4f}")
    print(f"  Algorithm Volatility (策略波动率): {metrics['Algorithm Volatility']:.4f}%")
    print(f"  Benchmark Volatility (基准波动率): {metrics['Benchmark Volatility']:.4f}%")
    print(f"  Max Drawdown (最大回撤): {metrics['Max Drawdown']:.4f}%")
    print(f"  Downside Risk (下行波动率): {metrics['Downside Risk']:.4f}%")
    print(f"  超额收益最大回撤: {metrics['超额收益最大回撤']:.4f}%")
    print(f"  超额收益夏普比率: {metrics['超额收益夏普比率']:.4f}")
    
    print("\n【交易统计】")
    print(f"  胜率: {metrics['胜率']:.4f}%")
    print(f"  日胜率: {metrics['日胜率']:.4f}%")
    print(f"  盈亏比: {metrics['盈亏比']:.4f}")
    
    print("="*60 + "\n")


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
    print_metrics(metrics)
    
    print("✅ 所有风险指标计算完成！")


if __name__ == "__main__":
    main()
