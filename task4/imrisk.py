import akshare as ak
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

Rf = 0.04  # 无风险利率
TRADING_DAYS_PER_YEAR = 250  # 每年交易日数


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
    start_date = daily_pnl['日期'].min().strftime('%Y%m%d')
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
        price_col = None
        for col in benchmark_df.columns:
            if '收盘' in str(col) or 'close' in str(col).lower():
                price_col = col
                break
        prices = pd.to_numeric(benchmark_df[price_col], errors='coerce').values if price_col else None
    
    if prices is None or len(prices) < 2:
        return {'benchmark_daily_returns': None, 'benchmark_total_return': None, 'benchmark_annualized_return': None}
    
    # 构建基准数据DataFrame（包含所有获取到的数据，用于计算收益率）
    benchmark_price_df = pd.DataFrame({
        '日期': pd.to_datetime(benchmark_df['date']).dt.normalize().values,
        '收盘价': prices
    }).drop_duplicates(subset=['日期']).sort_values('日期').reset_index(drop=True)
    
    # 先计算基准数据的日收益率（基于完整数据，确保第一个交易日有前一日数据）
    benchmark_price_df['基准日收益率'] = benchmark_price_df['收盘价'].pct_change()
    
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


def load_and_process_data(transaction_file="transaction.csv", benchmark_symbol="IF0"):
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
    
    # 8. 获取基准数据（沪深300指数 000300）
    # 根据聚宽策略代码：set_benchmark('000300.XSHG')，基准是沪深300指数
    benchmark_result = get_benchmark_data(daily_pnl)
    benchmark_daily_returns = benchmark_result['benchmark_daily_returns']
    benchmark_total_return = benchmark_result['benchmark_total_return']
    benchmark_annualized_return = benchmark_result['benchmark_annualized_return']
    return {
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
    
    Cov = np.cov(Dp, Dm)[0, 1]#np.cov()[0, 1] 获取协方差
    Var_Dm = np.var(Dm, ddof=0)#np.var() 计算方差
    
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
    excess = Dp[:n] - Dm[:n]#excess 策略每日收益与基准每日收益的差值
    σt = np.std(excess, ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR)#np.std() 计算标准差,ddof=0 表示分母为n
    
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
    
    cumulative = np.cumprod(1 + Dp) - 1
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative - running_max
    
    return np.min(drawdown) * 100


#Downside Risk 下行波动率 - σpd=根号下(250/n)∑(rp−△rpi均值)2f(t)
def get_downside_risk(Dp):
    """
    σpd=根号下(250/n)∑(rp−△rpi均值)2f(t)
    rp=策略每日收益率, △rpi均值=策略至第i日平均收益率=(1/i)∑(j从1到i)rj
    f(t)=1 if rp<△rpi, f(t)=0 if rp>=△rpi
    """
    if len(Dp) < 2:
        return 0.0
    
    n = len(Dp)
    downside_sum = 0.0
    
    for i in range(1, n + 1):  # i从1到n
        rpi_mean = np.mean(Dp[0:i])  # 至第i日平均收益率
        rp_i = Dp[i - 1]  # 第i日收益率
        if rp_i < rpi_mean:
            downside_sum += (rp_i - rpi_mean) ** 2
    
    σpd = np.sqrt((TRADING_DAYS_PER_YEAR / n) * downside_sum) * 100
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
    RpEI=年化超额收益率（基于EI的日变化率计算）
    σpEI=超额收益波动率（基于EI的日变化率的标准差，年化）
    """
    if Dm is None or len(Dp) < 2 or len(Dm) < 2:
        return 0.0
    
    n = min(len(Dp), len(Dm))
    sp_cum = np.cumprod(1 + Dp[:n])  # 策略累计净值
    bm_cum = np.cumprod(1 + Dm[:n])  # 基准累计净值
    
    # EI=(策略收益+100%)/(基准收益+100%)−100% = sp_cum/bm_cum - 1
    EI = sp_cum / bm_cum - 1
    
    # 计算EI的日变化率（类似收益率）
    # EI_return[i] = (EI[i] - EI[i-1]) / (1 + EI[i-1])
    # 但EI可能为负，所以需要特殊处理
    EI_prev = np.concatenate([[0], EI[:-1]])  # 前一日EI值，第一天为0
    EI_returns = np.diff(EI)  # EI的日变化（绝对值）
    
    # 计算EI的日变化率：如果前一日EI接近0，使用简单差值；否则使用相对变化
    # 但更简单的方法：直接使用EI的日变化的标准差
    # 或者：计算EI的日变化率 = (EI[i] - EI[i-1]) / max(1, abs(1 + EI[i-1]))
    
    # 重新理解：RpEI应该是超额收益的年化收益率
    # 可以通过计算EI的累计变化，然后年化
    EI_total_return = EI[-1] - EI[0]  # EI的累计变化
    RpEI = ((1 + abs(EI_total_return)) ** (TRADING_DAYS_PER_YEAR / n) - 1) if n > 0 and EI_total_return >= 0 else -((1 + abs(EI_total_return)) ** (TRADING_DAYS_PER_YEAR / n) - 1)
    
    # 或者：基于EI的日变化率计算年化收益率
    # 计算EI的日变化率（类似净值收益率）
    mask = np.abs(1 + EI_prev[:-1]) > 1e-10
    EI_returns_normalized = np.zeros_like(EI_returns)
    EI_returns_normalized[mask] = EI_returns[mask] / (1 + EI_prev[:-1][mask])
    EI_returns_normalized[~mask] = EI_returns[~mask]  # 如果分母接近0，使用绝对值
    
    # 使用日变化率的均值年化作为RpEI
    RpEI = np.mean(EI_returns_normalized) * TRADING_DAYS_PER_YEAR
    
    # 计算EI的日变化率的标准差（年化）
    σpEI = np.std(EI_returns_normalized, ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR)
    
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
