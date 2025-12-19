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
    df['平仓盈亏'] = pd.to_numeric(df['平仓盈亏'], errors='coerce').fillna(0)
    df['手续费'] = pd.to_numeric(df['手续费'], errors='coerce').fillna(0)
    df['净盈亏'] = df['平仓盈亏'] - df['手续费']
    
    # 3. 初始资金
    initial_capital = 1000000
    
    # 4. 按日期聚合,pnl是profit and loss的缩写，表示盈利和亏损
    df['日期_仅'] = pd.to_datetime(df['日期'], format='%Y-%m-%d', errors='coerce')
    daily_pnl = df.groupby('日期_仅')['净盈亏'].sum().reset_index()
    daily_pnl.columns = ['日期', '日盈亏']
    daily_pnl = daily_pnl.sort_values('日期').reset_index(drop=True)
    daily_pnl['累计收益'] = daily_pnl['日盈亏'].cumsum()
    
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


print(load_and_process_data())