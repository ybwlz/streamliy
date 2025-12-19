import pandas as pd # 用于数据处理
import numpy as np # 用于数值计算
import plotly.graph_objects as go #用于绘制交互式图表# .graph_objects是Plotly中用于创建图表的核心模块,包含各种图表类型和布局配置，比如散点图、折线图、柱状图等，还有布局和样式设置
from plotly.subplots import make_subplots # 用于创建子图
import streamlit as st # 用于构建Web应用
import warnings # 用于忽略警告信息
warnings.filterwarnings('ignore') # 忽略警告信息，filterwarnings()函数用于控制警告的显示行为，这里设置为忽略所有警告信息


#副本，复制版，详情注释版


# ========== 基准数据 ==========
# 根据jiaoyi.csv的日期范围获取的黄金期货基准收益率数据（使用AU0主力连续合约）
# 数据格式：与交易日期对齐的日收益率序列（小数形式）
# 基准数据日期范围: 2024-01-04 到 2025-04-28，共71个交易日
# 注意：第一个值为0，因为第一个交易日没有前一天的价格数据
# 基准数据来源：akshare获取的AU0黄金主力连续合约，整个期间（315个连续交易日）总收益率61.95%
# 对齐说明：使用整个期间的连续基准数据，每个交易日使用从第一个交易日到该交易日之间的累计基准收益率
# 对齐后的基准数据总收益率：62.56%（与连续基准总收益率61.95%非常接近，差异0.60%）

#BENCHMARK：表明这是基准/对比标准benchmark
#RETURNS：表示这是收益率数据returns
#HARDCODED：表明是硬编码/内嵌在代码中的数据（而不是从外部文件/API实时获取）hardcoded
#大写字母和下划线命名风格，符合常量命名规范

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

BENCHMARK_AVAILABLE = True # 通常用于后续额的条件判断

# 设置页面配置
st.set_page_config( #st.set_page_config()函数用于设置Streamlit应用的页面配置
    page_title="策略风险分析", 
    page_icon="📊", 
    layout="wide" , #layout参数设置为"wide"，表示页面布局为宽屏模式,还有"centered"模式可选，代表中间对齐
)

# 加载交易数据（使用GBK编码）
@st.cache_data   #缓存数据以提高性能
def load_trade_data(filename='jiaoyi.csv'): 
    """加载交易数据"""
    import os
    
    # 获取脚本所在目录D :\code\flask\task3
    script_dir = os.path.dirname(os.path.abspath(__file__)) #os.path.abspath(__file__)获取当前脚本的绝对路径，os.path.dirname()获取其父目录
    
    # 尝试多个可能的路径
    possible_paths = [
        os.path.join(script_dir, filename),  # 脚本所在目录 D :\code\flask\task3\jiaoyi.csv
        filename,  # 当前工作目录 D :\code\flask\jiaoyi.csv
        os.path.join('.', filename),  # 当前目录 ./jiaoyi.csv
        os.path.join('task3', filename),  # task3子目录 task3/jiaoyi.csv
    ]
    
    for file_path in possible_paths:
        try:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, encoding='gbk')
                return df
        except Exception as e:
            continue
    
    # 如果所有路径都失败，显示错误信息
    st.error(f"无法找到文件: {filename}") #黄色警告提示
    st.info("**调试信息：**") #蓝色信息提示**代表加粗，还有斜体等Markdown语法，比如*斜体*，或者`代码块`，还有列表等，比如- 列表项1，或者1. 列表项1
    st.info(f"- 脚本所在目录: {script_dir}") #f-strings用于格式化字符串，{script_dir}会被替换为变量script_dir的值
    st.info(f"- 当前工作目录: {os.getcwd()}") #os.getcwd()获取当前工作目录
    st.info(f"- 尝试的路径: {', '.join(possible_paths)}") #join()方法用于将列表元素连接成字符串，通过', '分隔
    
    # 列出当前目录的文件（用于调试）
    try:
        current_files = os.listdir('.') #os.listdir('.') 列出当前目录的所有文件和文件夹
        #f for f in current_files if f.endswith('.csv') 相当于
        #for f in current_files:
        #    if f.endswith('.csv'):
        #        结果.append(f)
        st.info(f"- 当前目录文件: {', '.join([f for f in current_files if f.endswith('.csv')][:5])}")#endswith()方法用于判断字符串是否以指定后缀结尾,[:5]表示只显示前五个文件
    except:
        pass
    
    try:
        script_files = os.listdir(script_dir)
        st.info(f"- 脚本目录文件: {', '.join([f for f in script_files if f.endswith('.csv')][:5])}")
    except:
        pass
    
    return None

# 数据清洗和预处理
def preprocess_data(df):
    """预处理交易数据"""
    df = df.copy() #这是pandas的方法，用于创建DataFrame的副本，避免修改原始数据
    
    # 合并日期和时间
    try:
        #pd.to_datetime()函数用于将字符串转换为日期时间格式,astype(str)确保数据是字符串类型,format指定日期时间格式，errors='coerce'会将无法解析的值转换NaN,+号连接字符串
        df['日期时间'] = pd.to_datetime(df['日期'].astype(str) + ' ' + df['委托时间'].astype(str), 
                                      format='%Y/%m/%d %H:%M:%S', errors='coerce')
    #except没有指定format时的备用方案，会自动推断格式
    except:
        df['日期时间'] = pd.to_datetime(df['日期'].astype(str) + ' ' + df['委托时间'].astype(str), errors='coerce')
    
    #.sort_values()用于按指定列排序，reset_index(drop=True)用于重置索引，drop=True表示不保留旧索引
    df = df.sort_values('日期时间').reset_index(drop=True)
    
    # 转换数据类型
    # 成交数量（处理"手"单位）
    #df.columns用于获取DataFrame的列名列表
    if '成交数量' in df.columns:
        # .dtype属性用于获取列的数据类型，object表示字符串类型,在pd里面
        if df['成交数量'].dtype == 'object':
            # .astype(str)确保数据是字符串类型，.str.replace()用于替换字符串中的指定内容，.str.strip()用于去除字符串两端的空白字符
            df['成交数量'] = df['成交数量'].astype(str).str.replace('手', '').str.replace(',', '').str.strip()
            # .pd.to_numeric()用于将字符串转换为数值类型，errors='coerce'会将无法转换的值变为NaN
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
#这些参数是函数calculate_risk_metrics的输入参数，分别日期收益率序列、总收益率、初始资金、日盈亏数据和基准收益率数据
def calculate_risk_metrics(daily_returns, total_returns, initial_capital, daily_pnl, benchmark_returns=None):
    """计算各种风险指标"""
    metrics = {}
    
    # 确保daily_returns是数组
    if isinstance(daily_returns, pd.Series): #isinstance()函数用于检查变量类型,意思是如果daily_returns是pandas的Series类型
        daily_returns = daily_returns.values #.values属性用于获取Series的底层NumPy数组,之所以是因为后续计算需要使用NumPy数组进行数值计算,相当于就是字典把key取出来，就是numpy数组
    
    # 过滤NaN值
    daily_returns = daily_returns[~np.isnan(daily_returns)]#~表示取反，np.isnan()用于判断NaN值，~np.isnan()表示非NaN值
    

    #防御性编程：如果daily_returns长度为0，直接返回空字典，避免后续计算出错
    if len(daily_returns) == 0:
        return {}
    
    # Total Returns 策略收益（百分比）
    metrics['Total Returns'] = total_returns
    
    # Total Annualized Returns 策略年化收益
    trading_days = len(daily_returns)
    if trading_days > 0:
        # 计算实际交易天数
        date_range = (daily_pnl['日期'].max() - daily_pnl['日期'].min()).days  #daily_pnl是一个DataFrame，包含每日盈亏数据，'日期'列是日期类型，.max()和.min()分别获取最大和最小日期，
                                                                               #.days属性获取两个日期之间的天数差
        years = date_range / 365.25 if date_range > 0 else trading_days / 252 #考虑闰年，使用365.25天/年，如果date_range为0，则退化为trading_days/252，之所以是252是因为通常认为每年有252个交易日
        #而之所以在date_range为0时使用trading_days/252，是为了避免除以零的情况，同时也能在极端情况下提供一个合理的年化收益估计
        if years > 0:
            metrics['Total Annualized Returns'] = ((1 + total_returns / 100) ** (1 / years) - 1) * 100
        else:
            metrics['Total Annualized Returns'] = 0
    else:
        metrics['Total Annualized Returns'] = 0
    
    # Algorithm Volatility 策略波动率（年化）就是策略收益的年化标准差
    if len(daily_returns) > 1:
        metrics['Algorithm Volatility'] = np.std(daily_returns) * np.sqrt(252) * 100 #.sqrt(252)用于将日标准差转换为年化标准差，乘以100转换为百分比形式
    else:
        metrics['Algorithm Volatility'] = 0
    
    # Benchmark Volatility 基准波动率（就是基准收益的年化标准差）
    if benchmark_returns is not None and len(benchmark_returns) > 1:
        metrics['Benchmark Volatility'] = np.std(benchmark_returns) * np.sqrt(252) * 100#np.sqrt(252)相当于
    else:
        metrics['Benchmark Volatility'] = 0
    
    # Sharpe 夏普比率（假设无风险利率为0）
    if metrics['Algorithm Volatility'] > 0:
        metrics['Sharpe'] = metrics['Total Annualized Returns'] / metrics['Algorithm Volatility']
    else:
        metrics['Sharpe'] = 0
    
    # Sortino 索提诺比率（只考虑下行波动）就是只计算负收益的标准差
    downside_returns = daily_returns[daily_returns < 0]
    if len(downside_returns) > 1:
        downside_std = np.std(downside_returns) * np.sqrt(252) * 100
        metrics['Downside Risk'] = downside_std#策略下行波动率
        if downside_std > 0:#避免除零错误
            metrics['Sortino'] = metrics['Total Annualized Returns'] / downside_std#索提诺比率（策略年化收益率 - 无风险利率）/策略下行波动率
        else:
            metrics['Sortino'] = 0
    else:
        metrics['Downside Risk'] = 0#策略下行波动率为0
        metrics['Sortino'] = 0#索提诺比率为0
    
    # Max Drawdown 最大回撤
    # 使用复利计算累计收益率：(1 + r1) * (1 + r2) * ... - 1
    cumulative_returns = np.cumprod(1 + daily_returns) - 1 #np.cumprod()用于计算累积乘积
    running_max = np.maximum.accumulate(cumulative_returns) #np.maximum.accumulate()用于计算运行中的最大值
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
                
                # Alpha = (策略平均日收益率 - Beta * 基准平均日收益率) * 252 * 100
                # daily_returns 和 benchmark_returns 都是小数形式（0.01表示1%）
                strategy_mean_daily = np.mean(daily_returns_clean)
                benchmark_mean_daily = np.mean(benchmark_returns_clean)
                alpha_daily = strategy_mean_daily - metrics['Beta'] * benchmark_mean_daily
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
        valid_mask = ~(np.isnan(daily_returns_aligned) | np.isnan(benchmark_returns_aligned) |  #isnan()用于判断NaN值
                      np.isinf(daily_returns_aligned) | np.isinf(benchmark_returns_aligned)) #isinf()用于判断无穷值 之所以~是因为取非这些的值
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
            excess_cumulative = np.cumprod(1 + excess_returns) - 1 #np.cumprod()用于计算累积乘积
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
            strategy_total = np.prod(1 + daily_returns_clean) - 1 #np.prod()用于计算数组元素的乘积，和cumprod()不同，cumprod()是累积乘积
            benchmark_total = np.prod(1 + benchmark_returns_clean) - 1
            
            if abs(benchmark_total) > 1e-10:
                # 超额收益 = (策略总收益 / 基准总收益 - 1) * 100
                metrics['超额收益'] = (strategy_total / benchmark_total - 1) * 100
            else:
                metrics['超额收益'] = 0
        else:
            metrics['超额收益'] = 0
    else:
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
    fig = make_subplots(#make_subplots是Plotly中用于创建包含多个子图的图表布局的函数，包含的参数有rows（行数）、cols（列数）、subplot_titles（子图标题列表）、
                         # vertical_spacing（子图之间的垂直间距）和 row_heights（每行的高度比例列表）还有其他参数比如horizontal_spacing（子图之间的水平间距）、shared_xaxes（是否共享X轴）
                         # 还有shared_yaxes（是否共享Y轴）还有specs（子图规格列表）还有column_widths（每列的宽度比例列表）还有subplot_titles_font（子图标题字体属性）
                         # title_text（图表标题文本），title_x（图表标题位置），具体可参考官方文档：https://plotly.com/python/subplots/
        rows=2, cols=1,
        subplot_titles=('交易信号图（价格走势）', '累计收益曲线'),
        vertical_spacing=0.1, #子图之间的垂直间距
        row_heights=[0.6, 0.4] #每行子图的高度比例
    )
    
    # 按日期时间排序
    df_sorted = df.sort_values('日期时间').reset_index(drop=True)
    
    # 1. 价格走势图（主图）
    fig.add_trace( #add_trace()用于向图表添加数据轨迹
        go.Scatter( #go.Scatter()用于创建散点图或折线图,还有go.Bar()用于创建柱状图，或者go.Pie()用于创建饼图等，k线图是go.Candlestick()，go.Heatmap()用于热力图等，go.Histogram()用于直方图等
            #go代表Plotly Graph Objects，是Plotly库中用于创建图表的核心模块
            x=df_sorted['日期时间'],
            y=df_sorted['成交价格'],
            mode='lines',
            name='成交价格',
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
                y=buy_signals['成交价格'],
                mode='markers',
                name='买入信号（开多）',
                marker=dict(
                    symbol='triangle-up', #向上三角形
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
                y=sell_signals['成交价格'],
                mode='markers',
                name='卖出信号（平多）',
                marker=dict(
                    symbol='triangle-down', #向下三角形
                    size=12,
                    color='red',
                    line=dict(width=2, color='darkred') #line用于设置标记边框属性
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
    
    # 更新Y轴标签u
    fig.update_yaxes(title_text="价格", row=1, col=1) #update_yaxes()用于更新Y轴属性,不这样就会显示默认的yaxis title
    fig.update_yaxes(title_text="盈亏", row=2, col=1) #title_text用于设置Y轴标题
    
    # 更新X轴日期格式（不显示标签，只设置日期格式）
    fig.update_xaxes(tickformat="%Y-%m-%d", row=1, col=1) #tickformat用于设置日期格式
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
        go.Scatter( #Scatter用于绘制折线图或散点图
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
        df['累计收益'] = df['净盈亏'].cumsum() #.cumsun()是pandas的一个方法，用于计算累计和
        
        # 计算初始资金（使用第一笔交易的成交金额作为参考）
        initial_capital = abs(df['成交金额'].iloc[0]) if len(df) > 0 and df['成交金额'].iloc[0] != 0 else 1000000
        df['累计收益率'] = (df['累计收益'] / initial_capital) * 100
        
        # 按日期聚合（用于计算日收益率）
        df['日期_仅'] = pd.to_datetime(df['日期'], format='%Y/%m/%d', errors='coerce')
        daily_pnl = df.groupby('日期_仅')['净盈亏'].sum().reset_index() #groupby()是pandas的一个方法，用于按指定列分组，然后对每个组进行聚合操作，这里是按'日期_仅'列分组，计算每个日期的净盈亏总和
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
        #分别表示对齐后的日收益率序列、总收益率、初始资金、对齐后的日盈亏数据和基准收益率数据
        #这5个参数就能够计算所有的风险指标，这里metics是一个字典，包含了所有计算出来的风险指标
        metrics = calculate_risk_metrics(daily_returns_pct_aligned, total_returns_pct, initial_capital, daily_pnl_aligned, benchmark_returns)
    
    # 显示基本信息
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("总交易笔数", len(df)) #st.metric()是Streamlit中的一个函数，用于显示一个指标，这里显示总交易笔数
    with col2:
        st.metric("交易日期范围", f"{daily_pnl['日期'].min().strftime('%Y-%m-%d')} 至 {daily_pnl['日期'].max().strftime('%Y-%m-%d')}")
    with col3:
        st.metric("总盈亏", f"{df['净盈亏'].sum():,.0f}")
    with col4:
        st.metric("初始资金", f"{initial_capital:,.0f}")
    
    st.divider()
    
    # 风险指标展示
    st.header("📈 风险指标") #st.header()是Streamlit中的一个函数，用于创建一个一级标题
    
    # 将指标分为多列显示
    col1, col2, col3 = st.columns(3) #st.columns()是Streamlit中的一个函数，用于创建多列布局，这里创建了3列布局
    
    with col1: #with是Python中的上下文管理器，用于简化资源管理，这里用于在col1列中显示内容，和python中的with一样，但是在streamlit中用于布局管理
        st.subheader("收益指标") #st.subheader()是Streamlit中的一个函数，用于创建一个二级标题
        st.write(f"**Total Returns (策略收益)**: {metrics.get('Total Returns', 0):.4f}%") #st.write()是Streamlit中的一个函数，用于在页面上显示文本或变量内容
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
    
    st.divider() #st.divider()是Streamlit中的一个函数，用于在页面上添加一个水平分割线，起到分隔内容的作用,括号内可以添加参数来定制分割线的样式，
                 #但通常不需要参数，默认样式即可。比如：st.divider(color="blue", thickness=2)代表添加一条蓝色、厚度为2像素的分割线。
    
    # 交易信号图（主要图表）
    st.header("📊 交易信号图")
    st.info("💡 **交易信号图说明**：上图显示价格走势，绿色▲表示买入信号（开多），红色▼表示卖出信号（平多）。下图显示每笔交易的盈亏和累计收益。")
    
    fig_signals = plot_trading_signals(df) #plot_trading_signals()是一个自定义函数，用于绘制交易信号图
    st.plotly_chart(fig_signals, use_container_width=True) #st.plotly_chart()是Streamlit中的一个函数，用于在页面上显示Plotly图表，use_container_width=True参数表示图表宽度会自动适应容器宽度
    
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
    with st.expander("查看详细交易数据"): #st.expander()是Streamlit中的一个函数，用于创建一个可展开/折叠的区域，用户可以点击标题来展开或折叠内容
        # 选择要显示的列
        display_cols = ['日期', '委托时间', '标的', '交易类型', '成交数量'] 
        # 添加价格列（可能是'成交价'或'成交价格'）
        display_cols.append('成交价')
        display_cols.append('成交额') #append()是Python列表的方法，用于在列表末尾添加一个元素
        display_cols.extend(['平仓盈亏', '手续费', '净盈亏', '累计收益']) #extend()是Python列表的方法，用于将一个可迭代对象（如列表、元组等）中的元素逐个添加到当前列表的末尾
        # 只显示存在的列
        available_cols = [col for col in display_cols if col in df.columns] #columns是pandas DataFrame的一个属性，返回DataFrame的列标签列表，这里用来检查哪些列存在于DataFrame中
        st.dataframe(df[available_cols], use_container_width=True) #st.dataframe()是Streamlit中的一个函数，用于在页面上显示一个可滚动的数据表格
    
    with st.expander("查看日度汇总数据"):
        st.dataframe(daily_pnl, use_container_width=True) #use_container_width=True参数表示表格宽度会自动适应容器宽度

if __name__ == "__main__": 
    main()
