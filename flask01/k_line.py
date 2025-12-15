"""

接口: futures_zh_daily_sina，该接口使用英文参数，不需要重命名列

目标地址: https://finance.sina.com.cn/futures/quotes/V2105.shtml

描述: 新浪财经-期货-日频数据

限量: 单次返回指定 symbol 的所有日频数据; 期货连续合约为 品种代码+0，比如螺纹钢连续合约为 RB0;

输入参数

名称	类型	描述
symbol	str	symbol="RB0"; 具体合约可以通过 ak.match_main_contract(symbol="shfe") 获取或者访问网页
输出参数

名称	类型	描述
date	object	-
open	float64	开盘价
high	float64	最高价
low	    float64	最低价
close	float64	收盘价
volume	int64	成交量
hold	int64	持仓量
settle	float64	结算价

"""


import akshare as ak
import mplfinance as mpf
import pandas as pd
from matplotlib import pyplot as plt


# 步骤1：获取K线数据
def get_futures_data():
    """
    从AKShare获取股票日K线数据
    """
    df = ak.futures_zh_daily_sina(symbol="IF2401")#如果是别的接口可能需要重命名中文参数IF0为连续合约，也可使用IF2512
    try:
        if df is None or df.empty:
            print("获取数据失败")
            exit(1)
        return df
    except Exception as e:
        print(f"获取数据时出错: {str(e)[:100]}")
        return None

df = get_futures_data()

print("原始数据列名:", df.columns.tolist())
print("数据前6行:")
print(df.head(6))



# 步骤2：数据清洗和格式化


# 清洗数据
def clean_data(df):
    """
    清洗数据，转换为mplfinance需要的格式
    """
    # 确保日期列为datetime类型并设为索引
    df['date'] = pd.to_datetime(df['date']) 
    df = df.set_index('date')  
    
    # 确保数值列是数字类型
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']  
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 按日期排序
    df = df.sort_index()
    
    return df

df_clean = clean_data(df)

# 步骤3：计算技术指标函数
def calculate_rsi(prices, period=14):
    """计算RSI指标"""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()#收盘价上涨幅度平均值（滚动平均）where条件筛选，a,b不满足设为b，rolling滑动窗口，求windows内的平均值。
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()#下跌平均值，-delta是取绝对值，取的是delta.where(delta < 0,0)的相反数，rolling向量化操作，结果是组数据
    rs = gain / (loss + 1e-10)#避免除0报错
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_kdj(df, n=9, m1=3, m2=3):
    """计算KDJ指标"""
    low_list = df['low'].rolling(window=n).min()
    high_list = df['high'].rolling(window=n).max()
    
    rsv = (df['close'] - low_list) / (high_list - low_list + 1e-10) * 100
    k = rsv.ewm(span=m1).mean()#ewm指数移动平均，小于span的是NAN，用历史数据默认向前滑动
    d = k.ewm(span=m2).mean()
    j = 3 * k - 2 * d
    
    return k, d, j

df_clean['RSI'] = calculate_rsi(df_clean['close'])
df_clean['K'], df_clean['D'], df_clean['J'] = calculate_kdj(df_clean)
print(df_clean.columns.tolist())
print(df_clean.tail(6))#查看计算的指标值


#步骤4：作图
def plot_with_mplfinance(df):
    """
    使用mplfinance绘制K线图
    """
    # 创建自定义样式
    mc = mpf.make_marketcolors(
        up='red',      # 上涨为红色
        down='green',  # 下跌为绿色
        edge='inherit',
        wick='inherit',
        volume='in'
    )
    
    style = mpf.make_mpf_style(
        marketcolors=mc,
        gridstyle=':',  # 网格线样式,还有-等
        gridcolor='lightgray',
        facecolor='white'
    )
    
    # 准备副图数据
    apds = [
        # RSI副图
        mpf.make_addplot(df['RSI'], panel=1, color='orange', ylabel='RSI'),#面板，颜色，ylabel轴名
        # KDJ副图（K线）
        mpf.make_addplot(df['K'], panel=2, color='blue', ylabel='KDJ'),
        # KDJ副图（D线）
        mpf.make_addplot(df['D'], panel=2, color='red'),
        # KDJ副图（J线）
        mpf.make_addplot(df['J'], panel=2, color='purple'),
    ]
    
    # 绘制图表
    fig, axes = mpf.plot(
        df,
        type='candle',           # K线类型：蜡烛图
        style=style,            # 样式
        addplot=apds,          # 副图指标
        volume=False,           # 不显示成交量
        title='IF-K-line picture',  # 标题
        ylabel='Price',          # Y轴标签
        #ylabel_lower='Trading volume',   # 成交量Y轴标签
        figratio=(12, 8),      # 图表比例
        figscale=1.2,          # 图表缩放
        returnfig=True         # 返回figure对象以便进一步定制
    )
    
    # 添加RSI参考线
    axes[1].axhline(y=70, color='red', linestyle='--', alpha=0.7)
    axes[1].axhline(y=30, color='green', linestyle='--', alpha=0.7)
    axes[1].legend(['RSI(14)', '超买线(70)', '超卖线(30)'])
    
    # 添加KDJ图例
    axes[2].legend(['K', 'D', 'J'])
    
    # 调整布局
    plt.tight_layout()
    plt.show()
    
# 绘制图表
plot_with_mplfinance(df_clean.tail(30))  # 显示一个月的个交易日

