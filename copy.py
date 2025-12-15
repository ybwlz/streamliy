"""
Streamlit 期货数据分析应用
支持 IF/IM/IC/IH 标的的分钟级和日级数据分析
主图：K线图（蜡烛图，显示开高低收）
副图：RSI/KDJ 技术指标
"""

import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib支持中文显示
import platform
system = platform.system()
if system == 'Windows':
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong']
elif system == 'Darwin':  # macOS
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'STHeiti']
else:  # Linux
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置页面配置
st.set_page_config(
    page_title="期货技术指标分析",
    page_icon="📈",
    layout="wide"
)

# 标的代码映射
SYMBOL_MAP = {
    "IF": "IF0",  # 沪深300主力连续
    "IM": "IM0",  # 中证1000主力连续
    "IC": "IC0",  # 中证500主力连续
    "IH": "IH0"   # 上证50主力连续
}

def get_daily_data(symbol: str) -> pd.DataFrame:
    """
    获取日级期货数据
    :param symbol: 合约代码，如 "IF0"
    :return: DataFrame
    """
    try:
        df = ak.futures_zh_daily_sina(symbol=symbol)
        if df is None or df.empty:
            return None
        # 确保日期列为datetime类型
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        # 确保数值列为数字类型
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.sort_index()
        return df
    except Exception as e:
        st.error(f"获取日级数据失败: {str(e)}")
        return None

def get_minute_data(symbol: str, period: str = "1") -> pd.DataFrame:
    """
    获取分钟级期货数据
    :param symbol: 合约代码，如 "IF0"
    :param period: 周期，"1"表示1分钟
    :return: DataFrame
    """
    try:
        df = ak.futures_zh_minute_sina(symbol=symbol, period=period)
        if df is None or df.empty:
            return None
        # 确保日期列为datetime类型
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.set_index('datetime')
        # 确保数值列为数字类型
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.sort_index()
        return df
    except Exception as e:
        st.error(f"获取分钟级数据失败: {str(e)}")
        return None

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    计算RSI指标
    :param prices: 价格序列（收盘价）
    :param period: 周期，默认14
    :return: RSI序列
    """
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)  # 避免除0
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_kdj(df: pd.DataFrame, n: int = 9, m1: int = 3, m2: int = 3):
    """
    计算KDJ指标
    :param df: 包含 high, low, close 的DataFrame
    :param n: 周期，默认9
    :param m1: K值平滑周期，默认3
    :param m2: D值平滑周期，默认3
    :return: K, D, J 序列
    """
    low_list = df['low'].rolling(window=n).min()
    high_list = df['high'].rolling(window=n).max()
    
    rsv = (df['close'] - low_list) / (high_list - low_list + 1e-10) * 100
    k = rsv.ewm(span=m1).mean()
    d = k.ewm(span=m2).mean()
    j = 3 * k - 2 * d
    
    return k, d, j

def filter_data_by_date(df: pd.DataFrame, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    根据日期范围筛选数据
    :param df: 原始数据
    :param start_date: 开始日期
    :param end_date: 结束日期
    :return: 筛选后的数据
    """
    mask = (df.index >= start_date) & (df.index <= end_date)
    return df[mask]

def plot_charts(df: pd.DataFrame, symbol: str, data_type: str):
    """
    绘制主图和副图
    :param df: 数据DataFrame
    :param symbol: 标的代码
    :param data_type: 数据类型（"分钟级" 或 "日级"）
    """
    if df is None or df.empty:
        st.warning("数据为空，无法绘制图表")
        return
    
    # 计算技术指标
    df['RSI'] = calculate_rsi(df['close'])
    df['K'], df['D'], df['J'] = calculate_kdj(df)
    
    # 创建图表
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # ========== 主图：K线图（蜡烛图）==========
    # 准备数据
    dates = df.index
    opens = df['open']
    highs = df['high']
    lows = df['low']
    closes = df['close']
    
    # 将日期转换为matplotlib可以识别的格式
    import matplotlib.dates as mdates
    dates_num = mdates.date2num(dates)
    
    # K线实体宽度（根据数据类型调整）
    if data_type == "日级":
        width = 0.8  # 日级数据，宽度0.8天
    else:
        # 分钟级数据，计算平均时间间隔
        if len(dates) > 1:
            avg_interval = (dates_num[-1] - dates_num[0]) / len(dates)
            width = avg_interval * 0.6  # 宽度为平均间隔的60%
        else:
            width = 1/1440  # 默认1分钟
    
    # 绘制每根K线
    for i, (date_num, date, open_price, high, low, close) in enumerate(zip(dates_num, dates, opens, highs, lows, closes)):
        # 判断涨跌：收盘价 >= 开盘价为上涨（红色），否则为下跌（绿色）
        color = 'red' if close >= open_price else 'green'
        
        # 绘制影线（上下影线）：从最低价到最高价的垂直线
        axes[0].plot([date_num, date_num], [low, high], color='black', linewidth=0.8, alpha=0.6)
        
        # 绘制实体（矩形）：开盘价和收盘价之间的矩形
        body_bottom = min(open_price, close)
        body_height = abs(close - open_price)
        
        # 如果开盘价等于收盘价（十字星），画一条横线
        if body_height < 0.0001:
            axes[0].plot([date_num-width/2, date_num+width/2], [close, close], color=color, linewidth=2)
        else:
            # 绘制矩形实体
            rect = plt.Rectangle((date_num-width/2, body_bottom), width, body_height, 
                               facecolor=color, edgecolor='black', linewidth=0.5, alpha=0.8)
            axes[0].add_patch(rect)
    
    # 设置x轴日期格式
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d' if data_type == "日级" else '%m-%d %H:%M'))
    axes[0].xaxis.set_major_locator(mdates.AutoDateLocator())
    
    axes[0].set_title(f'{symbol} {data_type}K线图', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('价格', fontsize=12)
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.8, edgecolor='black', label='上涨（红）'),
        Patch(facecolor='green', alpha=0.8, edgecolor='black', label='下跌（绿）')
    ]
    axes[0].legend(handles=legend_elements, loc='best')
    axes[0].grid(True, alpha=0.3)
    
    # 副图1：RSI指标
    axes[1].plot(df.index, df['RSI'], label='RSI(14)', color='orange', linewidth=1.5)
    axes[1].axhline(y=70, color='red', linestyle='--', alpha=0.7, label='超买线(70)')
    axes[1].axhline(y=30, color='green', linestyle='--', alpha=0.7, label='超卖线(30)')
    axes[1].fill_between(df.index, 30, 70, alpha=0.1, color='gray')
    axes[1].set_ylabel('RSI', fontsize=12)
    axes[1].set_ylim(0, 100)
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    
    # 副图2：KDJ指标
    axes[2].plot(df.index, df['K'], label='K', color='blue', linewidth=1.5)
    axes[2].plot(df.index, df['D'], label='D', color='red', linewidth=1.5)
    axes[2].plot(df.index, df['J'], label='J', color='purple', linewidth=1.5)
    axes[2].axhline(y=80, color='red', linestyle='--', alpha=0.3)
    axes[2].axhline(y=20, color='green', linestyle='--', alpha=0.3)
    axes[2].set_ylabel('KDJ', fontsize=12)
    axes[2].set_xlabel('时间', fontsize=12)
    axes[2].legend(loc='best')
    axes[2].grid(True, alpha=0.3)
    
    # 格式化x轴日期（所有子图共享x轴，只需在最后一个子图设置）
    import matplotlib.dates as mdates
    if data_type == "日级":
        axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    else:
        axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    
    # 旋转x轴标签
    for ax in axes:
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    
    # 显示图表
    st.pyplot(fig)
    plt.close()

# 主程序
def main():
    st.title("📈 期货技术指标分析系统")
    st.markdown("---")
    
    # 侧边栏：参数设置
    with st.sidebar:
        st.header("⚙️ 参数设置")
        
        # 标的选择
        symbol_option = st.selectbox(
            "选择标的",
            options=["IF", "IM", "IC", "IH"],
            index=0,
            help="IF: 沪深300, IM: 中证1000, IC: 中证500, IH: 上证50"
        )
        
        # 数据类型选择
        data_type = st.radio(
            "数据类型",
            options=["日级", "分钟级"],
            index=0,
            help="选择数据的时间粒度"
        )
        
        # 日期选择
        if data_type == "日级":
            # 日级数据：选择日期范围
            end_date = st.date_input(
                "结束日期",
                value=datetime.now().date(),
                max_value=datetime.now().date()
            )
            start_date = st.date_input(
                "开始日期",
                value=end_date - timedelta(days=30),
                max_value=end_date
            )
        else:
            # 分钟级数据：选择日期和时间
            selected_date = st.date_input(
                "选择日期",
                value=datetime.now().date(),
                max_value=datetime.now().date()
            )
            start_date = selected_date
            end_date = selected_date
        
        # 数据量限制提示
        if data_type == "分钟级":
            st.info("⚠️ 分钟级数据量较大，建议选择单日数据")
        
        # 刷新按钮
        if st.button("🔄 刷新数据", type="primary"):
            st.rerun()
    
    # 主内容区
    symbol_code = SYMBOL_MAP[symbol_option]
    
    # 显示加载提示
    with st.spinner(f"正在获取 {symbol_option} {data_type}数据..."):
        # 获取数据
        if data_type == "日级":
            df = get_daily_data(symbol_code)
        else:
            df = get_minute_data(symbol_code)
        
        if df is None or df.empty:
            st.error(f"❌ 无法获取 {symbol_option} 的{data_type}数据，请检查网络连接或稍后重试")
            return
        
        # 日期筛选
        try:
            start_datetime = pd.to_datetime(start_date)
            end_datetime = pd.to_datetime(end_date)
            if data_type == "日级":
                end_datetime = end_datetime.replace(hour=23, minute=59, second=59)
            else:
                end_datetime = end_datetime.replace(hour=23, minute=59, second=59)
            
            df_filtered = filter_data_by_date(df, start_datetime, end_datetime)
            
            if df_filtered.empty:
                st.warning(f"⚠️ 在选择的日期范围内没有数据")
                return
            
        except Exception as e:
            st.error(f"日期筛选失败: {str(e)}")
            df_filtered = df
    
    # 显示数据信息
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("数据点数", len(df_filtered))
    with col2:
        if not df_filtered.empty:
            st.metric("最新价格", f"{df_filtered['close'].iloc[-1]:.2f}")
    with col3:
        if not df_filtered.empty:
            st.metric("最高价", f"{df_filtered['high'].max():.2f}")
    with col4:
        if not df_filtered.empty:
            st.metric("最低价", f"{df_filtered['low'].min():.2f}")
    
    st.markdown("---")
    
    # 绘制图表
    plot_charts(df_filtered, symbol_option, data_type)
    
    # 显示数据表格（可选）
    with st.expander("📊 查看原始数据"):
        st.dataframe(df_filtered[['open', 'high', 'low', 'close', 'volume', 'RSI', 'K', 'D', 'J']].tail(100))

if __name__ == "__main__":
    main()

