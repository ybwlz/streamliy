"""
Streamlit 期货数据分析应用
- K线 + RSI + KDJ
- 10年数据查询
- Plotly图表
"""

import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

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

# 数据缓存
@st.cache_data(ttl=60 * 30)#30分钟
def get_daily_data(symbol: str) -> pd.DataFrame:
    try:
        df = ak.futures_zh_daily_sina(symbol=symbol)
        if df is None or df.empty:
            return None
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.sort_index()
        return df
    except Exception as e:
        st.error(f"获取日级数据失败: {str(e)}")
        return None

# 缓存
@st.cache_data(ttl=60 * 5)#5分钟刷新一次
def get_minute_data(symbol: str, period: str = "1") -> pd.DataFrame:
    try:
        df = ak.futures_zh_minute_sina(symbol=symbol, period=period)
        if df is None or df.empty:
            return None
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.set_index('datetime')
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.sort_index()
        return df
    except Exception as e:
        st.error(f"获取分钟级数据失败: {str(e)}")
        return None

# 指标计算（RSI+KDJ）
def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_kdj(df: pd.DataFrame, n: int = 9, m1: int = 3, m2: int = 3):
    low_list = df['low'].rolling(window=n).min()
    high_list = df['high'].rolling(window=n).max()
    rsv = (df['close'] - low_list) / (high_list - low_list + 1e-10) * 100
    k = rsv.ewm(span=m1).mean()
    d = k.ewm(span=m2).mean()
    j = 3 * k - 2 * d
    return k, d, j

def filter_data_by_date(df: pd.DataFrame, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    mask = (df.index >= start_date) & (df.index <= end_date)
    return df[mask]

def filter_trading_hours(df: pd.DataFrame) -> pd.DataFrame:
    """
    过滤非交易时间（仅对分钟级数据有效）
    中国期货交易时间：
    - 日盘：09:00-11:30, 13:30-15:00
    - 夜盘：21:00-02:30（次日）
    """
    if df is None or df.empty:
        return df
    
    df = df.copy()
    
    # 提取小时和分钟
    hours = df.index.hour
    minutes = df.index.minute
    time_total = hours * 60 + minutes  # 转换为总分钟数，便于比较
    
    # 定义交易时间段（以分钟为单位）
    # 日盘上午：09:00-11:30 (540-690分钟)
    morning_start = 9 * 60 + 0   # 09:00
    morning_end = 11 * 60 + 30  # 11:30
    
    # 日盘下午：13:30-15:00 (810-900分钟)
    afternoon_start = 13 * 60 + 30  # 13:30
    afternoon_end = 15 * 60 + 0     # 15:00
    
    # 夜盘：21:00-23:59 (1260-1439分钟) 和 00:00-02:30 (0-150分钟)
    night_start1 = 21 * 60 + 0   # 21:00
    night_end1 = 23 * 60 + 59    # 23:59
    night_start2 = 0 * 60 + 0     # 00:00
    night_end2 = 2 * 60 + 30      # 02:30
    
    # 创建交易时间掩码
    mask = (
        ((time_total >= morning_start) & (time_total <= morning_end)) |      # 日盘上午
        ((time_total >= afternoon_start) & (time_total <= afternoon_end)) |  # 日盘下午
        ((time_total >= night_start1) & (time_total <= night_end1)) |        # 夜盘第一段
        ((time_total >= night_start2) & (time_total <= night_end2))          # 夜盘第二段
    )
    
    return df[mask]

# 绘图：K线+RSI+KDJ
def plot_charts_plotly(df: pd.DataFrame, symbol: str, data_type: str, rsi_period: int = 14):
    if df is None or df.empty:
        st.warning("数据为空，无法绘制图表")
        return

    df = df.copy()
    df['RSI'] = calculate_rsi(df['close'], period=rsi_period)
    df['K'], df['D'], df['J'] = calculate_kdj(df)

    # 3行1列子图（K线→RSI→KDJ）
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],  # 合理分配高度
        subplot_titles=(f'{symbol} {data_type}K线', f'RSI({rsi_period})', 'KDJ指标')
    )

    # 1. 主图：K线
    candlestick = go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='K线',
        increasing_line_color='red',    # 上涨为红色
        decreasing_line_color='green',   # 下跌为绿色
        increasing_fillcolor='red',
        decreasing_fillcolor='green',
        line=dict(width=1),
    )
    fig.add_trace(candlestick, row=1, col=1)

    # 2. 副图1：RSI、超买超卖线
    fig.add_trace(
        go.Scatter(x=df.index, y=df['RSI'], mode='lines', name=f'RSI({rsi_period})', line=dict(color='orange', width=1.5)),
        row=2, col=1
    )
    fig.add_hline(y=70, line_dash='dash', row=2, col=1, line_color='red', opacity=0.8, name='超买线')
    fig.add_hline(y=30, line_dash='dash', row=2, col=1, line_color='green', opacity=0.8, name='超卖线')
    fig.update_yaxes(range=[0, 100], row=2, col=1)  # RSI固定0-100范围

    # 3. 副图2：KDJ
    fig.add_trace(go.Scatter(x=df.index, y=df['K'], mode='lines', name='K', line=dict(color='blue', width=1.5)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['D'], mode='lines', name='D', line=dict(color='red', width=1.5)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['J'], mode='lines', name='J', line=dict(color='purple', width=1.5)), row=3, col=1)
    fig.add_hline(y=80, line_dash='dash', row=3, col=1, line_color='red', opacity=0.5)
    fig.add_hline(y=20, line_dash='dash', row=3, col=1, line_color='green', opacity=0.5)


    # 更新布局
    fig.update_layout(
        height=900,  # 图表总高度
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10)
        ),
        hovermode='x unified',  # 鼠标悬停时显示所有数据点
        template='plotly_white',  # 使用白色主题，更清晰
        margin=dict(l=60, r=50, t=80, b=60),
        font=dict(family="Arial, sans-serif", size=11),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    fig.update_xaxes(rangeslider_visible=False)  # 隐藏x轴缩放条（用鼠标滚轮缩放更灵活）
    
    # 更新x轴（所有子图共享x轴）
    if data_type == "日级":
        date_format = '%Y-%m-%d'
    else:
        date_format = '%m-%d %H:%M'
    
    fig.update_xaxes(
        title_text="时间",
        row=3, col=1,
        type='date',
        tickformat=date_format,
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray'
    )
    
    # 更新y轴标签和样式
    fig.update_yaxes(
        title_text="价格", 
        row=1, col=1,
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray'
    )
    fig.update_yaxes(
        title_text="RSI", 
        range=[0, 100], 
        row=2, col=1,
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray'
    )
    fig.update_yaxes(
        title_text="KDJ", 
        row=3, col=1,
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray'
    )
    
    # 显示图表
    st.plotly_chart(fig, use_container_width=True)
    

    

# 主程序（简化侧边栏）
def main():
    st.title("📈 期货技术指标分析系统")
    st.markdown("---")
    
    with st.sidebar:#上下文管理器，在这里相当于打开配置参数后保持主页面上下文
        st.header("⚙️ 配置参数")
        
        # 1. 标的选择
        symbol_option = st.selectbox(
            "选择标的",
            options=["IF", "IM", "IC", "IH"],
            index=0,
            help="IF: 沪深300, IM: 中证1000, IC: 中证500, IH: 上证50"
        )
        
        # 2. 数据类型选择
        data_type = st.radio(
            "数据类型",
            options=["日级", "分钟级"],
            index=0,
            help="选择数据的时间粒度"
        )
        
        # 3. 日期选择（日级+分钟级都定义start_date/end_date）
        if data_type == "日级":
            default_end = datetime.now().date()
            start_date = st.date_input(
                "开始日期",
                value=default_end - timedelta(days=365/2),
                max_value=default_end,
                min_value=default_end - timedelta(days=3650)
            )
            end_date = st.date_input(
                "结束日期",
                value=default_end,
                min_value=start_date,
                max_value=default_end
            )
        else:  # 分钟级：单独处理，确保start_date/end_date有值
            selected_date = st.date_input(
                "选择日期",
                value=datetime.now().date(),
                max_value=datetime.now().date()
            )
            start_date = selected_date
            end_date = selected_date
        
        #分钟级数据提示
        if data_type == "分钟级":
            st.info("⚠️ 分钟级数据量较大 建议选择单日数据")

        # 刷新按钮
        if st.button("🔄 刷新数据", type="primary"):
            st.rerun()
    
    # 数据获取与筛选
    symbol_code = SYMBOL_MAP[symbol_option]
    with st.spinner(f"正在获取 {symbol_option} {data_type}数据..."):
        df = get_daily_data(symbol_code) if data_type == "日级" else get_minute_data(symbol_code)
        if df is None or df.empty:
            st.error(f"❌ 无法获取 {symbol_option} 的{data_type}数据，请检查网络连接或稍后重试")
            return

        # 日期筛选
        try:
            start_datetime = pd.to_datetime(start_date)
            end_datetime = pd.to_datetime(end_date).replace(hour=23, minute=59, second=59)
            df_filtered = filter_data_by_date(df, start_datetime, end_datetime)
            
            # 如果是分钟级数据，过滤非交易时间
            if data_type == "分钟级" and not df_filtered.empty:
                original_count = len(df_filtered)
                df_filtered = filter_trading_hours(df_filtered)
                filtered_count = len(df_filtered)
                if original_count > filtered_count:
                    st.info(f"已过滤 {original_count - filtered_count} 条非交易时间数据")
            
            if df_filtered.empty:
                st.warning(f"⚠️ 周末没有数据或所选日期范围内无数据")
                return
        except Exception as e:
            st.error(f"日期筛选失败: {str(e)}")
            df_filtered = df

    # 指标计算
    df_filtered['RSI'] = calculate_rsi(df_filtered['close'], period=14)
    df_filtered['K'], df_filtered['D'], df_filtered['J'] = calculate_kdj(df_filtered)

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
    plot_charts_plotly(df_filtered, symbol_option, data_type, rsi_period=14)
    
    # 原始数据表格
    with st.expander("📊 查看原始数据"):
        st.dataframe(
            df_filtered[['open', 'high', 'low', 'close', 'RSI', 'K', 'D', 'J']].tail(200),
            use_container_width=True
        )

if __name__ == "__main__":
    main()