"""
Streamlit 期货数据分析应用
- 仅保留：K线 + RSI + KDJ
- 支持10年数据查询
- 纯Plotly交互式图表
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
@st.cache_data(ttl=60 * 30)
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
@st.cache_data(ttl=60 * 5)
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

# 简化绘图函数：仅K线+RSI+KDJ（3行子图）
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
        vertical_spacing=0.03,
        row_heights=[0.5, 0.25, 0.25],  # 合理分配高度
        subplot_titles=(f'{symbol} {data_type}K线', f'RSI({rsi_period})', 'KDJ指标')
    )

    # 1. 主图：K线
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'], high=df['high'], low=df['low'], close=df['close'],
            name='K线',
            # 正确的涨跌颜色设置：字典格式
            increasing=dict(fillcolor='red', line=dict(color='red')),  # 上涨（红）
            decreasing=dict(fillcolor='green', line=dict(color='green'))  # 下跌（绿）
        ),
        row=1, col=1
    )

    # 2. 副图1：RSI（含超买超卖线）
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

    # 图表样式优化
    fig.update_layout(
        height=800,
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        hovermode='x unified'  # 悬停时显示同一x轴所有数据
    )
    fig.update_xaxes(rangeslider_visible=False)  # 隐藏x轴缩放条（用鼠标滚轮缩放更灵活）

    st.plotly_chart(fig, use_container_width=True)

# 主程序（简化侧边栏）
def main():
    st.title("📈 期货技术指标分析系统")
    st.markdown("---")
    
    with st.sidebar:
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
        
        # # 4. RSI周期设置
        # st.markdown("---")
        # st.subheader("指标配置")
        # rsi_period = st.number_input("RSI 周期", min_value=5, max_value=50, value=14, step=1)
        
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
            if df_filtered.empty:
                st.warning(f"⚠️ 周末没有数据或所选日期范围内无数据")
                return
        except Exception as e:
            st.error(f"日期筛选失败: {str(e)}")
            df_filtered = df

    # 指标计算
    df_filtered['RSI'] = calculate_rsi(df_filtered['close'], period=14)
    df_filtered['K'], df_filtered['D'], df_filtered['J'] = calculate_kdj(df_filtered)

    # 数据概览
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("数据点数", len(df_filtered))
    with col2:
        st.metric("最新价格", f"{df_filtered['close'].iloc[-1]:.2f}")
    with col3:
        st.metric("最新 RSI", f"{calculate_rsi(df_filtered['close'], 14).iloc[-1]:.2f}")
    k, d, _ = calculate_kdj(df_filtered)
    with col4:
        st.metric("最新 K/D", f"{k.iloc[-1]:.2f}/{d.iloc[-1]:.2f}")
    
    st.markdown("---")
    
    # 绘制简化图表
    plot_charts_plotly(df_filtered, symbol_option, data_type, rsi_period=14)
    
    # 原始数据表格（仅保留核心列）
    with st.expander("📊 查看原始数据"):
        st.dataframe(
            df_filtered[['open', 'high', 'low', 'close', 'RSI', 'K', 'D', 'J']].tail(200),
            use_container_width=True
        )

if __name__ == "__main__":
    main()