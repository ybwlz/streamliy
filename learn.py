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
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 导入 Plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    使用 Plotly 绘制交互式K线图和技术指标
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
    
    # 创建子图：3行1列，共享x轴
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],  # 主图占50%，两个副图各占25%
        subplot_titles=(
            f'{symbol} {data_type}K线图',
            'RSI指标',
            'KDJ指标'
        )
    )
    
    # ========== 主图：K线图（蜡烛图）==========
    # 准备自定义悬停文本
    hover_texts = []
    for idx, row in df.iterrows():
        change = row['close'] - row['open']
        change_pct = (change / row['open'] * 100) if row['open'] != 0 else 0
        hover_text = (
            f"<b>{idx.strftime('%Y-%m-%d %H:%M' if data_type == '分钟级' else '%Y-%m-%d')}</b><br>"
            f"开盘: {row['open']:.2f}<br>"
            f"最高: {row['high']:.2f}<br>"
            f"最低: {row['low']:.2f}<br>"
            f"收盘: {row['close']:.2f}<br>"
            f"涨跌: {change:+.2f} ({change_pct:+.2f}%)<br>"
            f"成交量: {row['volume']:.0f}"
        )
        hover_texts.append(hover_text)
    
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
        customdata=hover_texts,
        hovertemplate='%{customdata}<extra></extra>'
    )
    fig.add_trace(candlestick, row=1, col=1)
    
    # ========== 副图1：RSI指标 ==========
    # RSI曲线（带详细悬停信息）
    rsi_hover_texts = []
    for idx, row in df.iterrows():
        rsi_value = row['RSI']
        status = "超买" if rsi_value > 70 else ("超卖" if rsi_value < 30 else "正常")
        rsi_hover = (
            f"<b>RSI指标</b><br>"
            f"时间: {idx.strftime('%Y-%m-%d %H:%M' if data_type == '分钟级' else '%Y-%m-%d')}<br>"
            f"RSI值: {rsi_value:.2f}<br>"
            f"状态: {status}<br>"
            f"收盘价: {row['close']:.2f}"
        )
        rsi_hover_texts.append(rsi_hover)
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['RSI'],
            mode='lines',
            name='RSI(14)',
            line=dict(color='orange', width=1.5),
            customdata=rsi_hover_texts,
            hovertemplate='%{customdata}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # 超买线（70）
    fig.add_hline(
        y=70, 
        line_dash="dash", 
        line_color="red", 
        opacity=0.7,
        annotation_text="超买线(70)",
        annotation_position="right",
        row=2, col=1
    )
    
    # 超卖线（30）
    fig.add_hline(
        y=30, 
        line_dash="dash", 
        line_color="green", 
        opacity=0.7,
        annotation_text="超卖线(30)",
        annotation_position="right",
        row=2, col=1
    )
    
    # 填充30-70区域（使用fill='tonexty'）
    # 先添加上边界线（70）
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=[70] * len(df),
            mode='lines',
            name='正常区间',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip',
            fill=None
        ),
        row=2, col=1
    )
    # 再添加下边界线（30），并填充到上一条线
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=[30] * len(df),
            mode='lines',
            name='',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip',
            fill='tonexty',
            fillcolor='rgba(128,128,128,0.1)'
        ),
        row=2, col=1
    )
    
    # ========== 副图2：KDJ指标 ==========
    # 准备KDJ悬停信息
    kdj_hover_texts = []
    for idx, row in df.iterrows():
        k_value = row['K']
        d_value = row['D']
        j_value = row['J']
        kdj_hover = (
            f"<b>KDJ指标</b><br>"
            f"时间: {idx.strftime('%Y-%m-%d %H:%M' if data_type == '分钟级' else '%Y-%m-%d')}<br>"
            f"K值: {k_value:.2f}<br>"
            f"D值: {d_value:.2f}<br>"
            f"J值: {j_value:.2f}<br>"
            f"收盘价: {row['close']:.2f}"
        )
        kdj_hover_texts.append(kdj_hover)
    
    # K线
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['K'],
            mode='lines',
            name='K',
            line=dict(color='blue', width=1.5),
            customdata=kdj_hover_texts,
            hovertemplate='%{customdata}<extra></extra>'
        ),
        row=3, col=1
    )
    
    # D线
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['D'],
            mode='lines',
            name='D',
            line=dict(color='red', width=1.5),
            customdata=kdj_hover_texts,
            hovertemplate='%{customdata}<extra></extra>'
        ),
        row=3, col=1
    )
    
    # J线
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['J'],
            mode='lines',
            name='J',
            line=dict(color='purple', width=1.5),
            customdata=kdj_hover_texts,
            hovertemplate='%{customdata}<extra></extra>'
        ),
        row=3, col=1
    )
    
    # KDJ超买超卖参考线
    fig.add_hline(y=80, line_dash="dot", line_color="red", opacity=0.3, row=3, col=1)
    fig.add_hline(y=20, line_dash="dot", line_color="green", opacity=0.3, row=3, col=1)
    
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

