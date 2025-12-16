import akshare as ak
import mplfinance as mpf
import pandas as pd
from matplotlib import pyplot as plt

# 步骤3：获取K线数据
def get_stock_data():
    """
    从AKShare获取股票日K线数据
    """
    df = ak.futures_zh_daily_sina(symbol="IF2512")
    try:
        if df is None or df.empty:
            print("获取数据失败")
            exit(1)
        return df
    except Exception as e:
        print(f"获取数据时出错: {str(e)[:100]}")
        return None

df = get_stock_data()

print("原始数据列名:", df.columns.tolist())
print("数据前5行:")
print(df.head())

# 步骤4：数据清洗和格式化

print(type(df))

# 清洗数据
def clean_data(df):
    """
    清洗数据，转换为mplfinance需要的格式
    """
    # 不需要重命名，直接使用原始列名
    # 确保日期列为datetime类型并设为索引
    df['date'] = pd.to_datetime(df['date']) 
    df = df.set_index('date')  

print(type(df))