import akshare as ak
import pandas as pd

def get_stock_data():
    """
    从AKShare获取股票日K线数据
    """
    df = ak.futures_zh_daily_sina(symbol="IF2511")
    try:
        if df is None or df.empty:
            print("获取数据失败")
            exit(1)
        return df
    except Exception as e:
        print(f"获取数据时出错: {str(e)[:100]}")
        return None

df = get_stock_data()

df["date"] = pd.to_datetime(df["date"])

if df is None:  # 添加检查
    print("数据获取失败，程序退出")
    exit(1)  # 退出程序
print("原始数据列名:", df.columns.tolist())
print("数据前5行:")
print(df.head())
print(df)