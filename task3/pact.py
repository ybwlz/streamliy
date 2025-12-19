import akshare as ak

df = ak.futures_zh_daily_sina(symbol="AU0")
print(df.tail(10))
