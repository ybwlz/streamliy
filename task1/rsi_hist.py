import akshare as ak
import pandas as pd
import numpy as np
import json

def calculate_rsi(prices, period=14):
    """计算RSI指标"""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

code = "IM0"
def get_im_rsi_signals(start_date: str, end_date: str,code:str) -> dict:
    """
    获取IM期货RSI买卖信号
    :param start_date: 开始日期，格式"2024-01-01"
    :param end_date: 结束日期，格式"2024-12-10"
    """
    
    try:
        # 1. 获取IM期货日频数据 - 使用IM0（中证1000期货连续合约）
        df = ak.futures_zh_daily_sina(symbol=code)
        
        if df.empty:
            return {"code": 400, "msg": "获取数据失败", "data": []}
        
        # 2. 计算RSI
        df['rsi'] = calculate_rsi(df['close'], 14)
        
        # 3. 筛选时间范围
        df['date'] = pd.to_datetime(df['date'])
        mask = (df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))
        df = df[mask]
        
        if df.empty:
            return {"code": 400, "msg": f"在{start_date}到{end_date}范围内无数据", "data": []}
        
        # 4. 生成信号
        data = []
        for _, row in df.iterrows():
            rsi = row.get('rsi', np.nan)
            data.append({
                "eob": row['date'].strftime("%Y-%m-%d 15:00:00"),  # 补全时间
                "B": False if pd.isna(rsi) else rsi < 30,
                "S": False if pd.isna(rsi) else rsi > 70
            })
        
        return {"code": 200, "msg": "请求成功", "data": data}
    
    except Exception as e:
        return {"code": 500, "msg": f"错误: {str(e)}", "data": []}

# 测试
if __name__ == "__main__":
    result = get_im_rsi_signals("2024-01-01", "2024-12-10",code)
    print(json.dumps(result, indent=2, ensure_ascii=False))