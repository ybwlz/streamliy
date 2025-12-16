from flask import Flask, request, jsonify
import akshare as ak
import pandas as pd
import numpy as np
import json

app = Flask(__name__)

def get_im_minute_rsi(code: str, start_time: str, end_time: str) -> dict:
    """
    获取IM期货分钟级RSI信号
    :param code: 合约代码，如"IM0"
    :param start_time: 开始时间，如"2025-12-11 09:30:00"
    :param end_time: 结束时间，如"2025-12-11 10:00:00"
    """
    try:
        # 1. 使用分时行情数据接口
        # 获取1分钟K线数据
        df = ak.futures_zh_minute_sina(symbol=code, period="1")
        
        if df.empty:
            return {"code": 400, "msg": "获取数据失败", "data": []}
        
        # 2. 计算RSI
        close_prices = df['close']
        delta = close_prices.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 3. 时间筛选
        df['datetime'] = pd.to_datetime(df['datetime'])
        start_dt = pd.to_datetime(start_time)
        end_dt = pd.to_datetime(end_time)
        
        mask = (df['datetime'] >= start_dt) & (df['datetime'] <= end_dt)
        df_filtered = df[mask]
        
        if df_filtered.empty:
            return {"code": 400, "msg": "无数据", "data": []}
        
        # 4. 生成信号
        data = []
        for _, row in df_filtered.iterrows():
            rsi = row.get('rsi', np.nan)
            data.append({
                "eob": row['datetime'].strftime("%Y-%m-%d %H:%M:%S"),
                "B": False if pd.isna(rsi) else rsi < 30,
                "S": False if pd.isna(rsi) else rsi > 70
            })
        
        return {"code": 200, "msg": "请求成功", "data": data}
    
    except Exception as e:
        return {"code": 500, "msg": f"错误: {str(e)}", "data": []}

@app.route('/get_rsi', methods=['GET'])
def get_rsi():
    """
    GET接口：获取RSI信号
    参数：
    - code: 合约代码（如IM0）
    - start_time: 开始时间
    - end_time: 结束时间
    """
    # 从GET参数中获取数据
    code = request.args.get('code', 'IM0')
    start_time = request.args.get('start_time')
    end_time = request.args.get('end_time')
    
    # 参数校验
    if not start_time or not end_time:
        return jsonify({
            "code": 400,
            "msg": "参数错误：start_time和end_time为必填参数",
            "data": []
        })
    
    # 调用函数获取RSI数据
    result = get_im_minute_rsi(code, start_time, end_time)
    return jsonify(result)

# 测试用例
if __name__ == '__main__':
    app.run(debug=True, port=5000)

#示例URL: http://127.0.0.1:5000/get_rsi?code=IM0&start_time=2025-12-11 13:30:00&end_time=2025-12-11 14:00:00