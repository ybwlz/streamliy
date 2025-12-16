# app.py：Flask接口服务端（核心业务逻辑）
from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
import requests

# ---------------------- 1. 初始化Flask应用 ----------------------
app = Flask(__name__)

# ---------------------- 2. RSI计算核心函数 ----------------------
def calculate_rsi(price, period=14):
    """计算RSI指标（默认周期14）"""
    delta = price.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)  # 避免除零
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ---------------------- 3. 核心接口（接收start_time/end_time参数） ----------------------
@app.route("/djkline", methods=['GET'])
def futures_k_line():
    try:
        # 1. 接收并校验用户传入的查询参数
        start_time = request.args.get('start_time')  # 示例：2025-01-01 00:00:00
        end_time = request.args.get('end_time')      # 示例：2025-01-02 23:59:59
        if not start_time or not end_time:
            return jsonify({
                "code": 400,
                "msg": "缺少必填参数：start_time 或 end_time",
                "data": []
            }), 400

        # 2. 调用第三方K线数据接口（固定code=IM）
        third_party_url = "http://dz.szdjct.com:5208/new_project/k_line_data"
        request_params = {
            "code": "IM",
            "start_time": start_time,
            "end_time": end_time
        }
        # 发送GET请求到第三方接口（设置超时10秒）
        response = requests.get(third_party_url, params=request_params, timeout=10)
        response.raise_for_status()  # 状态码非200直接抛异常
        third_data = response.json()

        # 3. 校验第三方接口返回数据
        if third_data.get("code") != 200 or not isinstance(third_data.get("result"), list):
            return jsonify({
                "code": 400,
                "msg": f"第三方接口返回异常：{third_data.get('msg', '无详细信息')}",
                "data": []
            }), 400

        # 4. 数据处理（DataFrame格式化、RSI计算、买卖点标记）
        df = pd.DataFrame(third_data["result"])
        if "close" not in df.columns:
            return jsonify({
                "code": 400,
                "msg": "第三方数据缺少核心字段：close（收盘价）",
                "data": []
            }), 400

        # 计算RSI（周期默认14）
        df['rsi'] = calculate_rsi(df['close'], period=14)
        # 标记买卖点：RSI<30→买入(B=True)，RSI>70→卖出(S=True)
        df['B'] = df['rsi'] < 30
        df['S'] = df['rsi'] > 70
        # 时间戳转换为字符串格式（示例：2025-07-07 09:30:00）
        df['eob'] = pd.to_datetime(df['eob'], unit='s').dt.strftime("%Y-%m-%d %H:%M:%S")

        # 5. 构造最终返回数据（只保留需要的字段）
        result_list = df[['eob', 'B', 'S']].to_dict("records")

        # 6. 按要求格式返回响应
        return jsonify({
            "code": 200,
            "msg": "请求成功",
            "data": result_list
        }), 200

    # 捕获接口请求异常（超时、网络错误等）
    except requests.exceptions.RequestException as e:
        return jsonify({
            "code": 500,
            "msg": f"调用K线接口失败：{str(e)}",
            "data": []
        }), 500
    # 捕获其他未知异常
    except Exception as e:
        return jsonify({
            "code": 500,
            "msg": f"服务器内部错误：{str(e)}",
            "data": []
        }), 500

# ---------------------- 4. 启动服务 ----------------------
if __name__ == '__main__':
    # debug=True：开发模式（自动重载、显示错误详情），生产环境需关闭
    app.run(debug=True, port=5000)  # 服务运行在 http://127.0.0.1:5000