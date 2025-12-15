# test.py：接口测试端（模拟GET请求）
import requests

def test_djkline_api():
    """测试/djkline接口，传入时间范围参数"""
    # 1. 接口地址（Flask服务启动后的地址）
    api_url = "http://127.0.0.1:5000/djkline"
    
    # 2. 测试参数（需替换为实际要查询的时间范围，格式：YYYY-MM-DD HH:MM:SS）
    test_params = {
        "start_time": "2024-07-07 00:00:00",
        "end_time": "2025-07-08 23:59:59"
    }
    
    try:
        # 3. 发送GET请求
        response = requests.get(api_url, params=test_params, timeout=15)
        # 4. 打印响应结果（格式化输出，方便查看）
        print("="*50)
        print(f"接口请求地址：{response.url}")
        print(f"响应状态码：{response.status_code}")
        print(f"响应数据：\n{response.json()}")
        print("="*50)

    except Exception as e:
        print(f"测试失败：{str(e)}")
        print("提示：请先启动 app.py 中的Flask服务，再运行此测试脚本！")

if __name__ == '__main__':
    # 执行测试
    test_djkline_api()