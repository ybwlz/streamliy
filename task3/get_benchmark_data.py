"""
脚本：获取基准数据并输出为可硬编码的格式

使用方法：
1. 运行此脚本：python get_benchmark_data.py
2. 复制输出的 BENCHMARK_RETURNS_HARDCODED 数组
3. 粘贴到 risk.py 文件中替换 BENCHMARK_RETURNS_HARDCODED = None
"""
import pandas as pd
import numpy as np

try:
    from data import get_benchmark_daily_returns_aligned
    
    # 读取交易日期
    trade_df = pd.read_csv('jiaoyi.csv', encoding='gbk')
    trade_df['日期_仅'] = pd.to_datetime(trade_df['日期'], format='%Y/%m/%d', errors='coerce')
    trade_dates = trade_df['日期_仅'].dropna().unique()
    
    # 获取基准数据
    print("正在获取基准数据...")
    benchmark = get_benchmark_daily_returns_aligned(pd.Series(trade_dates))
    
    if benchmark is not None:
        print(f"\n✅ 成功获取基准数据")
        print(f"基准数据长度: {len(benchmark)}")
        print(f"基准数据统计:")
        print(f"  均值: {np.mean(benchmark)*100:.6f}%")
        print(f"  标准差: {np.std(benchmark)*100:.6f}%")
        print(f"  最小值: {np.min(benchmark)*100:.6f}%")
        print(f"  最大值: {np.max(benchmark)*100:.6f}%")
        
        # 输出为Python数组格式
        print("\n" + "="*80)
        print("请将以下代码复制到 risk.py 中，替换 BENCHMARK_RETURNS_HARDCODED = None")
        print("="*80)
        print("\nBENCHMARK_RETURNS_HARDCODED = np.array([")
        # 每行显示10个值
        for i in range(0, len(benchmark), 10):
            chunk = benchmark[i:i+10]
            values_str = ", ".join([f"{x:.8f}" for x in chunk])
            if i + 10 < len(benchmark):
                print(f"    {values_str},")
            else:
                print(f"    {values_str}")
        print("])")
        print("\n" + "="*80)
        
        # 同时输出日期范围信息
        print(f"\n基准数据日期范围: {trade_dates.min()} 到 {trade_dates.max()}")
        print(f"基准数据交易日数: {len(benchmark)}")
    else:
        print("❌ 无法获取基准数据")
        print("请检查：")
        print("1. akshare是否已安装：pip install akshare")
        print("2. 网络连接是否正常")
        print("3. data.py文件是否存在")
        
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保已安装所需依赖：pip install pandas numpy akshare")
except Exception as e:
    print(f"❌ 获取数据时出错: {e}")
    import traceback
    traceback.print_exc()
