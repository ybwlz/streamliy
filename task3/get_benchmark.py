"""
获取基准数据并生成硬编码数组
使用整个期间的连续基准数据，每个交易日使用从第一个交易日到该交易日之间的累计基准收益率
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from data import get_benchmark_returns

print("="*80)
print("获取基准数据")
print("="*80)

# 1. 读取交易日期范围
trade_df = pd.read_csv('jiaoyi.csv', encoding='gbk')
trade_df['日期_parsed'] = pd.to_datetime(trade_df['日期'], format='%Y/%m/%d', errors='coerce')
start_date = trade_df['日期_parsed'].min()
end_date = trade_df['日期_parsed'].max()

print(f"\n交易日期范围: {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}")

# 2. 获取整个期间的连续基准数据
print("\n获取连续基准数据...")

benchmark_returns, benchmark_dates = get_benchmark_returns(
    start_date.strftime('%Y-%m-%d'),
    end_date.strftime('%Y-%m-%d')
)

if benchmark_returns is None or benchmark_dates is None:
    print("\n[错误] 无法获取基准数据")
    sys.exit(1)

print(f"\n[成功] 成功获取基准数据")
print(f"基准数据长度: {len(benchmark_returns)}")
print(f"基准日期范围: {benchmark_dates[0]} 到 {benchmark_dates[-1]}")

# 统计信息
total_return = np.prod(1 + benchmark_returns) - 1
print(f"\n连续基准收益率统计:")
print(f"  总收益率（复利）: {total_return*100:.4f}%")

# 3. 对齐到交易日期（使用累计基准收益率）
print("\n对齐到交易日期...")

trade_dates = trade_df['日期_parsed'].dropna().unique()
trade_dates = pd.Series(trade_dates).sort_values()

# 创建基准数据DataFrame（包含所有连续交易日）
benchmark_df = pd.DataFrame({
    '日期': pd.to_datetime(benchmark_dates),
    '收益率': benchmark_returns
})

# 计算累计基准收益率
benchmark_df['累计收益率'] = (1 + benchmark_df['收益率']).cumprod() - 1

# 对齐逻辑：
# 1. 第一个交易日：0（因为没有前一天价格）
# 2. 后续交易日：计算从第一个交易日到该交易日之间的累计基准收益率
#    然后转换为日收益率（相对于前一个交易日）

aligned_returns = []
prev_cumulative = 0.0  # 前一个交易日的累计基准收益率

for i, trade_date in enumerate(trade_dates):
    if i == 0:
        # 第一个交易日：0
        aligned_returns.append(0.0)
        prev_cumulative = 0.0
    else:
        # 找到该交易日对应的基准数据
        matching = benchmark_df[benchmark_df['日期'] == trade_date]
        if len(matching) > 0:
            # 找到匹配的基准数据
            current_cumulative = matching['累计收益率'].iloc[0]
            # 计算相对于前一个交易日的收益率
            if prev_cumulative != 0:
                daily_return = (1 + current_cumulative) / (1 + prev_cumulative) - 1
            else:
                # 如果前一个累计收益率为0，使用该交易日的基准收益率
                daily_return = matching['收益率'].iloc[0]
            aligned_returns.append(daily_return)
            prev_cumulative = current_cumulative
        else:
            # 如果没有匹配，找到该交易日之前最近的基准日期
            prev_benchmarks = benchmark_df[benchmark_df['日期'] < trade_date]
            if len(prev_benchmarks) > 0:
                # 使用最近的基准累计收益率
                nearest_cumulative = prev_benchmarks.iloc[-1]['累计收益率']
                nearest_return = prev_benchmarks.iloc[-1]['收益率']
                if prev_cumulative != 0:
                    daily_return = (1 + nearest_cumulative) / (1 + prev_cumulative) - 1
                else:
                    daily_return = nearest_return
                aligned_returns.append(daily_return)
                prev_cumulative = nearest_cumulative
            else:
                # 如果还是没有，使用0
                aligned_returns.append(0.0)

aligned_returns = np.array(aligned_returns)

print(f"[成功] 对齐成功，数据长度: {len(aligned_returns)}")

# 验证对齐后的数据
aligned_total_return = np.prod(1 + aligned_returns) - 1
print(f"对齐后基准数据总收益率: {aligned_total_return*100:.4f}%")

# 4. 生成硬编码数组
print("\n" + "="*80)
print("生成的基准数据数组（复制到risk.py中）")
print("="*80)

print("\nBENCHMARK_RETURNS_HARDCODED = np.array([")

# 每行10个值
for i in range(0, len(aligned_returns), 10):
    chunk = aligned_returns[i:i+10]
    values_str = ", ".join([f"{x:.8f}" for x in chunk])
    if i + 10 < len(aligned_returns):
        print(f"    {values_str},")
    else:
        print(f"    {values_str}")

print("])")
print("\n完成！")
