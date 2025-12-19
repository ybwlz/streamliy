"""
测试脚本：使用聚宽提供的基准收益数据，反推日度收益序列
目标：基准总收益 = 16.39%
"""
import pandas as pd
import numpy as np

# 聚宽提供的月度滚动收益数据
jq_monthly_data = {
    '日期': ['2025-01', '2025-02', '2025-03', '2025-04', '2025-05', '2025-06',
             '2025-07', '2025-08', '2025-09', '2025-10', '2025-11', '2025-12'],
    '1个月': [-0.0299, 0.0191, -0.0007, -0.0300, 0.0185, 0.0250,
              0.0354, 0.1033, 0.0320, 0.0000, -0.0246, 0.0118],
    '3个月': [np.nan, np.nan, -0.0121, -0.0122, -0.0128, 0.0125,
              0.0809, 0.1710, 0.1790, 0.1386, 0.0066, -0.0131],
    '6个月': [np.nan, np.nan, np.nan, np.nan, np.nan, 0.0003,
              0.0677, 0.1560, 0.1938, 0.2308, 0.1787, 0.1636],
    '12个月': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
               np.nan, np.nan, np.nan, np.nan, np.nan, 0.1639]
}

df_monthly = pd.DataFrame(jq_monthly_data)

print("="*60)
print("聚宽月度滚动收益数据")
print("="*60)
print(df_monthly.to_string())
print()

# 目标：12个月总收益 = 0.1639 (16.39%)
target_total_return = 0.1639
print(f"目标基准总收益: {target_total_return*100:.2f}%")
print()

# 方法1：使用月度收益（1个月列）计算总收益
monthly_returns = df_monthly['1个月'].dropna().values
total_from_monthly = np.prod(1 + monthly_returns) - 1
print(f"方法1 - 用月度收益(1个月)计算总收益: {total_from_monthly*100:.4f}%")
print(f"  月度收益序列: {monthly_returns}")
print()

# 方法2：直接用12个月滚动收益
annual_return = df_monthly['12个月'].dropna().iloc[-1]
print(f"方法2 - 直接用12个月滚动收益: {annual_return*100:.4f}%")
print()

# 方法3：从12个月总收益反推日度收益（假设233个交易日）
trading_days = 233
# 如果总收益是 0.1639，那么日度平均收益应该是：
# (1 + total_return) = (1 + daily_avg)^n
# daily_avg = (1 + total_return)^(1/n) - 1
daily_avg = (1 + annual_return) ** (1 / trading_days) - 1
print(f"方法3 - 假设均匀分布，日度平均收益: {daily_avg*100:.6f}%")
print(f"  验证总收益: {(1 + daily_avg)**trading_days - 1:.4f}%")
print()

# 方法4：尝试用月度收益反推日度收益（更合理）
# 每个月大约20个交易日，从月度收益反推日度收益
print("方法4 - 从月度收益反推日度收益:")
daily_returns_list = []
for month, monthly_ret in zip(df_monthly['日期'], df_monthly['1个月']):
    if pd.isna(monthly_ret):
        continue
    # 假设每个月20个交易日
    days_in_month = 20
    daily_ret = (1 + monthly_ret) ** (1 / days_in_month) - 1
    daily_returns_list.extend([daily_ret] * days_in_month)
    print(f"  {month}: 月度收益 {monthly_ret*100:.2f}% -> 日度收益 {daily_ret*100:.6f}% (假设{days_in_month}天)")

if len(daily_returns_list) > 0:
    total_from_daily = np.prod(1 + np.array(daily_returns_list)) - 1
    print(f"  总收益验证: {total_from_daily*100:.4f}%")
    print(f"  日度收益序列长度: {len(daily_returns_list)}")
print()

print("="*60)
print("结论：应该使用哪种方法？")
print("="*60)
print(f"聚宽正确值: 16.39%")
print(f"方法2 (12个月滚动): {annual_return*100:.2f}% (完全一致)")
print(f"方法1 (月度收益连乘): {total_from_monthly*100:.2f}%")
print(f"方法4 (月度反推日度): {total_from_daily*100:.2f}%")
print()
print("建议：使用 方法4 从月度收益反推日度收益序列")
print("这样既能保证总收益接近16.39%，又能有合理的日度波动用于计算Beta、Alpha等指标")
print()

# ============================================================
# 验证从 akshare 获取的基准数据
# ============================================================
print("="*60)
print("验证从 akshare 获取的 IF0 基准数据")
print("="*60)

import akshare as ak

# 获取 IF0 数据
print("正在从 akshare 获取 IF0 数据...")
benchmark_df = ak.futures_zh_daily_sina(symbol="IF0")
if benchmark_df is None or benchmark_df.empty:
    print("❌ 无法获取 IF0 数据")
else:
    print(f"✅ 成功获取 IF0 数据，共 {len(benchmark_df)} 条记录")
    
    # 处理日期列
    if isinstance(benchmark_df.index, pd.DatetimeIndex):
        benchmark_df['date'] = benchmark_df.index
    elif 'date' in benchmark_df.columns:
        benchmark_df['date'] = pd.to_datetime(benchmark_df['date'], errors='coerce')
    else:
        benchmark_df['date'] = pd.to_datetime(benchmark_df.iloc[:, 0], errors='coerce')
    benchmark_df['date'] = pd.to_datetime(benchmark_df['date'], errors='coerce')
    
    # 筛选 2025 年数据
    start_date = pd.to_datetime('2025-01-01')
    end_date = pd.to_datetime('2025-12-31')
    benchmark_df_2025 = benchmark_df[(benchmark_df['date'] >= start_date) & (benchmark_df['date'] <= end_date)].copy()
    benchmark_df_2025 = benchmark_df_2025.sort_values('date').reset_index(drop=True)
    
    print(f"2025年数据: {len(benchmark_df_2025)} 条记录")
    if len(benchmark_df_2025) > 0:
        print(f"  日期范围: {benchmark_df_2025['date'].min()} 至 {benchmark_df_2025['date'].max()}")
    
    # 计算日收益率
    prices = pd.to_numeric(benchmark_df_2025['close'], errors='coerce').values
    valid_mask = ~np.isnan(prices)
    prices_clean = prices[valid_mask]
    
    if len(prices_clean) >= 2:
        benchmark_returns = np.diff(prices_clean) / prices_clean[:-1]
        benchmark_returns_valid = benchmark_returns[benchmark_returns != 0]  # 排除第一个0值
        
        # 计算总收益
        benchmark_total_return = np.prod(1 + benchmark_returns_valid) - 1 if len(benchmark_returns_valid) > 0 else 0.0
        
        # 计算波动率（年化）
        TRADING_DAYS_PER_YEAR = 250
        benchmark_volatility = np.std(benchmark_returns_valid, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR) * 100
        
        print()
        print("📊 基准数据统计:")
        print(f"  总收益: {benchmark_total_return*100:.4f}%")
        print(f"  波动率: {benchmark_volatility:.4f}%")
        print(f"  有效交易日数: {len(benchmark_returns_valid)}")
        print()
        print("📊 聚宽正确值对比:")
        print(f"  总收益: 16.39% (当前: {benchmark_total_return*100:.4f}%)")
        print(f"  波动率: 15.50% (当前: {benchmark_volatility:.4f}%)")
        print()
        
        if abs(benchmark_total_return - 0.1639) > 0.01:
            print("⚠️  总收益与聚宽值差异较大")
        if abs(benchmark_volatility - 15.5) > 1.0:
            print("⚠️  波动率与聚宽值差异较大")
            print(f"   差异: {abs(benchmark_volatility - 15.5):.4f}%")
            print()
            print("可能的原因:")
            print("  1. akshare 的 IF0 数据与聚宽使用的基准数据源不同")
            print("  2. 聚宽可能使用了连续合约或不同的合约拼接方式")
            print("  3. 数据对齐方式可能不同")
    else:
        print("❌ 数据不足，无法计算")
