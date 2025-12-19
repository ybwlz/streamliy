"""
沪深300指数基准收益和波动率计算
时间范围：2025-01-01 到 2025-12-17
初始资金：100万
"""

import akshare as ak
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 设置参数
INITIAL_CAPITAL = 1_000_000  # 初始资金100万
START_DATE = "2025-01-01"
END_DATE = "2025-12-17"
TRADING_DAYS_PER_YEAR = 252  # 年交易日数
RF_RATE = 0.04  # 无风险利率4%


def get_csi300_data():
    """获取沪深300指数日线数据"""
    print("📡 获取沪深300指数数据...")
    
    try:
        # 方法1：使用akshare的CSIndex接口（更准确）
        df = ak.stock_zh_index_daily_csindex("000300")
        print("✅ 使用 CSIndex 接口获取数据")
    except:
        # 方法2：备用接口
        df = ak.stock_zh_index_daily("sh000300")
        print("⚠️ 使用备用接口获取数据")
    
    # 确保索引是datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # 筛选日期范围
    df = df[(df.index >= START_DATE) & (df.index <= END_DATE)].copy()
    df = df.sort_index()
    
    print(f"✅ 获取到 {len(df)} 个交易日数据")
    print(f"日期范围: {df.index.min().strftime('%Y-%m-%d')} 到 {df.index.max().strftime('%Y-%m-%d')}")
    
    return df


def calculate_daily_metrics(df):
    """计算日收益率和净值"""
    df = df.copy()
    
    # 计算日收益率
    df['日收益率'] = df['close'].pct_change()
    
    # 移除第一个NaN
    df_clean = df.dropna(subset=['日收益率']).copy()
    
    if len(df_clean) == 0:
        print("❌ 没有有效数据")
        return None
    
    # 计算累计收益率
    df_clean['累计收益率'] = (1 + df_clean['日收益率']).cumprod() - 1
    
    # 计算净值（基于100万初始资金）
    df_clean['净值'] = INITIAL_CAPITAL * (1 + df_clean['累计收益率'])
    
    return df_clean


def calculate_total_return(df_clean):
    """计算总收益"""
    if len(df_clean) == 0:
        return 0.0
    
    end_value = df_clean['净值'].iloc[-1]
    total_return = (end_value - INITIAL_CAPITAL) / INITIAL_CAPITAL
    
    return total_return


def calculate_annualized_return(total_return, trading_days):
    """计算年化收益"""
    if trading_days == 0:
        return 0.0
    
    years = trading_days / TRADING_DAYS_PER_YEAR
    annualized_return = (1 + total_return) ** (1 / years) - 1
    
    return annualized_return


def calculate_volatility(daily_returns):
    """计算年化波动率"""
    if len(daily_returns) < 2:
        return 0.0
    
    daily_vol = np.std(daily_returns, ddof=1)
    annual_vol = daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR)
    
    return annual_vol


def calculate_max_drawdown(net_values):
    """计算最大回撤"""
    if len(net_values) == 0:
        return 0.0
    
    peak = net_values.expanding(min_periods=1).max()
    drawdown = (net_values - peak) / peak
    max_drawdown = drawdown.min()
    
    return max_drawdown


def calculate_sharpe(annualized_return, annual_vol):
    """计算夏普比率"""
    if annual_vol == 0:
        return 0.0
    
    sharpe = (annualized_return - RF_RATE) / annual_vol
    return sharpe


def calculate_monthly_table(df_clean):
    """生成月度收益和波动率表（类似聚宽格式）"""
    if len(df_clean) == 0:
        return pd.DataFrame()
    
    # 创建月度索引
    monthly_ends = []
    current_year = df_clean.index.min().year
    current_month = df_clean.index.min().month
    
    while True:
        month_end = pd.Timestamp(year=current_year, month=current_month, day=1) + pd.offsets.MonthEnd(1)
        if month_end > df_clean.index.max():
            break
        
        monthly_ends.append(month_end)
        
        # 下个月
        current_month += 1
        if current_month > 12:
            current_month = 1
            current_year += 1
    
    results = []
    
    for i, month_end in enumerate(monthly_ends):
        row = {'日期': month_end.strftime('%Y-%m')}
        
        # 1个月窗口
        month_start = month_end - pd.offsets.MonthBegin(1)
        month_data = df_clean[(df_clean.index >= month_start) & (df_clean.index <= month_end)]
        
        if len(month_data) > 0:
            # 月度收益
            month_return = (month_data['净值'].iloc[-1] / month_data['净值'].iloc[0]) - 1
            row['1个月_收益'] = month_return
            
            # 月度波动率
            if len(month_data) >= 2:
                month_vol = np.std(month_data['日收益率'], ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
                row['1个月_波动'] = month_vol
            else:
                row['1个月_波动'] = np.nan
        else:
            row['1个月_收益'] = np.nan
            row['1个月_波动'] = np.nan
        
        # 3个月窗口（需要至少3个月数据）
        if i >= 2:
            three_month_start = monthly_ends[i-2] - pd.offsets.MonthBegin(1)
            three_month_data = df_clean[(df_clean.index >= three_month_start) & (df_clean.index <= month_end)]
            
            if len(three_month_data) > 0:
                three_month_return = (three_month_data['净值'].iloc[-1] / three_month_data['净值'].iloc[0]) - 1
                row['3个月_收益'] = three_month_return
                
                if len(three_month_data) >= 2:
                    three_month_vol = np.std(three_month_data['日收益率'], ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
                    row['3个月_波动'] = three_month_vol
                else:
                    row['3个月_波动'] = np.nan
            else:
                row['3个月_收益'] = np.nan
                row['3个月_波动'] = np.nan
        else:
            row['3个月_收益'] = np.nan
            row['3个月_波动'] = np.nan
        
        # 6个月窗口（需要至少6个月数据）
        if i >= 5:
            six_month_start = monthly_ends[i-5] - pd.offsets.MonthBegin(1)
            six_month_data = df_clean[(df_clean.index >= six_month_start) & (df_clean.index <= month_end)]
            
            if len(six_month_data) > 0:
                six_month_return = (six_month_data['净值'].iloc[-1] / six_month_data['净值'].iloc[0]) - 1
                row['6个月_收益'] = six_month_return
                
                if len(six_month_data) >= 2:
                    six_month_vol = np.std(six_month_data['日收益率'], ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
                    row['6个月_波动'] = six_month_vol
                else:
                    row['6个月_波动'] = np.nan
            else:
                row['6个月_收益'] = np.nan
                row['6个月_波动'] = np.nan
        else:
            row['6个月_收益'] = np.nan
            row['6个月_波动'] = np.nan
        
        # 12个月窗口（需要至少12个月数据）
        if i >= 11:
            twelve_month_start = monthly_ends[i-11] - pd.offsets.MonthBegin(1)
            twelve_month_data = df_clean[(df_clean.index >= twelve_month_start) & (df_clean.index <= month_end)]
            
            if len(twelve_month_data) > 0:
                twelve_month_return = (twelve_month_data['净值'].iloc[-1] / twelve_month_data['净值'].iloc[0]) - 1
                row['12个月_收益'] = twelve_month_return
                
                if len(twelve_month_data) >= 2:
                    twelve_month_vol = np.std(twelve_month_data['日收益率'], ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
                    row['12个月_波动'] = twelve_month_vol
                else:
                    row['12个月_波动'] = np.nan
            else:
                row['12个月_收益'] = np.nan
                row['12个月_波动'] = np.nan
        else:
            row['12个月_收益'] = np.nan
            row['12个月_波动'] = np.nan
        
        results.append(row)
    
    return pd.DataFrame(results)


def calculate_rolling_metrics(df_clean):
    """计算滚动窗口指标"""
    daily_returns = df_clean['日收益率'].values
    
    # 21个交易日 ≈ 1个月
    window_sizes = {
        '1个月': 21,
        '3个月': 63,
        '6个月': 126,
        '12个月': 252
    }
    
    results = {}
    
    for name, window_days in window_sizes.items():
        if len(daily_returns) < window_days:
            print(f"⚠️ 数据不足，无法计算{name}滚动指标")
            results[name] = {'收益': [], '波动率': []}
            continue
        
        rolling_returns = []
        rolling_vols = []
        
        for i in range(window_days, len(daily_returns)):
            window_data = daily_returns[i-window_days:i]
            
            # 窗口期总收益
            window_return = np.prod(1 + window_data) - 1
            
            # 年化收益
            years = window_days / TRADING_DAYS_PER_YEAR
            annualized_return = (1 + window_return) ** (1/years) - 1
            
            # 年化波动率
            daily_vol = np.std(window_data, ddof=1)
            annual_vol = daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR)
            
            rolling_returns.append(annualized_return)
            rolling_vols.append(annual_vol)
        
        results[name] = {
            '收益': rolling_returns,
            '波动率': rolling_vols
        }
    
    return results


def main():
    """主函数"""
    print("="*60)
    print("📈 沪深300指数基准分析（2025-01-01 到 2025-12-17）")
    print("="*60)
    
    # 1. 获取数据
    df = get_csi300_data()
    if len(df) == 0:
        print("❌ 没有获取到数据")
        return
    
    # 2. 计算日收益率和净值
    df_clean = calculate_daily_metrics(df)
    if df_clean is None:
        return
    
    trading_days = len(df_clean)
    
    # 3. 计算总收益
    total_return = calculate_total_return(df_clean)
    
    # 4. 计算年化收益
    annualized_return = calculate_annualized_return(total_return, trading_days)
    
    # 5. 计算波动率
    annual_vol = calculate_volatility(df_clean['日收益率'].values)
    
    # 6. 计算最大回撤
    max_dd = calculate_max_drawdown(df_clean['净值'])
    
    # 7. 计算夏普比率
    sharpe = calculate_sharpe(annualized_return, annual_vol)
    
    # 8. 生成月度表格
    monthly_table = calculate_monthly_table(df_clean)
    
    # 9. 计算滚动指标
    rolling_metrics = calculate_rolling_metrics(df_clean)
    
    # 打印结果
    print("\n" + "="*60)
    print("📊 基准表现汇总")
    print("="*60)
    print(f"初始资金: {INITIAL_CAPITAL:,.0f} 元")
    print(f"最终净值: {df_clean['净值'].iloc[-1]:,.2f} 元")
    print(f"交易日数: {trading_days} 天")
    print(f"总收益率: {total_return*100:.4f}%")
    print(f"年化收益率: {annualized_return*100:.4f}%")
    print(f"年化波动率: {annual_vol*100:.4f}%")
    print(f"最大回撤: {max_dd*100:.4f}%")
    print(f"夏普比率: {sharpe:.4f}")
    
    print("\n" + "="*60)
    print("📅 月度收益表（年化）")
    print("="*60)
    if not monthly_table.empty:
        # 格式化显示
        display_df = monthly_table.copy()
        
        # 收益列转为百分比
        for col in display_df.columns:
            if '收益' in col:
                display_df[col] = display_df[col].apply(lambda x: f"{x*100:.4f}%" if pd.notnull(x) else "NaN")
            elif '波动' in col:
                display_df[col] = display_df[col].apply(lambda x: f"{x*100:.4f}%" if pd.notnull(x) else "NaN")
        
        print(display_df.to_string(index=False))
    else:
        print("无月度数据")
    
    print("\n" + "="*60)
    print("🔄 滚动窗口统计")
    print("="*60)
    for window_name, metrics in rolling_metrics.items():
        if len(metrics['收益']) > 0:
            avg_return = np.mean(metrics['收益']) * 100
            avg_vol = np.mean(metrics['波动率']) * 100
            print(f"{window_name:8s} 平均年化收益: {avg_return:.4f}%, 平均波动率: {avg_vol:.4f}%")
    
    print("\n" + "="*60)
    print("📁 数据保存")
    print("="*60)
    
    # 保存净值曲线
    df_clean[['净值', '日收益率']].to_csv('csi300_net_value.csv')
    print("✅ 净值曲线已保存到: csi300_net_value.csv")
    
    # 保存月度表格
    monthly_table.to_csv('csi300_monthly_table.csv', index=False)
    print("✅ 月度表格已保存到: csi300_monthly_table.csv")
    
    # 保存日收益率序列
    df_clean[['日收益率']].to_csv('csi300_daily_returns.csv')
    print("✅ 日收益率序列已保存到: csi300_daily_returns.csv")
    
    print("\n✅ 计算完成！")


if __name__ == "__main__":
    main()