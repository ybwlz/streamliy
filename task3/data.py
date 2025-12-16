import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime

def get_futures_data(symbol="AU2404"):
    """
    从AKShare获取期货数据
    
    参数:
    symbol: 期货合约代码，如 "AU2404" 表示黄金2404合约
    """
    try:
        df = ak.futures_zh_daily_sina(symbol=symbol)
        if df is None or df.empty:
            print(f"获取 {symbol} 数据失败")
            return None
        return df
    except Exception as e:
        print(f"获取数据时出错: {str(e)[:100]}")
        return None

def get_benchmark_returns(start_date=None, end_date=None):
    """
    获取基准收益率序列（用于计算Alpha、Beta等指标）
    
    参数:
    start_date: 开始日期，格式 'YYYY-MM-DD' 或 None（自动从jiaoyi.csv获取）
    end_date: 结束日期，格式 'YYYY-MM-DD' 或 None（自动从jiaoyi.csv获取）
    
    返回:
    benchmark_returns: 基准日收益率序列（小数形式，如0.01表示1%）
    benchmark_dates: 对应的日期序列
    """
    # 如果未指定日期，从jiaoyi.csv读取交易日期范围
    if start_date is None or end_date is None:
        try:
            trade_df = pd.read_csv('jiaoyi.csv', encoding='gbk')
            trade_df['日期_parsed'] = pd.to_datetime(trade_df['日期'], format='%Y/%m/%d', errors='coerce')
            start_date = trade_df['日期_parsed'].min().strftime('%Y-%m-%d')
            end_date = trade_df['日期_parsed'].max().strftime('%Y-%m-%d')
        except Exception as e:
            print(f"读取jiaoyi.csv失败: {e}")
            return None, None
    
    # 获取黄金期货主力连续合约数据
    # 由于交易涉及多个合约（AU2404, AU2406, AU2408等），使用主力连续合约更合适
    try:
        # 尝试多个可能的合约代码
        symbols_to_try = [
            "AU9999",  # 黄金主力连续合约（根据akshare文档）
            "AU8888",  # 黄金指数合约
            "AU0",     # 可能的代码
            "AU2404",  # 第一个交易合约
        ]
        
        df = None
        successful_symbol = None
        
        for symbol in symbols_to_try:
            try:
                print(f"尝试获取 {symbol} 数据...")
                df_test = ak.futures_zh_daily_sina(symbol=symbol)
                
                if df_test is not None and not df_test.empty:
                    print(f"[成功] 成功获取 {symbol} 数据，共 {len(df_test)} 行")
                    print(f"   列名: {df_test.columns.tolist()}")
                    if isinstance(df_test.index, pd.DatetimeIndex):
                        print(f"   日期范围（索引）: {df_test.index.min()} 到 {df_test.index.max()}")
                    df = df_test
                    successful_symbol = symbol
                    break
            except Exception as e:
                error_msg = str(e)[:100].encode('ascii', 'ignore').decode('ascii')
                print(f"   [警告] {symbol} 获取失败: {error_msg}")
                continue
        
        if df is None or df.empty:
            print("[错误] 所有合约代码都无法获取基准数据")
            return None, None
        
        print(f"[成功] 使用合约代码: {successful_symbol}")
        
        # 处理日期列 - akshare返回的数据通常日期在索引或第一列
        if df.index.name == 'date' or isinstance(df.index, pd.DatetimeIndex):
            df['日期'] = pd.to_datetime(df.index)
        elif 'date' in df.columns:
            df['日期'] = pd.to_datetime(df['date'])
        elif '日期' in df.columns:
            df['日期'] = pd.to_datetime(df['日期'])
        elif 'time' in df.columns:
            df['日期'] = pd.to_datetime(df['time'])
        else:
            # 尝试使用索引
            df = df.reset_index()
            if 'date' in df.columns:
                df['日期'] = pd.to_datetime(df['date'])
            elif '日期' in df.columns:
                df['日期'] = pd.to_datetime(df['日期'])
            else:
                # 尝试第一列作为日期
                first_col = df.columns[0]
                try:
                    df['日期'] = pd.to_datetime(df[first_col])
                except:
                    print("无法识别日期列")
                    print(f"可用列: {df.columns.tolist()}")
                    return None, None
        
        # 确保日期列为datetime类型
        df['日期'] = pd.to_datetime(df['日期'])
        
        # 将start_date和end_date转换为datetime类型以便比较
        start_date_dt = pd.to_datetime(start_date)
        end_date_dt = pd.to_datetime(end_date)
        
        # 筛选日期范围
        print(f"筛选日期范围: {start_date} 到 {end_date}")
        df_before = len(df)
        df = df[(df['日期'] >= start_date_dt) & (df['日期'] <= end_date_dt)]
        df = df.sort_values('日期').reset_index(drop=True)
        print(f"筛选后数据: {df_before} 行 -> {len(df)} 行")
        
        if len(df) == 0:
            print(f"[错误] 警告: 在日期范围 {start_date} 到 {end_date} 内没有找到基准数据")
            return None, None
        
        # 显示筛选后的日期范围
        if len(df) > 0:
            print(f"筛选后日期范围: {df['日期'].min()} 到 {df['日期'].max()}")
        
        # 获取收盘价 - akshare通常返回'close'列
        if 'close' in df.columns:
            prices = pd.to_numeric(df['close'], errors='coerce').values
        elif '收盘' in df.columns:
            prices = pd.to_numeric(df['收盘'], errors='coerce').values
        elif '收盘价' in df.columns:
            prices = pd.to_numeric(df['收盘价'], errors='coerce').values
        elif 'close_price' in df.columns:
            prices = pd.to_numeric(df['close_price'], errors='coerce').values
        else:
            # 尝试找到包含'close'或'收盘'的列
            close_cols = [col for col in df.columns if 'close' in col.lower() or '收盘' in col]
            if close_cols:
                prices = pd.to_numeric(df[close_cols[0]], errors='coerce').values
            else:
                print("无法识别收盘价列")
                print(f"可用列: {df.columns.tolist()}")
                return None, None
        
        # 过滤掉NaN值
        valid_mask = ~np.isnan(prices)
        prices_clean = prices[valid_mask]
        dates_filtered = df.loc[valid_mask, '日期'].values
        
        print(f"有效价格数据: {len(prices_clean)} 个")
        if len(prices_clean) > 0:
            print(f"价格范围: {prices_clean.min():.2f} 到 {prices_clean.max():.2f}")
            print(f"价格变化: {((prices_clean[-1] / prices_clean[0]) - 1) * 100:.2f}%")
        
        if len(prices_clean) < 2:
            print("[错误] 有效价格数据不足，无法计算收益率")
            return None, None
        
        # 计算日收益率（简单收益率）
        returns = np.diff(prices_clean) / prices_clean[:-1]
        # 注意：也可以使用对数收益率: returns = np.diff(np.log(prices))
        
        # 对应的日期（去掉第一个日期，因为收益率比价格少一个）
        dates = dates_filtered[1:]
        
        # 确保返回的日期是datetime类型
        if isinstance(dates, np.ndarray):
            dates = pd.to_datetime(dates)
        
        # 验证数据
        total_return = np.prod(1 + returns) - 1
        print(f"[成功] 成功获取基准数据:")
        print(f"   日期范围: {dates[0]} 到 {dates[-1]}")
        print(f"   交易日数: {len(returns)}")
        print(f"   收益率均值: {np.mean(returns)*100:.6f}%")
        print(f"   总收益率（复利）: {total_return*100:.4f}%")
        
        return returns, dates
        
    except Exception as e:
        print(f"获取基准数据时出错: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def get_benchmark_daily_returns_aligned(trade_dates):
    """
    获取与交易日期对齐的基准日收益率
    
    参数:
    trade_dates: 交易日期序列（pandas DatetimeIndex或Series）
    
    返回:
    benchmark_returns: 与交易日期对齐的基准日收益率序列（小数形式）
    """
    # 获取基准数据
    benchmark_returns, benchmark_dates = get_benchmark_returns()
    
    if benchmark_returns is None or benchmark_dates is None:
        print("警告: 无法获取基准数据")
        return None
    
    # 确保基准数据是numpy数组
    if isinstance(benchmark_returns, pd.Series):
        benchmark_returns = benchmark_returns.values
    if isinstance(benchmark_dates, pd.Series):
        benchmark_dates = benchmark_dates.values
    
    # 将基准日期转换为DataFrame以便对齐
    benchmark_df = pd.DataFrame({
        '日期': pd.to_datetime(benchmark_dates),
        '收益率': benchmark_returns
    })
    
    # 将交易日期转换为DataFrame
    trade_df = pd.DataFrame({
        '日期': pd.to_datetime(trade_dates)
    })
    
    # 确保日期格式一致（统一为datetime类型，只保留日期部分）
    benchmark_df['日期'] = pd.to_datetime(benchmark_df['日期']).dt.normalize()
    trade_df['日期'] = pd.to_datetime(trade_df['日期']).dt.normalize()
    
    # 合并数据，使用前向填充（如果某天没有基准数据，使用前一天的收益率）
    merged = trade_df.merge(benchmark_df, on='日期', how='left')
    merged = merged.sort_values('日期').reset_index(drop=True)
    
    # 重要：第一个交易日不应该有收益率（因为需要前一天的价格）
    # 如果第一个交易日没有匹配的基准数据，应该保持为0
    # 后续交易日使用前向填充
    if merged['收益率'].isna().all():
        # 如果全部是NaN，说明日期完全不匹配
        merged['收益率'] = 0
        print("[警告] 所有交易日期都没有匹配的基准数据，全部填充为0")
    else:
        # 先处理第一个交易日：如果没有匹配数据，填充为0
        if pd.isna(merged['收益率'].iloc[0]):
            merged.loc[merged.index[0], '收益率'] = 0
        
        # 然后对剩余数据进行前向填充
        merged['收益率'] = merged['收益率'].ffill()
        
        # 其他NaN值也填充为0
        merged['收益率'] = merged['收益率'].fillna(0)
    
    # 检查对齐结果（统计非零且非NaN的匹配数）
    initial_matched = merged['收益率'].notna().sum()
    after_ffill = (merged['收益率'] != 0).sum() if initial_matched > 0 else 0
    print(f"基准数据对齐: 交易日期数={len(trade_df)}, 基准日期数={len(benchmark_df)}, "
          f"初始匹配数={initial_matched}, 填充后非零数={after_ffill}")
    
    return merged['收益率'].values

if __name__ == "__main__":
    # 测试代码
    print("测试获取基准数据...")
    
    # 测试1: 获取期货数据
    print("\n1. 测试获取AU2404数据:")
    df = get_futures_data("AU2404")
    if df is not None:
        print(f"成功获取数据，共 {len(df)} 行")
        print(f"列名: {df.columns.tolist()}")
        print(f"前5行:\n{df.head()}")
    
    # 测试2: 获取基准收益率（自动从jiaoyi.csv读取日期范围）
    print("\n2. 测试获取基准收益率（自动读取jiaoyi.csv的日期范围）:")
    returns, dates = get_benchmark_returns()  # 不传参数，自动从jiaoyi.csv读取
    if returns is not None:
        print(f"成功获取基准数据，共 {len(returns)} 个交易日")
        print(f"日期范围: {dates[0]} 到 {dates[-1]}")
        print(f"收益率统计: 均值={np.mean(returns)*100:.4f}%, 标准差={np.std(returns)*100:.4f}%")
        print(f"前5个收益率: {returns[:5]}")
    else:
        print("获取基准数据失败")
