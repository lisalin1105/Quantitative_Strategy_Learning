import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import os


def create_sample_config_file(filename, strategy_name):
    """如果文件不存在，则创建一个示例的、使用'固定交易比例'的配置文件"""
    if os.path.exists(filename):
        return

    print(f"未找到配置文件 {filename}，正在创建一个示例文件...")

    settings_data = {
        'Parameter': ['STRATEGY_NAME', 'START_DATE', 'END_DATE', 'INITIAL_CAPITAL'],
        'Value': [strategy_name, '2020-01-01', '2023-12-31', 100000]
    }
    settings_df = pd.DataFrame(settings_data)

    portfolio_data = {
        'Ticker': ['AAPL', 'MSFT', 'TSLA'],
        'Trade_Ratio': [0.10, 0.10, 0.15],
        'BB_WINDOW': [20, 20, 25],
        'BB_STD_DEV': [2.0, 2.0, 2.2]
    }
    portfolio_df = pd.DataFrame(portfolio_data)

    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        settings_df.to_excel(writer, sheet_name='Settings', index=False)
        portfolio_df.to_excel(writer, sheet_name='Portfolio', index=False)

    print("示例文件创建成功。")


def load_config(filepath):
    """从指定的Excel文件加载策略配置和投资组合"""
    try:
        settings_df = pd.read_excel(filepath, sheet_name='Settings')
        settings = settings_df.set_index('Parameter')['Value'].to_dict()
        settings['INITIAL_CAPITAL'] = float(settings['INITIAL_CAPITAL'])

        portfolio_df = pd.read_excel(filepath, sheet_name='Portfolio')

        required_cols = ['Ticker', 'Trade_Ratio', 'BB_WINDOW', 'BB_STD_DEV']
        if not all(col in portfolio_df.columns for col in required_cols):
            raise ValueError(f"错误：'Portfolio' Sheet 缺少必需的列。需要: {required_cols}")

        portfolio_config = portfolio_df.set_index('Ticker')

        print("配置加载成功。")
        return settings, portfolio_config

    except Exception as e:
        print(f"加载配置文件时出错: {e}")
        return None, None


def download_data(tickers, start, end):
    """下载股票的收盘价 (yfinance v0.2.x 之后默认为复权价)"""
    print(f"正在下载 {len(tickers)} 支股票从 {start} 到 {end} 的历史数据...")
    data = yf.download(tickers, start=start, end=end)['Close']
    if data.empty:
        raise ValueError("下载数据失败，请检查股票代码和日期范围。")
    print("数据下载完成。")
    return data


def calculate_bollinger_bands(data, portfolio_config):
    """为每支股票使用其独立的参数计算布林线"""
    print("正在为每支股票计算独立的布林线...")
    all_bands_data = []

    for ticker, params in portfolio_config.iterrows():
        window = int(params['BB_WINDOW'])
        std_dev = float(params['BB_STD_DEV'])
        price_series = data[ticker]
        sma = price_series.rolling(window=window).mean()
        std = price_series.rolling(window=window).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        ticker_bands = pd.concat([price_series, upper_band, lower_band],
                                 axis=1,
                                 keys=['Price', 'BB_Upper', 'BB_Lower'])
        ticker_bands.columns = pd.MultiIndex.from_product([[ticker], ticker_bands.columns])
        all_bands_data.append(ticker_bands)

    full_data = pd.concat(all_bands_data, axis=1)
    return full_data.dropna()


def run_backtest(data, portfolio_config, initial_capital):
    """
    【核心重构】
    执行回测，并记录每个资产的详细表现。
    """
    print("开始执行回测...")
    tickers = portfolio_config.index.tolist()

    cash = initial_capital
    positions = pd.Series(0.0, index=tickers)  # 持有股数
    cost_basis = pd.Series(0.0, index=tickers)  # 累计成本

    history_log = []
    trades_log = []

    for date in data.index:
        current_prices = data.loc[date].xs('Price', level=1)

        # 检查买入信号
        for ticker in tickers:
            price = current_prices[ticker]
            lower_band = data.loc[date, (ticker, 'BB_Lower')]

            if price < lower_band and cash > 1:
                trade_ratio = portfolio_config.loc[ticker, 'Trade_Ratio']
                amount_to_invest = cash * trade_ratio
                investment = min(amount_to_invest, cash)

                if investment > 1:
                    shares_to_buy = investment / price
                    positions[ticker] += shares_to_buy
                    cost_basis[ticker] += investment  # 【新增】累加成本
                    cash -= investment
                    trades_log.append({
                        'Date': date, 'Ticker': ticker, 'Action': 'BUY',
                        'Shares': shares_to_buy, 'Price': price, 'Cost': investment
                    })

        # --- 【核心新增】每日详细状态记录 ---
        market_values = positions * current_prices
        portfolio_market_value = market_values.sum()
        total_value = cash + portfolio_market_value

        daily_record = {
            'Date': date,
            'Total_Value': total_value,
            'Cash': cash,
            'Portfolio_Market_Value': portfolio_market_value
        }

        # 为每支股票记录详细信息
        for ticker in tickers:
            daily_record[f'{ticker}_Market_Value'] = market_values[ticker]
            daily_record[f'{ticker}_Cost_Basis'] = cost_basis[ticker]
            pnl = market_values[ticker] - cost_basis[ticker]
            daily_record[f'{ticker}_PNL'] = pnl
            # 计算ROI，避免除以零错误
            roi = (pnl / cost_basis[ticker]) if cost_basis[ticker] > 0 else 0.0
            daily_record[f'{ticker}_ROI'] = roi

        history_log.append(daily_record)

    print("回测执行完毕。")

    # 将历史记录转换为DataFrame
    portfolio_history_df = pd.DataFrame(history_log).set_index('Date')

    return portfolio_history_df, pd.DataFrame(trades_log)


def analyze_performance(portfolio_history, benchmark_history, settings, trades_log):
    """
    【核心重构】
    分析、展示并保存包含个股详情的回测结果。
    """
    strategy_name = settings['STRATEGY_NAME']
    initial_capital = settings['INITIAL_CAPITAL']
    tickers = [col.split('_')[0] for col in portfolio_history.columns if col.endswith('_ROI')]

    print(f"\n--- “{strategy_name}” 策略回测结果分析 ---")

    # --- 性能计算 ---
    final_stats = portfolio_history.iloc[-1]
    final_value = final_stats['Total_Value']
    total_return = (final_value / initial_capital) - 1
    days = (portfolio_history.index[-1] - portfolio_history.index[0]).days
    annual_return = (1 + total_return) ** (365.0 / days) - 1 if days > 0 else 0

    benchmark_final_value = benchmark_history['Value'].iloc[-1]
    benchmark_total_return = (benchmark_final_value / initial_capital) - 1
    benchmark_annual_return = (1 + benchmark_total_return) ** (365.0 / days) - 1 if days > 0 else 0

    print(f"回测时间范围: {portfolio_history.index[0].date()} to {portfolio_history.index[-1].date()}")
    print(f"初始资金: ${initial_capital:,.2f}")
    print("-" * 40)
    print("整体策略表现:")
    print(f"  最终组合价值: ${final_value:,.2f}")
    print(f"  总收益率: {total_return:.2%}")
    print(f"  年化收益率: {annual_return:.2%}")
    print(f"  总交易次数: {len(trades_log)}")

    # --- 【核心新增】展示个股最终表现 ---
    print("\n个股最终表现:")
    for ticker in tickers:
        roi = final_stats.get(f'{ticker}_ROI', 0)
        pnl = final_stats.get(f'{ticker}_PNL', 0)
        cost = final_stats.get(f'{ticker}_Cost_Basis', 0)
        print(f"  - {ticker}:")
        print(f"    总投资回报率 (ROI): {roi:.2%}")
        print(f"    未实现盈亏 (PNL): ${pnl:,.2f}")
        print(f"    累计投入成本: ${cost:,.2f}")

    print("-" * 40)
    print("基准表现 (买入并持有):")
    print(f"  最终组合价值: ${benchmark_final_value:,.2f}")
    print(f"  总收益率: {benchmark_total_return:.2%}")
    print(f"  年化收益率: {benchmark_annual_return:.2%}")
    print("-" * 40)

    # --- 绘制图表 ---
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(14, 7))
    # 【修改】使用 'Total_Value' 列绘图
    plt.plot(portfolio_history.index, portfolio_history['Total_Value'], label=f'策略: {strategy_name}',
             color='royalblue', linewidth=2)
    plt.plot(benchmark_history.index, benchmark_history['Value'], label='基准 (买入并持有)', color='grey',
             linestyle='--')

    if not trades_log.empty:
        buy_dates = trades_log['Date']
        # 【修改】从 'Total_Value' 列获取买入点的值
        buy_values = portfolio_history.loc[buy_dates]['Total_Value']
        plt.scatter(buy_dates, buy_values, marker='^', color='green', s=100, label='买入点', zorder=5)

    plt.title(f'"{strategy_name}" 投资组合价值回测')
    plt.xlabel('日期')
    plt.ylabel('投资组合价值 ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # --- 保存结果到文件 ---
    results_excel_filename = f"results_{strategy_name}.xlsx"
    chart_filename = f"chart_{strategy_name}.png"

    plt.savefig(chart_filename)
    print(f"图表已保存为: {chart_filename}")

    with pd.ExcelWriter(results_excel_filename, engine='openpyxl') as writer:
        # 【修改】保存包含详细信息的新 portfolio_history
        portfolio_history.to_excel(writer, sheet_name='Portfolio_History')
        if not trades_log.empty:
            trades_log.to_excel(writer, sheet_name='Trades_Log', index=False)
        else:
            pd.DataFrame([{'Message': 'No trades were executed.'}]).to_excel(writer, sheet_name='Trades_Log',
                                                                             index=False)

    print(f"详细数据已保存到Excel文件: {results_excel_filename}")
    print("-" * 40)

    plt.show()


def create_benchmark(price_data, portfolio_config, initial_capital):
    """创建'买入并持有'的基准策略"""
    num_tickers = len(portfolio_config.index)
    weights = np.array([1 / num_tickers] * num_tickers)
    initial_prices = price_data.iloc[0]
    shares_to_buy = (initial_capital * weights) / initial_prices
    benchmark_values = (price_data * shares_to_buy).sum(axis=1)
    return pd.DataFrame({'Value': benchmark_values})


def main():
    """主执行函数"""
    strategy_name = "Fixed_Ratio_Buy"
    config_filename = f"config_{strategy_name}.xlsx"

    create_sample_config_file(config_filename, strategy_name)

    settings, portfolio_config = load_config(config_filename)

    if settings and portfolio_config is not None:
        try:
            tickers = portfolio_config.index.tolist()
            price_data = download_data(tickers, settings['START_DATE'], settings['END_DATE'])
            full_data = calculate_bollinger_bands(price_data, portfolio_config)
            portfolio_history, trades_log = run_backtest(full_data, portfolio_config, settings['INITIAL_CAPITAL'])
            benchmark_history = create_benchmark(full_data.xs('Price', axis=1, level=1), portfolio_config,
                                                 settings['INITIAL_CAPITAL'])
            analyze_performance(portfolio_history, benchmark_history, settings, trades_log)

            print("\n--- 交易日志 (前10条) ---")
            if not trades_log.empty:
                print(trades_log.set_index('Date').head(10))
            else:
                print("在回测期间没有发生任何交易。")

        except Exception as e:
            print(f"\n回测过程中发生严重错误: {e}")


if __name__ == '__main__':
    main()