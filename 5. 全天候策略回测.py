import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


class PortfolioBacktester:
    def __init__(self, initial_capital=100000):
        """
        初始化回测器

        Args:
            initial_capital: 初始资金，默认10万
        """
        self.initial_capital = initial_capital
        self.results = []

    def load_portfolio_config(self, excel_file):
        """
        从Excel文件加载投资组合配置

        Excel格式应包含以下列：
        - 股票code: 股票代码（如AAPL, TSLA等）
        - 投资比例: 占总资金的比例（如0.3表示30%）
        - 投资开始时间: 格式YYYY-MM-DD
        - 投资结束时间: 格式YYYY-MM-DD
        - 是否止盈: True/False
        - 止盈比例: 达到多少收益率时止盈（如0.2表示20%）
        - 止盈卖出比例: 止盈时卖出多少比例（如0.5表示50%）
        """
        try:
            df = pd.read_excel(excel_file)
            required_columns = ['股票code', '投资比例', '投资开始时间', '投资结束时间',
                                '是否止盈', '止盈比例', '止盈卖出比例']

            # 检查必要的列是否存在
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"警告：缺少以下列：{missing_columns}")
                return None

            # 数据类型转换
            df['投资开始时间'] = pd.to_datetime(df['投资开始时间'])
            df['投资结束时间'] = pd.to_datetime(df['投资结束时间'])
            df['是否止盈'] = df['是否止盈'].astype(bool)

            return df

        except Exception as e:
            print(f"读取Excel文件时出错：{e}")
            return None

    def get_stock_data(self, symbol, start_date, end_date):
        """
        获取股票数据
        """
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(start=start_date, end=end_date)
            if data.empty:
                print(f"警告：无法获取 {symbol} 的数据")
                return None
            return data
        except Exception as e:
            print(f"获取 {symbol} 数据时出错：{e}")
            return None

    def calculate_max_drawdown(self, stock_data, initial_price):
        """
        计算最大回撤（修复版）
        """
        try:
            prices = stock_data['Close'].values
            # 计算从初始价格开始的累积收益率
            cumulative_returns = prices / initial_price
            # 计算运行最大值（峰值）
            peak = np.maximum.accumulate(cumulative_returns)
            # 计算回撤
            drawdown = (cumulative_returns - peak) / peak
            # 最大回撤
            max_drawdown = np.min(drawdown)
            return max_drawdown
        except Exception as e:
            print(f"计算最大回撤时出错：{e}")
            return 0

    def calculate_max_drawdown_alternative(self, stock_data):
        """
        基于日收益率计算最大回撤的替代方法
        """
        try:
            # 计算日收益率
            daily_returns = stock_data['Close'].pct_change().dropna()

            if len(daily_returns) == 0:
                return 0

            # 计算累积收益率
            cumulative_returns = (1 + daily_returns).cumprod()

            # 计算运行最大值
            peak = cumulative_returns.expanding(min_periods=1).max()

            # 计算回撤
            drawdown = (cumulative_returns - peak) / peak

            # 最大回撤
            max_drawdown = drawdown.min()

            return max_drawdown
        except Exception as e:
            print(f"计算最大回撤时出错：{e}")
            return 0

    def simulate_position(self, config_row):
        """
        模拟单个股票头寸（修复版）
        """
        symbol = config_row['股票code']
        allocation = config_row['投资比例']
        start_date = config_row['投资开始时间']
        end_date = config_row['投资结束时间']
        take_profit = config_row['是否止盈']
        profit_threshold = config_row['止盈比例'] if pd.notna(config_row['止盈比例']) else 0
        sell_ratio = config_row['止盈卖出比例'] if pd.notna(config_row['止盈卖出比例']) else 0

        print(f"  - 股票: {symbol}")
        print(f"  - 投资比例: {allocation * 100:.1f}%")
        print(f"  - 投资期间: {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}")
        print(f"  - 止盈设置: {'是' if take_profit else '否'}")
        if take_profit:
            print(f"  - 止盈阈值: {profit_threshold * 100:.1f}%")
            print(f"  - 止盈卖出比例: {sell_ratio * 100:.1f}%")

        # 获取股票数据
        stock_data = self.get_stock_data(symbol, start_date, end_date)
        if stock_data is None or len(stock_data) == 0:
            return self.create_empty_result(config_row)

        # 计算投资金额和股数
        investment_amount = self.initial_capital * allocation
        initial_price = stock_data['Close'].iloc[0]
        shares = investment_amount / initial_price

        # 初始化变量
        current_shares = shares
        realized_profit = 0
        transactions = []
        take_profit_triggered = False
        take_profit_date = None

        # 逐日模拟
        for date, row in stock_data.iterrows():
            current_price = row['Close']
            current_return = (current_price - initial_price) / initial_price

            # 检查是否触发止盈（只有当设置了止盈且参数有效时）
            if (take_profit and
                    profit_threshold > 0 and
                    sell_ratio > 0 and
                    not take_profit_triggered and
                    current_return >= profit_threshold):
                # 执行止盈
                shares_to_sell = current_shares * sell_ratio
                sell_amount = shares_to_sell * current_price
                realized_profit += sell_amount - (shares_to_sell * initial_price)
                current_shares -= shares_to_sell
                take_profit_triggered = True
                take_profit_date = date

                transactions.append({
                    'date': date,
                    'action': '止盈卖出',
                    'shares': shares_to_sell,
                    'price': current_price,
                    'amount': sell_amount
                })

        # 计算最终结果
        final_price = stock_data['Close'].iloc[-1]
        final_value = current_shares * final_price
        unrealized_profit = final_value - (current_shares * initial_price)
        total_return = (realized_profit + unrealized_profit) / investment_amount

        # 修复后的最大回撤计算
        max_drawdown = self.calculate_max_drawdown(stock_data, initial_price)

        # 计算夏普比率（简化版本，假设无风险利率为0）
        daily_returns = stock_data['Close'].pct_change().dropna()
        if len(daily_returns) > 1:
            sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0

        # 添加调试信息
        print(f"  - 初始价格: {initial_price:.2f}")
        print(f"  - 最终价格: {final_price:.2f}")
        print(f"  - 收益率: {total_return * 100:.2f}%")
        print(f"  - 最大回撤: {max_drawdown * 100:.2f}%")
        print()

        return {
            '股票code': symbol,
            '投资比例': f"{allocation * 100:.1f}%",
            '投资开始时间': start_date.strftime('%Y-%m-%d'),
            '投资结束时间': end_date.strftime('%Y-%m-%d'),
            '是否止盈': '是' if take_profit else '否',
            '止盈比例': f"{profit_threshold * 100:.1f}%" if profit_threshold > 0 else '未设置',
            '止盈卖出比例': f"{sell_ratio * 100:.1f}%" if sell_ratio > 0 else '未设置',
            '初始价格': round(initial_price, 2),
            '最终价格': round(final_price, 2),
            '投资金额': round(investment_amount, 2),
            '初始股数': round(shares, 2),
            '剩余股数': round(current_shares, 2),
            '最终价值': round(final_value, 2),
            '已实现收益': round(realized_profit, 2),
            '未实现收益': round(unrealized_profit, 2),
            '总收益': round(realized_profit + unrealized_profit, 2),
            '收益率': f"{total_return * 100:.2f}%",
            '是否触发止盈': '是' if take_profit_triggered else '否',
            '止盈日期': take_profit_date.strftime('%Y-%m-%d') if take_profit_date else '',
            '最大回撤': f"{max_drawdown * 100:.2f}%",
            '夏普比率': round(sharpe_ratio, 2),
            '交易次数': len(transactions),
            '交易详情': str(transactions) if transactions else '无'
        }

    def create_empty_result(self, config_row):
        """
        创建空结果（当无法获取数据时）
        """
        return {
            '股票code': config_row['股票code'],
            '投资比例': f"{config_row['投资比例'] * 100:.1f}%",
            '投资开始时间': config_row['投资开始时间'].strftime('%Y-%m-%d'),
            '投资结束时间': config_row['投资结束时间'].strftime('%Y-%m-%d'),
            '是否止盈': '是' if config_row['是否止盈'] else '否',
            '止盈比例': f"{config_row['止盈比例'] * 100:.1f}%" if pd.notna(config_row['止盈比例']) and config_row[
                '止盈比例'] > 0 else '未设置',
            '止盈卖出比例': f"{config_row['止盈卖出比例'] * 100:.1f}%" if pd.notna(config_row['止盈卖出比例']) and
                                                                          config_row['止盈卖出比例'] > 0 else '未设置',
            '初始价格': 0,
            '最终价格': 0,
            '投资金额': 0,
            '初始股数': 0,
            '剩余股数': 0,
            '最终价值': 0,
            '已实现收益': 0,
            '未实现收益': 0,
            '总收益': 0,
            '收益率': "0.00%",
            '是否触发止盈': '否',
            '止盈日期': '',
            '最大回撤': "0.00%",
            '夏普比率': 0,
            '交易次数': 0,
            '交易详情': '数据获取失败'
        }

    def run_backtest(self, config_file, output_file):
        """
        运行完整的回测
        """
        print("开始加载投资组合配置...")
        portfolio_config = self.load_portfolio_config(config_file)

        if portfolio_config is None:
            print("配置文件加载失败，退出回测")
            return

        print(f"成功加载 {len(portfolio_config)} 个投资配置")
        print("开始回测...")

        results = []
        for idx, row in portfolio_config.iterrows():
            print(f"正在回测 {row['股票code']} ({idx + 1}/{len(portfolio_config)})")
            result = self.simulate_position(row)
            results.append(result)

        # 创建结果DataFrame
        results_df = pd.DataFrame(results)

        # 计算组合总体统计（需要先转换百分比格式的数据）
        investment_amounts = []
        final_values = []
        realized_profits = []
        unrealized_profits = []

        for _, row in results_df[results_df['股票code'] != '组合汇总'].iterrows():
            investment_amounts.append(row['投资金额'])
            final_values.append(row['最终价值'])
            realized_profits.append(row['已实现收益'])
            unrealized_profits.append(row['未实现收益'])

        total_investment = sum(investment_amounts)
        total_final_value = sum(final_values)
        total_realized_profit = sum(realized_profits)
        total_unrealized_profit = sum(unrealized_profits)
        total_profit = total_realized_profit + total_unrealized_profit
        portfolio_return = (total_profit / total_investment) * 100 if total_investment > 0 else 0

        # 计算组合级别的最大回撤
        portfolio_max_drawdown = self.calculate_portfolio_max_drawdown(results)

        # 添加汇总行
        allocation_sum = sum(
            [float(row['投资比例'].rstrip('%')) / 100 for row in results if row['股票code'] != '组合汇总'])
        transaction_sum = sum([row['交易次数'] for row in results if row['股票code'] != '组合汇总'])

        summary_row = {
            '股票code': '组合汇总',
            '投资比例': f"{allocation_sum * 100:.1f}%",
            '投资开始时间': '',
            '投资结束时间': '',
            '是否止盈': '',
            '止盈比例': '',
            '止盈卖出比例': '',
            '初始价格': '',
            '最终价格': '',
            '投资金额': round(total_investment, 2),
            '初始股数': '',
            '剩余股数': '',
            '最终价值': round(total_final_value, 2),
            '已实现收益': round(total_realized_profit, 2),
            '未实现收益': round(total_unrealized_profit, 2),
            '总收益': round(total_profit, 2),
            '收益率': f"{portfolio_return:.2f}%",
            '是否触发止盈': '',
            '止盈日期': '',
            '最大回撤': f"{portfolio_max_drawdown:.2f}%",
            '夏普比率': '',
            '交易次数': transaction_sum,
            '交易详情': ''
        }

        # 将汇总行添加到结果中
        results_df = pd.concat([results_df, pd.DataFrame([summary_row])], ignore_index=True)

        # 保存结果到Excel
        try:
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                results_df.to_excel(writer, sheet_name='回测结果', index=False)

                # 创建详细统计表
                profit_triggered_count = sum(
                    1 for row in results if row['是否触发止盈'] == '是' and row['股票code'] != '组合汇总')
                stats_data = {
                    '统计指标': ['初始资金', '总投资金额', '最终价值', '已实现收益',
                                 '未实现收益', '总收益', '总收益率(%)', '组合最大回撤(%)', '成功触发止盈的股票数量'],
                    '数值': [self.initial_capital, total_investment, total_final_value,
                             total_realized_profit, total_unrealized_profit, total_profit,
                             round(portfolio_return, 2), round(portfolio_max_drawdown, 2), profit_triggered_count]
                }
                stats_df = pd.DataFrame(stats_data)
                stats_df.to_excel(writer, sheet_name='组合统计', index=False)

            print(f"\n回测完成！结果已保存到 {output_file}")
            print(f"组合总收益率: {portfolio_return:.2f}%")
            print(f"组合最大回撤: {portfolio_max_drawdown:.2f}%")
            print(f"总收益: {total_profit:.2f} 元")

        except Exception as e:
            print(f"保存结果时出错：{e}")

    def calculate_portfolio_max_drawdown(self, results):
        """
        计算组合级别的最大回撤（简化版本，基于权重平均）
        """
        try:
            total_allocation = 0
            weighted_drawdown = 0

            for result in results:
                if result['股票code'] != '组合汇总':
                    allocation = float(result['投资比例'].rstrip('%')) / 100
                    drawdown = float(result['最大回撤'].rstrip('%'))
                    weighted_drawdown += allocation * drawdown
                    total_allocation += allocation

            if total_allocation > 0:
                return weighted_drawdown / total_allocation
            else:
                return 0

        except Exception as e:
            print(f"计算组合最大回撤时出错：{e}")
            return 0


def create_sample_config():
    """
    创建示例配置文件
    """
    sample_data = {
        '股票code': ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA'],
        '投资比例': [0.3, 0.25, 0.2, 0.15, 0.1],
        '投资开始时间': ['2023-01-01', '2023-01-01', '2023-01-01', '2023-01-01', '2023-01-01'],
        '投资结束时间': ['2024-01-01', '2024-01-01', '2024-01-01', '2024-01-01', '2024-01-01'],
        '是否止盈': [True, True, False, True, False],
        '止盈比例': [0.3, 0.25, 0, 0.4, 0],
        '止盈卖出比例': [0.5, 0.6, 0, 0.3, 0]
    }

    df = pd.DataFrame(sample_data)
    df.to_excel('portfolio_config_sample.xlsx', index=False)
    print("示例配置文件已创建: portfolio_config_sample.xlsx")


def run_backtest_with_existing_config(config_file_path, initial_capital=1000000):
    """
    使用现有的Excel配置文件运行回测

    Args:
        config_file_path: Excel配置文件路径
        initial_capital: 初始资金，默认100万
    """
    import os

    if not os.path.exists(config_file_path):
        print(f"错误：找不到配置文件 {config_file_path}")
        return

    print(f"正在使用配置文件：{config_file_path}")

    # 初始化回测器
    backtester = PortfolioBacktester(initial_capital=initial_capital)

    # 运行回测
    output_file = 'All-Weather Strategy_backtest_results.xlsx'
    backtester.run_backtest(config_file_path, output_file)

    return output_file


def main():
    """
    主函数 - 使用示例
    """
    import os

    # 检查配置文件是否存在，如果不存在才创建示例文件
    config_file = 'portfolio_config_sample.xlsx'
    if not os.path.exists(config_file):
        print("未找到配置文件，创建示例配置文件...")
        create_sample_config()
    else:
        print(f"找到现有配置文件：{config_file}")
        print("将使用现有配置进行回测...")

    # 使用现有配置文件运行回测
    run_backtest_with_existing_config(config_file)

    print("\n使用说明：")
    print("1. 修改现有的 portfolio_config_sample.xlsx 文件中的股票配置")
    print("2. 重新运行程序进行回测")
    print("3. 查看 All-Weather Strategy_backtest_results.xlsx 获取详细结果")
    print("\n或者直接调用：")
    print("run_backtest_with_existing_config('your_config_file.xlsx')")


if __name__ == "__main__":
    main()