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
        - 是否分批建仓: True/False (新增)
        - 初始建仓比例: 初始投资占分配资金的比例 (如0.5) (新增)
        - 后续建仓阈值: 相比初始价下跌多少时买入剩余部分 (如-0.2) (新增)
        - 是否低位加仓: True/False
        - 低位加仓阈值: 下跌多少比例时加仓（如-0.2表示下跌20%）
        - 低位加仓比例: 加仓金额占初始投资的比例（如0.3表示30%）
        - 最大加仓次数: 最多加仓几次（如3次）
        """
        try:
            df = pd.read_excel(excel_file)
            # --- 新增列到必要列清单 ---
            required_columns = ['股票code', '投资比例', '投资开始时间', '投资结束时间',
                                '是否止盈', '止盈比例', '止盈卖出比例',
                                '是否分批建仓', '初始建仓比例', '后续建仓阈值',
                                '是否低位加仓', '低位加仓阈值', '低位加仓比例', '最大加仓次数']

            # 检查必要的列是否存在
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"警告：缺少以下列：{missing_columns}")
                # 为缺失的新列添加默认值
                for col in missing_columns:
                    if col == '是否止盈' or col == '是否低位加仓' or col == '是否分批建仓':
                        df[col] = False
                    elif col == '初始建仓比例':
                        df[col] = 1.0  # 默认为100%建仓，保持旧版兼容
                    elif col in ['止盈比例', '止盈卖出比例', '后续建仓阈值', '低位加仓阈值', '低位加仓比例']:
                        df[col] = 0
                    elif col == '最大加仓次数':
                        df[col] = 0

            # 数据类型转换
            df['投资开始时间'] = pd.to_datetime(df['投资开始时间'])
            df['投资结束时间'] = pd.to_datetime(df['投资结束时间'])
            df['是否止盈'] = df['是否止盈'].astype(bool)
            df['是否分批建仓'] = df['是否分批建仓'].astype(bool)  # 新增
            df['是否低位加仓'] = df['是否低位加仓'].astype(bool)

            return df

        except Exception as e:
            print(f"读取Excel文件时出错：{e}")
            return None

    def get_stock_data(self, symbol, start_date, end_date):
        """
        获取股票数据（包括分红信息）
        """
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(start=start_date, end=end_date)
            if data.empty:
                print(f"警告：无法获取 {symbol} 的价格数据")
                return None, None

            try:
                dividends = stock.dividends
                if not dividends.empty:
                    dividends = dividends[(dividends.index >= start_date) & (dividends.index <= end_date)]
                else:
                    dividends = pd.Series(dtype=float, name='Dividends')
            except Exception as e:
                print(f"警告：无法获取 {symbol} 的分红数据: {e}")
                dividends = pd.Series(dtype=float, name='Dividends')

            return data, dividends

        except Exception as e:
            print(f"获取 {symbol} 数据时出错：{e}")
            return None, None

    def calculate_max_drawdown(self, stock_data, initial_price):
        """
        计算最大回撤
        """
        try:
            prices = stock_data['Close'].values
            cumulative_returns = prices / initial_price
            peak = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - peak) / peak
            max_drawdown = np.min(drawdown)
            return max_drawdown
        except Exception as e:
            print(f"计算最大回撤时出错：{e}")
            return 0

    def calculate_dividend_income(self, dividends, shares_history, transactions):
        """
        计算分红收入
        """
        total_dividend_income = 0
        dividend_count = 0
        dividend_details = []

        if dividends.empty:
            return 0, 0, []

        for dividend_date, dividend_per_share in dividends.items():
            shares_at_dividend_date = 0
            for date, shares in shares_history.items():
                if date <= dividend_date:
                    shares_at_dividend_date = shares
                else:
                    break

            if shares_at_dividend_date > 0:
                dividend_income = shares_at_dividend_date * dividend_per_share
                total_dividend_income += dividend_income
                dividend_count += 1
                dividend_details.append({
                    'date': dividend_date,
                    'dividend_per_share': dividend_per_share,
                    'shares': shares_at_dividend_date,
                    'total_dividend': dividend_income
                })
                print(f"    分红: {dividend_date.strftime('%Y-%m-%d')}, "
                      f"每股分红: ${dividend_per_share:.4f}, "
                      f"持股: {shares_at_dividend_date:.2f}, "
                      f"分红收入: ${dividend_income:.2f}")

        return total_dividend_income, dividend_count, dividend_details

    def simulate_position(self, config_row):
        """
        模拟单个股票头寸（支持分批建仓、低位加仓和分红）
        """
        symbol = config_row['股票code']
        allocation = config_row['投资比例']
        start_date = config_row['投资开始时间']
        end_date = config_row['投资结束时间']

        # 止盈参数
        take_profit = config_row['是否止盈']
        profit_threshold = config_row['止盈比例'] if pd.notna(config_row['止盈比例']) else 0
        sell_ratio = config_row['止盈卖出比例'] if pd.notna(config_row['止盈卖出比例']) else 0

        # --- 新增：分批建仓参数 ---
        staged_investment = config_row['是否分批建仓'] if pd.notna(config_row['是否分批建仓']) else False
        initial_buy_ratio = config_row['初始建仓比例'] if pd.notna(config_row['初始建仓比例']) else 1.0
        staged_buy_threshold = config_row['后续建仓阈值'] if pd.notna(config_row['后续建仓阈值']) else 0

        # 低位加仓(DCA)参数
        enable_dca = config_row['是否低位加仓'] if pd.notna(config_row['是否低位加仓']) else False
        dca_threshold = config_row['低位加仓阈值'] if pd.notna(config_row['低位加仓阈值']) else 0
        dca_ratio = config_row['低位加仓比例'] if pd.notna(config_row['低位加仓比例']) else 0
        max_dca_times = int(config_row['最大加仓次数']) if pd.notna(config_row['最大加仓次数']) else 0

        print(f"  - 股票: {symbol}")
        print(f"  - 投资比例: {allocation * 100:.1f}%")
        print(f"  - 投资期间: {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}")
        print(f"  - 止盈设置: {'是' if take_profit else '否'}")
        if take_profit:
            print(f"  - 止盈阈值: {profit_threshold * 100:.1f}%")
            print(f"  - 止盈卖出比例: {sell_ratio * 100:.1f}%")
        # --- 新增：打印分批建仓信息 ---
        print(f"  - 分批建仓: {'是' if staged_investment else '否'}")
        if staged_investment:
            print(f"  - 初始建仓比例: {initial_buy_ratio * 100:.1f}%")
            print(f"  - 后续建仓阈值: {staged_buy_threshold * 100:.1f}% (相对初始价)")
        print(f"  - 低位加仓(DCA): {'是' if enable_dca else '否'}")
        if enable_dca:
            print(f"  - 加仓阈值: {dca_threshold * 100:.1f}% (相对上次买入价)")
            print(f"  - 加仓比例: {dca_ratio * 100:.1f}%")
            print(f"  - 最大加仓次数: {max_dca_times}次")

        stock_data, dividends = self.get_stock_data(symbol, start_date, end_date)
        if stock_data is None or len(stock_data) == 0:
            return self.create_empty_result(config_row)

        # --- 修改：初始投资计算 ---
        total_allocated_capital = self.initial_capital * allocation
        initial_investment = total_allocated_capital * initial_buy_ratio
        remaining_investment = total_allocated_capital - initial_investment if staged_investment else 0

        initial_price = stock_data['Close'].iloc[0]
        initial_shares = initial_investment / initial_price

        # 初始化变量
        current_shares = initial_shares
        total_investment = initial_investment
        realized_profit = 0
        transactions = []
        take_profit_triggered = False
        take_profit_date = None

        # --- 新增：分批建仓状态变量 ---
        staged_buy_triggered = False

        # 低位加仓(DCA)相关变量
        dca_count = 0
        last_dca_price = initial_price
        dca_transactions = []

        shares_history = {start_date: initial_shares}

        # 逐日模拟
        for date, row in stock_data.iterrows():
            current_price = row['Close']

            # --- 新增：检查是否触发后续建仓 ---
            if staged_investment and not staged_buy_triggered and remaining_investment > 0:
                decline_from_initial = (current_price - initial_price) / initial_price
                if decline_from_initial <= staged_buy_threshold:
                    # 执行后续建仓
                    staged_buy_shares = remaining_investment / current_price
                    current_shares += staged_buy_shares
                    total_investment += remaining_investment

                    staged_buy_triggered = True
                    last_dca_price = current_price  # 更新价格，用于后续DCA判断

                    shares_history[date] = current_shares

                    staged_buy_transaction = {
                        'date': date,
                        'action': '后续建仓',
                        'shares': staged_buy_shares,
                        'price': current_price,
                        'amount': remaining_investment,
                        'decline_from_initial': decline_from_initial
                    }
                    transactions.append(staged_buy_transaction)
                    remaining_investment = 0  # 剩余资金清零

                    print(f"    触发后续建仓: {date.strftime('%Y-%m-%d')}, 价格: {current_price:.2f}, "
                          f"初始价跌幅: {decline_from_initial * 100:.2f}%, "
                          f"买入金额: {staged_buy_transaction['amount']:.2f}")

            # 检查是否触发低位加仓(DCA)
            decline_from_last_buy = (current_price - last_dca_price) / last_dca_price
            if (enable_dca and
                    dca_threshold < 0 and
                    dca_ratio > 0 and
                    dca_count < max_dca_times and
                    decline_from_last_buy <= dca_threshold):
                # 注意：这里的加仓金额是基于最开始的投资额，而不是总分配额
                dca_base_investment = self.initial_capital * allocation * initial_buy_ratio
                dca_amount = dca_base_investment * dca_ratio
                dca_shares = dca_amount / current_price
                current_shares += dca_shares
                total_investment += dca_amount
                dca_count += 1
                last_dca_price = current_price

                shares_history[date] = current_shares
                dca_transaction = {
                    'date': date,
                    'action': f'第{dca_count}次DCA',
                    'shares': dca_shares,
                    'price': current_price,
                    'amount': dca_amount,
                    'decline': decline_from_last_buy
                }
                dca_transactions.append(dca_transaction)
                transactions.append(dca_transaction)
                print(f"    触发DCA加仓: {date.strftime('%Y-%m-%d')}, 价格: {current_price:.2f}, "
                      f"上次买入价跌幅: {decline_from_last_buy * 100:.2f}%, 加仓金额: {dca_amount:.2f}")

            # 检查是否触发止盈
            current_return_on_avg_cost = (current_price - (total_investment / current_shares)) / (
                        total_investment / current_shares) if current_shares > 0 else 0
            if (take_profit and
                    profit_threshold > 0 and
                    sell_ratio > 0 and
                    not take_profit_triggered and
                    current_return_on_avg_cost >= profit_threshold):
                shares_to_sell = current_shares * sell_ratio
                sell_amount = shares_to_sell * current_price
                avg_cost = total_investment / current_shares
                realized_profit += sell_amount - (shares_to_sell * avg_cost)
                current_shares -= shares_to_sell
                shares_history[date] = current_shares
                take_profit_triggered = True
                take_profit_date = date
                stop_profit_transaction = {
                    'date': date,
                    'action': '止盈卖出',
                    'shares': shares_to_sell,
                    'price': current_price,
                    'amount': sell_amount,
                    'avg_cost': avg_cost
                }
                transactions.append(stop_profit_transaction)

        total_dividend_income, dividend_count, dividend_details = self.calculate_dividend_income(
            dividends, shares_history, transactions)

        final_price = stock_data['Close'].iloc[-1]
        final_value = current_shares * final_price

        total_shares_bought = initial_shares + sum(
            [t['shares'] for t in transactions if '买' in t['action'] or '建仓' in t['action'] or 'DCA' in t['action']])
        avg_cost = total_investment / total_shares_bought if total_shares_bought > 0 else initial_price

        unrealized_profit = current_shares * (final_price - avg_cost)
        total_profit = realized_profit + unrealized_profit + total_dividend_income
        total_return = total_profit / total_investment if total_investment > 0 else 0
        max_drawdown = self.calculate_max_drawdown(stock_data, initial_price)

        daily_returns = stock_data['Close'].pct_change().dropna()
        sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if len(daily_returns) > 1 else 0
        dividend_yield = (total_dividend_income / total_investment) * 100 if total_investment > 0 else 0

        print(f"  - 初始价格: {initial_price:.2f}")
        print(f"  - 最终价格: {final_price:.2f}")
        print(f"  - 平均成本: {avg_cost:.2f}")
        print(f"  - 总投资: {total_investment:.2f}")
        print(f"  - 加仓次数(DCA): {dca_count}")
        print(f"  - 分红收入: ${total_dividend_income:.2f} ({dividend_count}次)")
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
            # --- 新增结果列 ---
            '是否分批建仓': '是' if staged_investment else '否',
            '初始建仓比例': f"{initial_buy_ratio * 100:.1f}%" if staged_investment else 'N/A',
            '后续建仓阈值': f"{staged_buy_threshold * 100:.1f}%" if staged_investment else 'N/A',
            '是否触发后续建仓': '是' if staged_buy_triggered else '否',
            # ---
            '是否低位加仓': '是' if enable_dca else '否',
            '低位加仓阈值': f"{dca_threshold * 100:.1f}%" if dca_threshold < 0 else '未设置',
            '低位加仓比例': f"{dca_ratio * 100:.1f}%" if dca_ratio > 0 else '未设置',
            '最大加仓次数': max_dca_times if max_dca_times > 0 else '未设置',
            '实际加仓次数(DCA)': dca_count,
            '初始价格': round(initial_price, 2),
            '最终价格': round(final_price, 2),
            '平均持股成本': round(avg_cost, 2),
            '初始投资': round(initial_investment, 2),
            '总投资': round(total_investment, 2),
            '初始股数': round(initial_shares, 2),
            '最终股数': round(current_shares, 2),
            '最终价值': round(final_value, 2),
            '已实现收益': round(realized_profit, 2),
            '未实现收益': round(unrealized_profit, 2),
            '分红收入': round(total_dividend_income, 2),
            '分红次数': dividend_count,
            '分红收益率': f"{dividend_yield:.2f}%",
            '总收益': round(total_profit, 2),
            '收益率': f"{total_return * 100:.2f}%",
            '是否触发止盈': '是' if take_profit_triggered else '否',
            '止盈日期': take_profit_date.strftime('%Y-%m-%d') if take_profit_date else '',
            '最大回撤': f"{max_drawdown * 100:.2f}%",
            '夏普比率': round(sharpe_ratio, 2),
            '交易次数': len(transactions),
            '交易详情': str(transactions) if transactions else '无',
            '分红详情': str(dividend_details) if dividend_details else '无分红'
        }

    def create_empty_result(self, config_row):
        """
        创建空结果（当无法获取数据时）
        """
        # --- 更新以包含新列 ---
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
            '是否分批建仓': '是' if config_row.get('是否分批建仓', False) else '否',
            '初始建仓比例': f"{config_row.get('初始建仓比例', 1.0) * 100:.1f}%" if config_row.get('是否分批建仓',
                                                                                                  False) else 'N/A',
            '后续建仓阈值': f"{config_row.get('后续建仓阈值', 0) * 100:.1f}%" if config_row.get('是否分批建仓',
                                                                                                False) else 'N/A',
            '是否触发后续建仓': '否',
            '是否低位加仓': '是' if config_row.get('是否低位加仓', False) else '否',
            '低位加仓阈值': f"{config_row.get('低位加仓阈值', 0) * 100:.1f}%" if config_row.get('低位加仓阈值',
                                                                                                0) < 0 else '未设置',
            '低位加仓比例': f"{config_row.get('低位加仓比例', 0) * 100:.1f}%" if config_row.get('低位加仓比例',
                                                                                                0) > 0 else '未设置',
            '最大加仓次数': config_row.get('最大加仓次数', 0) if config_row.get('最大加仓次数', 0) > 0 else '未设置',
            '实际加仓次数(DCA)': 0,
            '初始价格': 0, '最终价格': 0, '平均持股成本': 0, '初始投资': 0, '总投资': 0,
            '初始股数': 0, '最终股数': 0, '最终价值': 0, '已实现收益': 0, '未实现收益': 0,
            '分红收入': 0, '分红次数': 0, '分红收益率': "0.00%", '总收益': 0, '收益率': "0.00%",
            '是否触发止盈': '否', '止盈日期': '', '最大回撤': "0.00%", '夏普比率': 0, '交易次数': 0,
            '交易详情': '数据获取失败', '分红详情': '数据获取失败'
        }

    def calculate_portfolio_max_drawdown(self, results):
        """
        计算组合级别的最大回撤（简化版本）
        """
        try:
            max_drawdowns = []
            for result in results:
                if result['股票code'] != '组合汇总' and result['最大回撤'] != "0.00%":
                    drawdown_str = result['最大回撤'].rstrip('%')
                    if drawdown_str and drawdown_str != '0.00':
                        max_drawdowns.append(float(drawdown_str))

            if max_drawdowns:
                return sum(max_drawdowns) / len(max_drawdowns)
            else:
                return 0.0
        except Exception as e:
            print(f"计算组合最大回撤时出错：{e}")
            return 0.0

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

        results_df = pd.DataFrame(results)

        # 计算组合总体统计
        total_initial_investment = 0
        total_final_investment = 0
        total_final_value = 0
        total_realized_profit = 0
        total_unrealized_profit = 0
        total_dividend_income = 0

        for _, row in results_df[results_df['股票code'] != '组合汇总'].iterrows():
            total_initial_investment += row['初始投资']
            total_final_investment += row['总投资']
            total_final_value += row['最终价值']
            total_realized_profit += row['已实现收益']
            total_unrealized_profit += row['未实现收益']
            total_dividend_income += row['分红收入']

        total_profit = total_realized_profit + total_unrealized_profit + total_dividend_income
        portfolio_return = (total_profit / total_final_investment) * 100 if total_final_investment > 0 else 0
        portfolio_dividend_yield = (
                                               total_dividend_income / total_final_investment) * 100 if total_final_investment > 0 else 0
        portfolio_max_drawdown = self.calculate_portfolio_max_drawdown(results)

        allocation_sum = sum(
            [float(row['投资比例'].rstrip('%')) / 100 for row in results if row['股票code'] != '组合汇总'])
        transaction_sum = sum([row['交易次数'] for row in results if row['股票code'] != '组合汇总'])
        total_dca_count = sum([row['实际加仓次数(DCA)'] for row in results if row['股票code'] != '组合汇总'])
        total_dividend_count = sum([row['分红次数'] for row in results if row['股票code'] != '组合汇总'])

        summary_row = {
            '股票code': '组合汇总',
            '投资比例': f"{allocation_sum * 100:.1f}%",
            '投资开始时间': '', '投资结束时间': '', '是否止盈': '', '止盈比例': '', '止盈卖出比例': '',
            # --- 新增汇总行占位符 ---
            '是否分批建仓': '', '初始建仓比例': '', '后续建仓阈值': '', '是否触发后续建仓': '',
            # ---
            '是否低位加仓': '', '低位加仓阈值': '', '低位加仓比例': '', '最大加仓次数': '',
            '实际加仓次数(DCA)': total_dca_count,
            '初始价格': '', '最终价格': '', '平均持股成本': '',
            '初始投资': round(total_initial_investment, 2),
            '总投资': round(total_final_investment, 2),
            '初始股数': '', '最终股数': '',
            '最终价值': round(total_final_value, 2),
            '已实现收益': round(total_realized_profit, 2),
            '未实现收益': round(total_unrealized_profit, 2),
            '分红收入': round(total_dividend_income, 2),
            '分红次数': total_dividend_count,
            '分红收益率': f"{portfolio_dividend_yield:.2f}%",
            '总收益': round(total_profit, 2),
            '收益率': f"{portfolio_return:.2f}%",
            '是否触发止盈': '', '止盈日期': '',
            '最大回撤': f"{portfolio_max_drawdown:.2f}%",
            '夏普比率': '', '交易次数': transaction_sum, '交易详情': '', '分红详情': ''
        }

        results_df = pd.concat([results_df, pd.DataFrame([summary_row])], ignore_index=True)

        try:
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                results_df.to_excel(writer, sheet_name='回测结果', index=False)

                profit_triggered_count = sum(
                    1 for row in results if row['是否触发止盈'] == '是' and row['股票code'] != '组合汇总')
                staged_investment_count = sum(
                    1 for row in results if row['是否分批建仓'] == '是' and row['股票code'] != '组合汇总')
                staged_buy_triggered_count = sum(
                    1 for row in results if row['是否触发后续建仓'] == '是' and row['股票code'] != '组合汇总')
                dca_enabled_count = sum(
                    1 for row in results if row['是否低位加仓'] == '是' and row['股票code'] != '组合汇总')
                dividend_stocks_count = sum(
                    1 for row in results if row['分红次数'] > 0 and row['股票code'] != '组合汇总')

                stats_data = {
                    '统计指标': ['初始资金', '初始投资金额', '总投资金额(含加仓)', '最终价值',
                                 '已实现收益', '未实现收益', '分红收入', '总收益',
                                 '总收益率(%)', '分红收益率(%)', '组合最大回撤(%)',
                                 '启用分批建仓的股票数量', '成功触发后续建仓的股票数量',
                                 '成功触发止盈的股票数量', '启用低位加仓(DCA)的股票数量',
                                 '总加仓次数(DCA)', '有分红的股票数量', '总分红次数'],
                    '数值': [self.initial_capital, total_initial_investment, total_final_investment,
                             total_final_value, total_realized_profit, total_unrealized_profit,
                             total_dividend_income, total_profit, round(portfolio_return, 2),
                             round(portfolio_dividend_yield, 2), round(portfolio_max_drawdown, 2),
                             staged_investment_count, staged_buy_triggered_count,
                             profit_triggered_count, dca_enabled_count, total_dca_count,
                             dividend_stocks_count, total_dividend_count]
                }
                stats_df = pd.DataFrame(stats_data)
                stats_df.to_excel(writer, sheet_name='组合统计', index=False)

                dividend_details = []
                for result in results:
                    if result['分红详情'] != '无分红' and result['分红详情'] != '数据获取失败':
                        try:
                            details = eval(result['分红详情'])
                            for detail in details:
                                dividend_details.append({
                                    '股票代码': result['股票code'],
                                    '分红日期': detail['date'].strftime('%Y-%m-%d'),
                                    '每股分红($)': round(detail['dividend_per_share'], 4),
                                    '持股数量': round(detail['shares'], 2),
                                    '分红收入($)': round(detail['total_dividend'], 2)
                                })
                        except Exception as e:
                            print(f"处理 {result['股票code']} 分红详情时出错: {e}")
                            continue

                if dividend_details:
                    dividend_df = pd.DataFrame(dividend_details)
                    dividend_df.to_excel(writer, sheet_name='分红明细', index=False)
                else:
                    empty_dividend_df = pd.DataFrame(
                        {'股票代码': ['无'], '分红日期': [''], '每股分红($)': [0], '持股数量': [0], '分红收入($)': [0]})
                    empty_dividend_df.to_excel(writer, sheet_name='分红明细', index=False)

            print(f"回测完成！结果已保存到: {output_file}")

            print("\n=== 回测结果汇总 ===")
            print(f"初始资金: ${self.initial_capital:,.2f}")
            print(f"总投资金额(含加仓): ${total_final_investment:,.2f}")
            print(f"最终价值: ${total_final_value:,.2f}")
            print(f"总收益: ${total_profit:,.2f}")
            print(f"总收益率: {portfolio_return:.2f}%")
            print(f"分红收入: ${total_dividend_income:,.2f}")
            print(f"分红收益率: {portfolio_dividend_yield:.2f}%")
            print(f"组合最大回撤: {portfolio_max_drawdown:.2f}%")
            print(f"总交易次数: {transaction_sum}")

        except Exception as e:
            print(f"保存Excel文件时出错：{e}")


# 使用示例
if __name__ == "__main__":
    # 初始化回测器
    backtester = PortfolioBacktester(initial_capital=100000)

    # 运行回测
    # 请确保您的Excel文件已更新，包含了新的列
    config_file = "portfolio_config.xlsx"
    output_file = "backtest_results.xlsx"

    backtester.run_backtest(config_file, output_file)
