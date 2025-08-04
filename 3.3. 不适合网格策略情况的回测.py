import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ETFBacktester:
    def __init__(self):
        self.config = {}
        self.results = {}

    def create_config_template(self, filename="backtest_config.xlsx"):
        """创建配置文件模板"""
        config_data = {
            '参数名称': [
                '股票代码',
                '开始日期',
                '结束日期',
                '定投金额',
                '定投频率(天)',
                '止盈阈值(%)',
                '止盈卖出比例(%)',
                '是否启用止盈'
            ],
            '参数值': [
                '515100.SS',
                '2024-07-01',
                '2024-12-31',
                50,
                30,
                10,
                80,
                'YES'
            ],
            '说明': [
                'ETF代码，如515100.SS',
                '开始投资日期，格式：YYYY-MM-DD',
                '结束日期，格式：YYYY-MM-DD',
                '每次定投金额（元）',
                '多少天定投一次（如30表示每月）',
                '达到多少收益率时止盈',
                '止盈时卖出持仓的百分比',
                'YES启用止盈，NO不启用'
            ]
        }

        config_df = pd.DataFrame(config_data)

        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            config_df.to_excel(writer, sheet_name='配置参数', index=False)

        print(f"配置文件模板已创建: {filename}")
        print("请编辑配置文件中的参数值，然后重新运行程序")
        return filename

    def load_config(self, filename="backtest_config.xlsx"):
        """从Excel加载配置"""
        try:
            df = pd.read_excel(filename, sheet_name='配置参数')
            config_dict = dict(zip(df['参数名称'], df['参数值']))

            self.config = {
                'symbol': config_dict['股票代码'],
                'start_date': config_dict['开始日期'],
                'end_date': config_dict['结束日期'],
                'investment_amount': float(config_dict['定投金额']),
                'investment_frequency': int(config_dict['定投频率(天)']),
                'profit_threshold': float(config_dict['止盈阈值(%)']) / 100,
                'sell_ratio': float(config_dict['止盈卖出比例(%)']) / 100,
                'enable_profit_taking': config_dict['是否启用止盈'].upper() == 'YES'
            }

            print("配置加载成功:")
            for key, value in self.config.items():
                print(f"  {key}: {value}")
            return True

        except FileNotFoundError:
            print(f"配置文件 {filename} 不存在，将创建模板")
            self.create_config_template(filename)
            return False
        except Exception as e:
            print(f"加载配置文件时出错: {e}")
            return False

    def get_etf_data(self):
        """获取ETF数据"""
        try:
            stock = yf.Ticker(self.config['symbol'])
            data = stock.history(start=self.config['start_date'], end=self.config['end_date'])
            if data.empty:
                print("无法获取数据，请检查股票代码和日期范围")
                return None
            return data
        except Exception as e:
            print(f"获取数据时出错: {e}")
            return None

    def calculate_max_drawdown(self, portfolio_values):
        """计算最大回撤"""
        if len(portfolio_values) == 0:
            return 0, 0, None, None

        # 计算累计最高点
        cumulative_max = np.maximum.accumulate(portfolio_values)

        # 计算回撤
        drawdowns = (portfolio_values - cumulative_max) / cumulative_max

        # 找到最大回撤
        max_drawdown = np.min(drawdowns)
        max_drawdown_idx = np.argmin(drawdowns)

        # 找到最大回撤对应的峰值
        peak_idx = np.argmax(cumulative_max[:max_drawdown_idx + 1]) if max_drawdown_idx > 0 else 0

        return max_drawdown, max_drawdown_idx, peak_idx, drawdowns

    def calculate_average_cost(self, total_cost, total_shares):
        """计算平均成本"""
        return total_cost / total_shares if total_shares > 0 else 0

    def backtest(self):
        """执行回测"""
        data = self.get_etf_data()
        if data is None:
            return None

        # 初始化变量
        total_shares = 0  # 当前持有份额
        total_cost = 0  # 当前持仓的总成本
        cash_in = 0  # 累计投入现金
        cash_out = 0  # 累计卖出获得现金
        realized_profit = 0  # 已实现利润
        transactions = []
        portfolio_values = []  # 组合价值历史
        investment_dates = []

        # 生成投资日期
        start_date = pd.Timestamp(self.config['start_date'])
        end_date = pd.Timestamp(self.config['end_date'])
        frequency_days = self.config['investment_frequency']

        # 处理时区问题
        if data.index.tz is not None:
            start_date = start_date.tz_localize(data.index.tz)
            end_date = end_date.tz_localize(data.index.tz)

        current_date = start_date
        investment_count = 0

        while current_date <= end_date:
            # 找到最接近的交易日
            available_dates = data.index[data.index >= current_date]
            if len(available_dates) == 0:
                break

            invest_date = available_dates[0]
            if invest_date > end_date:
                break

            price = data.loc[invest_date, 'Close']

            # 定投买入
            shares_bought = self.config['investment_amount'] / price
            total_shares += shares_bought
            total_cost += self.config['investment_amount']  # 增加持仓成本
            cash_in += self.config['investment_amount']  # 增加现金投入
            investment_count += 1

            current_market_value = total_shares * price
            current_unrealized_pnl = current_market_value - total_cost

            # 记录买入交易
            transaction = {
                'date': invest_date,
                'type': 'buy',
                'price': price,
                'shares': shares_bought,
                'amount': self.config['investment_amount'],
                'total_shares': total_shares,
                'total_cost': total_cost,
                'average_cost': self.calculate_average_cost(total_cost, total_shares),
                'current_market_value': current_market_value,
                'unrealized_pnl': current_unrealized_pnl,
                'unrealized_return_rate': (current_unrealized_pnl / total_cost * 100) if total_cost > 0 else 0,
                'realized_profit': realized_profit,
                'cash_in': cash_in,
                'cash_out': cash_out
            }
            transactions.append(transaction)

            # 检查止盈条件
            if (self.config['enable_profit_taking'] and
                    total_cost > 0 and
                    (current_market_value / total_cost - 1) >= self.config['profit_threshold']):
                # 执行止盈
                shares_to_sell = total_shares * self.config['sell_ratio']
                sell_amount = shares_to_sell * price

                # 计算卖出部分的成本（按比例分摊）
                cost_of_sold_shares = total_cost * self.config['sell_ratio']

                # 计算这次止盈的利润
                profit_from_sale = sell_amount - cost_of_sold_shares

                # 更新持仓
                total_shares -= shares_to_sell
                total_cost -= cost_of_sold_shares  # 减少持仓成本
                cash_out += sell_amount  # 增加卖出获得的现金
                realized_profit += profit_from_sale  # 增加已实现利润

                # 记录卖出交易
                sell_transaction = {
                    'date': invest_date,
                    'type': 'sell',
                    'price': price,
                    'shares': shares_to_sell,
                    'amount': sell_amount,
                    'cost_of_sold_shares': cost_of_sold_shares,
                    'profit_from_sale': profit_from_sale,
                    'total_shares': total_shares,
                    'total_cost': total_cost,
                    'average_cost': self.calculate_average_cost(total_cost, total_shares),
                    'realized_profit': realized_profit,
                    'cash_in': cash_in,
                    'cash_out': cash_out
                }
                transactions.append(sell_transaction)

                print(
                    f"{invest_date.strftime('%Y-%m-%d')}: 触发止盈，卖出{shares_to_sell:.2f}份，卖出金额{sell_amount:.2f}元，成本{cost_of_sold_shares:.2f}元，获利{profit_from_sale:.2f}元")

            # 计算当前总资产价值（用于最大回撤计算）
            remaining_market_value = total_shares * price
            # total_assets = remaining_market_value + realized_profit
            total_assets = remaining_market_value + cash_out
            portfolio_values.append(total_assets)
            investment_dates.append(invest_date)

            print(
                f"{invest_date.strftime('%Y-%m-%d')}: 第{investment_count}次定投，买入{shares_bought:.2f}份，价格{price:.4f}元，平均成本{self.calculate_average_cost(total_cost, total_shares):.4f}元")

            # 移动到下一个投资日期
            current_date += timedelta(days=frequency_days)

        # 计算最终结果
        if len(transactions) == 0:
            print("没有执行任何交易")
            return None

        final_price = data.iloc[-1]['Close']
        final_market_value = total_shares * final_price  # 剩余持仓市值

        # 总资产 = 剩余持仓市值 + 已实现利润
        total_assets = final_market_value + cash_out

        # 净投入 = 投入现金 - 卖出获得现金
        net_cash_invested = cash_in - cash_out

        # 总收益 = 总资产 - 投入现金
        total_return = total_assets - cash_in

        # 收益率 = 总收益 / 投入现金 * 100%
        total_return_rate = (total_return / cash_in * 100) if cash_in > 0 else 0

        # 剩余持仓的未实现盈亏
        unrealized_pnl = final_market_value - total_cost

        # 计算最大回撤
        max_drawdown, max_dd_idx, peak_idx, drawdowns = self.calculate_max_drawdown(np.array(portfolio_values))

        self.results = {
            'transactions': transactions,
            'cash_in': cash_in,  # 累计投入现金
            'cash_out': cash_out,  # 累计卖出获得现金
            'net_cash_invested': net_cash_invested,  # 净投入现金
            'remaining_shares': total_shares,  # 剩余持有份额
            'remaining_cost': total_cost,  # 剩余持仓成本
            'average_cost': self.calculate_average_cost(total_cost, total_shares),  # 平均成本
            'final_price': final_price,  # 最终价格
            'final_market_value': final_market_value,  # 剩余持仓市值
            'unrealized_pnl': unrealized_pnl,  # 未实现盈亏
            'realized_profit': realized_profit,  # 已实现利润
            'total_assets': total_assets,  # 总资产
            'total_return': total_return,  # 总收益
            'total_return_rate': total_return_rate,  # 总收益率
            'max_drawdown': max_drawdown * 100,  # 最大回撤
            'investment_count': investment_count,  # 投资次数
            'portfolio_values': portfolio_values,
            'investment_dates': investment_dates,
            'data': data
        }

        return self.results

    def export_results(self, filename="backtest_results.xlsx"):
        """导出结果到Excel"""
        if not self.results:
            print("没有结果可导出，请先运行回测")
            return

        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # 1. 汇总结果
            summary_data = {
                '指标': [
                    '股票代码',
                    '回测期间',
                    '定投频率(天)',
                    '单次投资金额(元)',
                    '投资次数',
                    '累计投入现金(元)',
                    '累计卖出获得现金(元)',
                    '净投入现金(元)',
                    '剩余持有份额',
                    '剩余持仓成本(元)',
                    '平均持仓成本(元)',
                    '最终价格(元)',
                    '剩余持仓市值(元)',
                    '未实现盈亏(元)',
                    '已实现利润(元)',
                    '总资产(元)',
                    '总收益(元)',
                    '总收益率(%)',
                    '最大回撤(%)',
                    '是否启用止盈',
                    '止盈阈值(%)',
                    '止盈比例(%)'
                ],
                '数值': [
                    self.config['symbol'],
                    f"{self.config['start_date']} 至 {self.config['end_date']}",
                    self.config['investment_frequency'],
                    self.config['investment_amount'],
                    self.results['investment_count'],
                    f"{self.results['cash_in']:.2f}",
                    f"{self.results['cash_out']:.2f}",
                    f"{self.results['net_cash_invested']:.2f}",
                    f"{self.results['remaining_shares']:.2f}",
                    f"{self.results['remaining_cost']:.2f}",
                    f"{self.results['average_cost']:.4f}",
                    f"{self.results['final_price']:.4f}",
                    f"{self.results['final_market_value']:.2f}",
                    f"{self.results['unrealized_pnl']:.2f}",
                    f"{self.results['realized_profit']:.2f}",
                    f"{self.results['total_assets']:.2f}",
                    f"{self.results['total_return']:.2f}",
                    f"{self.results['total_return_rate']:.2f}",
                    f"{self.results['max_drawdown']:.2f}",
                    '是' if self.config['enable_profit_taking'] else '否',
                    f"{self.config['profit_threshold'] * 100:.1f}",
                    f"{self.config['sell_ratio'] * 100:.1f}"
                ]
            }

            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='回测汇总', index=False)

            # 2. 详细交易记录
            transactions_data = []
            for t in self.results['transactions']:
                if t['type'] == 'buy':
                    transactions_data.append({
                        '日期': t['date'].strftime('%Y-%m-%d'),
                        '交易类型': '买入',
                        '价格': f"{t['price']:.4f}",
                        '份额': f"{t['shares']:.2f}",
                        '金额': f"{t['amount']:.2f}",
                        '累计份额': f"{t['total_shares']:.2f}",
                        '累计成本': f"{t['total_cost']:.2f}",
                        '平均成本': f"{t['average_cost']:.4f}",
                        '当前市值': f"{t['current_market_value']:.2f}",
                        '未实现盈亏': f"{t['unrealized_pnl']:.2f}",
                        '未实现收益率%': f"{t['unrealized_return_rate']:.2f}",
                        '已实现利润': f"{t['realized_profit']:.2f}",
                        '累计投入': f"{t['cash_in']:.2f}",
                        '累计卖出': f"{t['cash_out']:.2f}",
                        '卖出成本': '',
                        '单次获利': ''
                    })
                else:  # sell
                    transactions_data.append({
                        '日期': t['date'].strftime('%Y-%m-%d'),
                        '交易类型': '卖出',
                        '价格': f"{t['price']:.4f}",
                        '份额': f"{t['shares']:.2f}",
                        '金额': f"{t['amount']:.2f}",
                        '累计份额': f"{t['total_shares']:.2f}",
                        '累计成本': f"{t['total_cost']:.2f}",
                        '平均成本': f"{t['average_cost']:.4f}",
                        '当前市值': '',
                        '未实现盈亏': '',
                        '未实现收益率%': '',
                        '已实现利润': f"{t['realized_profit']:.2f}",
                        '累计投入': f"{t['cash_in']:.2f}",
                        '累计卖出': f"{t['cash_out']:.2f}",
                        '卖出成本': f"{t['cost_of_sold_shares']:.2f}",
                        '单次获利': f"{t['profit_from_sale']:.2f}"
                    })

            transactions_df = pd.DataFrame(transactions_data)
            transactions_df.to_excel(writer, sheet_name='交易明细', index=False)

            # 3. 组合价值变化
            portfolio_data = []
            cumulative_investment = 0
            for i, (date, value) in enumerate(zip(self.results['investment_dates'], self.results['portfolio_values'])):
                # 简化处理：假设每次投资固定金额
                if i < self.results['investment_count']:
                    cumulative_investment += self.config['investment_amount']

                portfolio_data.append({
                    '日期': date.strftime('%Y-%m-%d'),
                    '组合总价值': f"{value:.2f}",
                    '累计投入': f"{cumulative_investment:.2f}",
                    '收益': f"{value - cumulative_investment:.2f}",
                    '收益率%': f"{((value / cumulative_investment - 1) * 100) if cumulative_investment > 0 else 0:.2f}"
                })

            portfolio_df = pd.DataFrame(portfolio_data)
            portfolio_df.to_excel(writer, sheet_name='组合价值变化', index=False)

        print(f"结果已导出到: {filename}")

    def plot_results(self):
        """绘制结果图表"""
        if not self.results:
            print("没有结果可绘制")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # 1. 价格走势
        data = self.results['data']
        ax1.plot(data.index, data['Close'], label=f'{self.config["symbol"]} 价格走势', linewidth=2, color='blue')
        ax1.axhline(y=self.results['average_cost'], color='red', linestyle='--', alpha=0.7,
                    label=f'平均成本 {self.results["average_cost"]:.4f}')
        ax1.set_title('ETF价格走势 vs 平均成本')
        ax1.set_ylabel('价格(元)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 组合价值变化
        dates = self.results['investment_dates']
        values = self.results['portfolio_values']
        # 构建累计投入曲线
        cumulative_investments = []
        investment_amount = self.config['investment_amount']
        for i in range(len(values)):
            cumulative_investments.append(investment_amount * (i + 1))

        ax2.plot(dates, values, label='组合总价值', linewidth=2, color='green')
        ax2.plot(dates, cumulative_investments, label='累计投入', linewidth=2, color='red', linestyle='--')
        ax2.fill_between(dates, values, cumulative_investments, alpha=0.3,
                         color='green' if values[-1] > cumulative_investments[-1] else 'red')
        ax2.set_title('组合价值 vs 累计投入')
        ax2.set_ylabel('金额(元)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. 收益分析
        metrics = ['总收益率(%)', '最大回撤(%)', '已实现利润率(%)']
        values_metrics = [
            self.results['total_return_rate'],
            self.results['max_drawdown'],
            (self.results['realized_profit'] / self.results['cash_in'] * 100) if self.results['cash_in'] > 0 else 0
        ]
        colors = ['green' if v >= 0 else 'red' for v in values_metrics]

        bars = ax3.bar(metrics, values_metrics, color=colors, alpha=0.7)
        ax3.set_title('关键指标')
        ax3.set_ylabel('百分比(%)')
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)

        # 在柱状图上显示数值
        for bar, val in zip(bars, values_metrics):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2., height + (0.5 if height >= 0 else -1),
                     f'{val:.2f}%', ha='center', va='bottom' if height >= 0 else 'top')

        # 4. 资产构成
        labels = []
        sizes = []
        colors_pie = []

        if self.results['final_market_value'] > 0:
            labels.append(f'剩余持仓\n{self.results["final_market_value"]:.0f}元')
            sizes.append(self.results['final_market_value'])
            colors_pie.append('lightskyblue')

        if self.results['realized_profit'] > 0:
            labels.append(f'已实现利润\n{self.results["realized_profit"]:.0f}元')
            sizes.append(self.results['realized_profit'])
            colors_pie.append('lightgreen')
        elif self.results['realized_profit'] < 0:
            labels.append(f'已实现亏损\n{self.results["realized_profit"]:.0f}元')
            sizes.append(-self.results['realized_profit'])  # 取绝对值用于绘图
            colors_pie.append('lightcoral')

        if len(sizes) > 0:
            ax4.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
            ax4.set_title(f'总资产构成\n总计: {self.results["total_assets"]:.0f}元')

        plt.tight_layout()
        plt.show()

    def run(self, config_file="backtest_config.xlsx"):
        """运行完整的回测流程"""
        print("=" * 60)
        print("ETF定投回测系统")
        print("=" * 60)

        # 加载配置
        if not self.load_config(config_file):
            return

        print("\n开始回测...")
        print("-" * 40)

        # 执行回测
        results = self.backtest()
        if results is None:
            return

        # 显示结果
        print("\n" + "=" * 40)
        print("回测结果汇总:")
        print("=" * 40)
        print(f"股票代码: {self.config['symbol']}")
        print(f"回测期间: {self.config['start_date']} 至 {self.config['end_date']}")
        print(f"投资次数: {results['investment_count']} 次")
        print(f"累计投入现金: {results['cash_in']:.2f} 元")
        print(f"累计卖出获得现金: {results['cash_out']:.2f} 元")
        print(f"净投入现金: {results['net_cash_invested']:.2f} 元")
        print(f"剩余持仓: {results['remaining_shares']:.2f} 份")
        print(f"平均成本: {results['average_cost']:.4f} 元")
        print(f"最终价格: {results['final_price']:.4f} 元")
        print(f"剩余持仓市值: {results['final_market_value']:.2f} 元")
        print(f"未实现盈亏: {results['unrealized_pnl']:.2f} 元")
        print(f"已实现利润: {results['realized_profit']:.2f} 元")
        print(f"总资产: {results['total_assets']:.2f} 元")
        print(f"总收益: {results['total_return']:.2f} 元")
        print(f"总收益率: {results['total_return_rate']:.2f}%")
        print(f"最大回撤: {results['max_drawdown']:.2f}%")

        # 导出结果
        self.export_results()

        # 绘制图表
        self.plot_results()


# 主程序
if __name__ == "__main__":
    backtester = ETFBacktester()
    backtester.run()