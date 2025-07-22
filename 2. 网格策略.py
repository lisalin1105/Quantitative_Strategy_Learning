import backtrader as bt
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class GridTradingStrategy(bt.Strategy):
    """
    网格交易策略
    在预设的价格区间内，设置多个买入和卖出价格网格
    当价格触及网格线时自动执行相应的买入或卖出操作
    """

    params = (
        ('grid_lines', 10),  # 网格线数量
        ('grid_range', 0.20),  # 网格范围（相对于基准价格的百分比）
        ('position_size', 0.1),  # 每次交易的仓位大小（相对于总资金）
        ('base_price', None),  # 基准价格（如果不设置，使用初始价格）
        ('max_positions', 5),  # 最大持仓数量
        ('rebalance_period', 20),  # 重新平衡网格的周期（交易日）
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.datahigh = self.datas[0].high
        self.datalow = self.datas[0].low

        # 初始化网格相关变量
        self.grid_levels = []
        self.grid_orders = {}
        self.positions_count = 0
        self.last_rebalance = 0
        self.initial_setup_done = False

        # 存储交易记录
        self.trades_log = []

    def start(self):
        """策略开始时的初始化"""
        print(f"策略开始运行，初始资金: ${self.broker.getvalue():.2f}")

    def setup_grid(self, base_price=None):
        """设置网格线"""
        if base_price is None:
            base_price = self.dataclose[0]

        if self.params.base_price:
            base_price = self.params.base_price

        # 计算网格范围
        grid_range = base_price * self.params.grid_range
        grid_step = (2 * grid_range) / self.params.grid_lines

        # 生成网格价格线
        self.grid_levels = []
        for i in range(self.params.grid_lines + 1):
            price = base_price - grid_range + (i * grid_step)
            self.grid_levels.append(price)

        self.grid_levels.sort()
        print(f"网格设置完成，基准价格: ${base_price:.2f}")
        print(f"网格范围: ${min(self.grid_levels):.2f} - ${max(self.grid_levels):.2f}")

    def next(self):
        """每个交易日的主要逻辑"""
        current_price = self.dataclose[0]

        # 首次运行时设置网格
        if not self.initial_setup_done:
            self.setup_grid()
            self.initial_setup_done = True
            return

        # 定期重新平衡网格
        if len(self.data) - self.last_rebalance > self.params.rebalance_period:
            self.rebalance_grid()
            self.last_rebalance = len(self.data)

        # 执行网格交易逻辑
        self.execute_grid_orders(current_price)

    def execute_grid_orders(self, current_price):
        """执行网格交易订单"""
        available_cash = self.broker.getcash()
        position_value = min(self.broker.getvalue() * self.params.position_size, available_cash * 0.8)
        shares_to_trade = max(1, int(position_value / current_price))  # 至少交易1股

        # 寻找最近的网格线
        closest_grid_below = None
        closest_grid_above = None

        for level in self.grid_levels:
            if level < current_price:
                closest_grid_below = level
            elif level > current_price and closest_grid_above is None:
                closest_grid_above = level
                break

        # 买入逻辑：价格跌至网格支撑位且持仓未满
        if (closest_grid_below and
                current_price <= closest_grid_below * 1.002 and  # 允许小幅偏差
                self.positions_count < self.params.max_positions and
                closest_grid_below not in self.grid_orders and
                available_cash >= shares_to_trade * current_price):  # 确保有足够现金

            order = self.buy(size=shares_to_trade)
            if order:
                self.grid_orders[closest_grid_below] = order
                self.positions_count += 1

                self.trades_log.append({
                    'date': self.data.datetime.date(0),
                    'action': 'BUY',
                    'price': current_price,
                    'size': shares_to_trade,
                    'grid_level': closest_grid_below
                })

                print(
                    f"{self.data.datetime.date(0)}: 买入 {shares_to_trade} 股，价格 ${current_price:.2f}，网格线 ${closest_grid_below:.2f}")

        # 卖出逻辑：价格涨至网格阻力位且有持仓
        if (closest_grid_above and
                current_price >= closest_grid_above * 0.998 and  # 允许小幅偏差
                self.position.size > 0 and
                closest_grid_above not in self.grid_orders):

            sell_size = min(shares_to_trade, self.position.size)
            order = self.sell(size=sell_size)
            if order:
                self.grid_orders[closest_grid_above] = order

                if sell_size >= self.position.size:
                    self.positions_count = max(0, self.positions_count - 1)

                self.trades_log.append({
                    'date': self.data.datetime.date(0),
                    'action': 'SELL',
                    'price': current_price,
                    'size': sell_size,
                    'grid_level': closest_grid_above
                })

                print(
                    f"{self.data.datetime.date(0)}: 卖出 {sell_size} 股，价格 ${current_price:.2f}，网格线 ${closest_grid_above:.2f}")

    def rebalance_grid(self):
        """重新平衡网格"""
        print(f"重新平衡网格，当前价格: ${self.dataclose[0]:.2f}")
        self.setup_grid(self.dataclose[0])
        self.grid_orders.clear()  # 清除旧的订单记录

    def notify_order(self, order):
        """订单状态通知"""
        if order.status in [order.Completed]:
            # 从网格订单记录中移除已完成的订单
            for level, grid_order in list(self.grid_orders.items()):
                if grid_order == order:
                    del self.grid_orders[level]
                    break

    def stop(self):
        """策略结束时的处理"""
        final_value = self.broker.getvalue()
        print(f"策略结束，最终资金: ${final_value:.2f}")
        print(f"总交易次数: {len(self.trades_log)}")


class GridTradingAnalyzer(bt.Analyzer):
    """网格交易分析器"""

    def __init__(self):
        self.trades = []
        self.daily_values = []

    def notify_trade(self, trade):
        if trade.isclosed:
            # 防止除零错误
            profit_pct = 0
            if trade.value != 0:
                profit_pct = trade.pnlcomm / abs(trade.value) * 100

            self.trades.append({
                'profit': trade.pnl,
                'profit_pct': profit_pct,
                'bars': trade.barlen,
                'value': trade.value
            })

    def next(self):
        self.daily_values.append(self.strategy.broker.getvalue())

    def get_analysis(self):
        if not self.trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_profit': 0,
                'avg_profit': 0,
                'avg_profit_pct': 0,
                'max_profit': 0,
                'max_loss': 0,
                'daily_values': self.daily_values
            }

        profits = [trade['profit'] for trade in self.trades]
        profit_pcts = [trade['profit_pct'] for trade in self.trades]

        winning_trades = [p for p in profits if p > 0]
        losing_trades = [p for p in profits if p < 0]

        return {
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(self.trades) * 100 if self.trades else 0,
            'total_profit': sum(profits),
            'avg_profit': np.mean(profits) if profits else 0,
            'avg_profit_pct': np.mean(profit_pcts) if profit_pcts else 0,
            'max_profit': max(profits) if profits else 0,
            'max_loss': min(profits) if profits else 0,
            'daily_values': self.daily_values
        }


def run_grid_trading_backtest(symbol='AAPL', start_date='2020-01-01', end_date='2024-01-01',
                              initial_cash=10000, grid_lines=10, grid_range=0.15):
    """
    运行网格交易回测

    参数:
    symbol: 股票代码
    start_date: 开始日期
    end_date: 结束日期
    initial_cash: 初始资金
    grid_lines: 网格线数量
    grid_range: 网格范围
    """

    print(f"开始回测 {symbol} 网格交易策略")
    print(f"回测期间: {start_date} 至 {end_date}")
    print(f"初始资金: ${initial_cash}")
    print(f"网格设置: {grid_lines} 条网格线，范围 ±{grid_range * 100}%")
    print("-" * 50)

    # 下载股票数据
    try:
        # 修复yfinance的FutureWarning并确保数据格式正确
        print(f"正在下载 {symbol} 的数据...")
        data = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)

        if data.empty:
            print(f"无法获取 {symbol} 的数据，请检查股票代码和日期范围")
            return None

        print(f"原始数据列名: {list(data.columns)}")
        print(f"数据形状: {data.shape}")

        # 重置索引，将Date从索引变为列
        data = data.reset_index()
        print(f"重置索引后列名: {list(data.columns)}")

        # 如果是多层列名(MultiIndex)，需要展平
        if isinstance(data.columns, pd.MultiIndex):
            # 正确的展平多层列名逻辑
            new_columns = []
            for col in data.columns:
                if col[1] == '':  # Date列的情况
                    new_columns.append(col[0])
                else:  # 其他数据列，使用第一个元素（数据类型名称）
                    new_columns.append(col[0])
            data.columns = new_columns
            print(f"展平后列名: {list(data.columns)}")

        # 直接使用标准的列名映射
        column_mapping = {
            'Date': 'datetime',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Adj Close': 'close'  # 如果有调整后收盘价
        }

        # 应用重命名
        data = data.rename(columns=column_mapping)
        print(f"重命名后列名: {list(data.columns)}")

        # 确保Date列存在并检查必要的列
        if 'datetime' not in data.columns:
            print(f"错误: 找不到日期列。现有列: {list(data.columns)}")
            return None

        # 检查必要的列
        required_columns = ['datetime', 'open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            print(f"错误: 缺少必要的列: {missing_columns}")
            print(f"可用列: {list(data.columns)}")
            return None

        # 如果没有volume列，创建默认值
        if 'volume' not in data.columns:
            data['volume'] = 1000000  # 默认成交量
            print("添加了默认的volume列")

        # 转换datetime列
        data['datetime'] = pd.to_datetime(data['datetime'])

        # 设置datetime为索引
        data.set_index('datetime', inplace=True)

        # 确保数据类型正确
        for col in ['open', 'high', 'low', 'close', 'volume']:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        # 删除任何包含NaN的行
        data = data.dropna()

        if data.empty:
            print("错误: 处理后数据为空")
            return None

        print(f"成功加载数据，共 {len(data)} 个交易日")
        print(f"数据范围: {data.index[0].strftime('%Y-%m-%d')} 至 {data.index[-1].strftime('%Y-%m-%d')}")
        print(f"最终数据列: {list(data.columns)}")

    except Exception as e:
        print(f"数据下载失败: {e}")
        import traceback
        traceback.print_exc()
        return None

    # 创建Cerebro引擎
    cerebro = bt.Cerebro()

    # 添加数据
    data_feed = bt.feeds.PandasData(
        dataname=data,
        datetime=None,  # 使用索引作为datetime
        open='open',
        high='high',
        low='low',
        close='close',
        volume='volume',
        openinterest=None
    )
    cerebro.adddata(data_feed)

    # 添加策略
    cerebro.addstrategy(GridTradingStrategy,
                        grid_lines=grid_lines,
                        grid_range=grid_range)

    # 添加分析器
    cerebro.addanalyzer(GridTradingAnalyzer, _name='grid_analyzer')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

    # 设置初始资金和佣金
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=0.001)  # 0.1% 佣金

    # 运行回测
    print("正在运行回测...")
    results = cerebro.run()

    # 获取结果
    strat = results[0]

    # 打印分析结果
    print("\n" + "=" * 50)
    print("回测结果分析")
    print("=" * 50)

    final_value = cerebro.broker.getvalue()
    total_return = (final_value - initial_cash) / initial_cash * 100

    print(f"初始资金: ${initial_cash:,.2f}")
    print(f"最终资金: ${final_value:,.2f}")
    print(f"总收益率: {total_return:.2f}%")

    # 网格交易分析
    grid_analysis = strat.analyzers.grid_analyzer.get_analysis()
    if grid_analysis:
        print(f"\n交易统计:")
        print(f"总交易次数: {grid_analysis['total_trades']}")
        print(f"盈利交易: {grid_analysis['winning_trades']}")
        print(f"亏损交易: {grid_analysis['losing_trades']}")
        print(f"胜率: {grid_analysis['win_rate']:.1f}%")
        print(f"平均收益: ${grid_analysis['avg_profit']:.2f}")
        print(f"平均收益率: {grid_analysis['avg_profit_pct']:.2f}%")
        print(f"最大盈利: ${grid_analysis['max_profit']:.2f}")
        print(f"最大亏损: ${grid_analysis['max_loss']:.2f}")

    # 其他分析指标
    if hasattr(strat.analyzers.sharpe, 'get_analysis'):
        sharpe_ratio = strat.analyzers.sharpe.get_analysis().get('sharperatio', 'N/A')
        print(f"夏普比率: {sharpe_ratio}")

    drawdown_analysis = strat.analyzers.drawdown.get_analysis()
    print(f"最大回撤: {drawdown_analysis['max']['drawdown']:.2f}%")

    return {
        'final_value': final_value,
        'total_return': total_return,
        'grid_analysis': grid_analysis,
        'drawdown': drawdown_analysis,
        'cerebro': cerebro
    }


# 示例用法
if __name__ == '__main__':
    # 回测苹果股票的网格交易策略
    result = run_grid_trading_backtest(
        symbol='AAPL',  # 苹果股票
        start_date='2024-01-01',  # 开始日期
        end_date='2025-07-22',  # 结束日期
        initial_cash=10000,  # 初始资金1万美元
        grid_lines=10,  # 10条网格线
        grid_range=0.15  # 网格范围±15%
    )

    # 也可以尝试其他股票和参数
    print("\n" + "=" * 50)
    print("尝试不同参数的回测")
    print("=" * 50)

    # 回测特斯拉，使用更密集的网格
    result2 = run_grid_trading_backtest(
        symbol='TSLA',  # 特斯拉股票
        start_date='2024-01-01',  # 开始日期
        end_date='2025-07-22',  # 结束日期
        initial_cash=10000,
        grid_lines=15,  # 15条网格线
        grid_range=0.25  # 网格范围±25%（特斯拉波动较大）
    )