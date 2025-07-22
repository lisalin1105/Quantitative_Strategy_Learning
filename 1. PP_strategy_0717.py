import akshare as ak
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings

# 忽略字体警告
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


# 尝试查找系统中可用的中文字体
def find_chinese_fonts():
    """查找系统中可用的中文字体"""
    chinese_fonts = []
    for font in fm.findSystemFonts():
        try:
            font_prop = fm.FontProperties(fname=font)
            if font_prop.get_name() in ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC',
                                        'Microsoft YaHei', 'SimSun', 'KaiTi']:
                chinese_fonts.append(font)
        except:
            continue
    return chinese_fonts


# 设置字体
chinese_fonts = find_chinese_fonts()
if chinese_fonts:
    plt.rcParams["font.family"] = fm.FontProperties(fname=chinese_fonts[0]).get_name()
else:
    print("警告: 未找到中文字体，图表中的中文可能无法正确显示")
    plt.rcParams["font.family"] = ["sans-serif"]  # 使用默认字体

plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


def get_hs300_data(start_date, end_date):
    """获取沪深300指数历史数据，自动处理请求频次"""
    print("正在获取沪深300指数数据...")
    try:
        # 尝试获取数据
        hs300_data = ak.stock_zh_index_daily(symbol="sh000300")

        # 检查数据是否有效
        if hs300_data.empty:
            raise Exception("获取的数据为空")

        # 转换日期格式并筛选时间范围
        hs300_data['date'] = pd.to_datetime(hs300_data['date'])
        filtered_data = hs300_data[(hs300_data['date'] >= start_date) &
                                   (hs300_data['date'] <= end_date)]

        # 检查筛选后的数据是否有效
        if filtered_data.empty:
            raise Exception(f"在 {start_date} 至 {end_date} 范围内没有数据")

        # 按日期排序
        filtered_data = filtered_data.sort_values('date')
        print(f"成功获取 {len(filtered_data)} 天的沪深300指数数据")

        # 添加请求间隔，避免API限制
        time.sleep(1)
        return filtered_data

    except Exception as e:
        print(f"获取数据时出错: {e}")
        return None


def calculate_pivot_points(df):
    """计算每日枢轴点、支撑位和阻力位"""
    print("正在计算枢轴点和支撑阻力位...")

    # 确保数据按日期升序排列
    df = df.sort_values('date')

    # 初始化结果列
    df['PP'] = 0.0  # 枢轴点
    df['S1'] = 0.0  # 第一支撑位
    df['S2'] = 0.0  # 第二支撑位
    df['S3'] = 0.0  # 第三支撑位
    df['R1'] = 0.0  # 第一阻力位
    df['R2'] = 0.0  # 第二阻力位
    df['R3'] = 0.0  # 第三阻力位

    # 标准枢轴点计算方法（基于前一天数据）
    for i in range(1, len(df)):
        high = df.iloc[i - 1]['high']
        low = df.iloc[i - 1]['low']
        close = df.iloc[i - 1]['close']

        # 检查前一天数据是否有效
        if np.isnan(high) or np.isnan(low) or np.isnan(close):
            continue

        pp = (high + low + close) / 3
        s1 = 2 * pp - high
        s2 = pp - (high - low)
        s3 = low - 2 * (high - pp)
        r1 = 2 * pp - low
        r2 = pp + (high - low)
        r3 = high + 2 * (pp - low)

        # 更新当前行的枢轴点和支撑阻力位
        df.loc[df.index[i], 'PP'] = pp
        df.loc[df.index[i], 'S1'] = s1
        df.loc[df.index[i], 'S2'] = s2
        df.loc[df.index[i], 'S3'] = s3
        df.loc[df.index[i], 'R1'] = r1
        df.loc[df.index[i], 'R2'] = r2
        df.loc[df.index[i], 'R3'] = r3

    # 检查计算结果是否有效
    valid_pivot_days = len(df[df['PP'] > 0])
    print(f"成功计算 {valid_pivot_days} 天的枢轴点数据")

    if valid_pivot_days < 2:
        print("警告: 有效枢轴点数据不足，策略可能无法执行")

    return df


def backtest_strategy(df, initial_capital=10000, verbose=True):
    """回测固定金额交易策略"""
    print("正在回测交易策略...")

    # 初始化资金和持仓
    capital = initial_capital  # 可用资金
    shares = 0  # 持有的基金份额
    portfolio_value = []  # 每日组合价值
    trades = []  # 交易记录
    daily_decisions = []  # 每日决策日志

    for i in range(len(df)):
        current_date = df.iloc[i]['date']
        current_price = df.iloc[i]['close']
        pp = df.iloc[i]['PP']
        s1 = df.iloc[i]['S1']
        s2 = df.iloc[i]['S2']
        s3 = df.iloc[i]['S3']
        r1 = df.iloc[i]['R1']
        r2 = df.iloc[i]['R2']
        r3 = df.iloc[i]['R3']

        # 跳过枢轴点数据无效的日期
        if pp <= 0:
            if verbose:
                print(f"{current_date.strftime('%Y-%m-%d')}: 枢轴点数据无效，跳过")
            portfolio_value.append(capital + shares * current_price)
            daily_decisions.append({
                'date': current_date,
                'price': current_price,
                'pp': pp,
                'action': '跳过（枢轴点无效）',
                'capital': capital,
                'shares': shares,
                'portfolio_value': capital + shares * current_price
            })
            continue

        # 计算当前总资产价值
        current_value = capital + shares * current_price
        portfolio_value.append(current_value)

        # 交易策略逻辑
        trade_action = None
        trade_amount = 0
        decision_reason = ""

        # 检查卖出条件（按优先级排序）
        if shares > 0 and current_price > r3:
            # 高于第三阻力位，卖出3000元
            amount_to_sell = min(3000, shares * current_price)  # 确保不超过持仓价值
            shares_to_sell = amount_to_sell / current_price
            capital += amount_to_sell
            shares -= shares_to_sell
            trade_action = f"卖出 {shares_to_sell:.2f} 份"
            trade_amount = amount_to_sell
            decision_reason = f"价格({current_price:.2f}) > R3({r3:.2f})"
            if verbose:
                print(
                    f"{current_date.strftime('%Y-%m-%d')} | 卖出 {amount_to_sell} 元 | 价格: {current_price:.2f} | 剩余资金: {capital:.2f}元 | 持有份额: {shares:.2f}")
        elif shares > 0 and current_price > r2:
            # 高于第二阻力位，卖出2000元
            amount_to_sell = min(2000, shares * current_price)  # 确保不超过持仓价值
            shares_to_sell = amount_to_sell / current_price
            capital += amount_to_sell
            shares -= shares_to_sell
            trade_action = f"卖出 {shares_to_sell:.2f} 份"
            trade_amount = amount_to_sell
            decision_reason = f"R3({r3:.2f}) >= 价格({current_price:.2f}) > R2({r2:.2f})"
            if verbose:
                print(
                    f"{current_date.strftime('%Y-%m-%d')} | 卖出 {amount_to_sell} 元 | 价格: {current_price:.2f} | 剩余资金: {capital:.2f}元 | 持有份额: {shares:.2f}")
        elif shares > 0 and current_price > r1:
            # 高于第一阻力位，卖出1000元
            amount_to_sell = min(1000, shares * current_price)  # 确保不超过持仓价值
            shares_to_sell = amount_to_sell / current_price
            capital += amount_to_sell
            shares -= shares_to_sell
            trade_action = f"卖出 {shares_to_sell:.2f} 份"
            trade_amount = amount_to_sell
            decision_reason = f"R2({r2:.2f}) >= 价格({current_price:.2f}) > R1({r1:.2f})"
            if verbose:
                print(
                    f"{current_date.strftime('%Y-%m-%d')} | 卖出 {amount_to_sell} 元 | 价格: {current_price:.2f} | 剩余资金: {capital:.2f}元 | 持有份额: {shares:.2f}")

        # 检查买入条件（按优先级排序，确保不超过可用资金）
        elif current_price < s3 and capital >= 3000:
            # 低于第三支撑位，买入3000元
            amount_to_buy = 3000
            shares_to_buy = amount_to_buy / current_price
            capital -= amount_to_buy
            shares += shares_to_buy
            trade_action = f"买入 {shares_to_buy:.2f} 份"
            trade_amount = amount_to_buy
            decision_reason = f"价格({current_price:.2f}) < S3({s3:.2f})"
            if verbose:
                print(
                    f"{current_date.strftime('%Y-%m-%d')} | 买入 {amount_to_buy} 元 | 价格: {current_price:.2f} | 剩余资金: {capital:.2f}元 | 持有份额: {shares:.2f}")
        elif current_price < s2 and capital >= 2000:
            # 低于第二支撑位，买入2000元
            amount_to_buy = 2000
            shares_to_buy = amount_to_buy / current_price
            capital -= amount_to_buy
            shares += shares_to_buy
            trade_action = f"买入 {shares_to_buy:.2f} 份"
            trade_amount = amount_to_buy
            decision_reason = f"S3({s3:.2f}) <= 价格({current_price:.2f}) < S2({s2:.2f})"
            if verbose:
                print(
                    f"{current_date.strftime('%Y-%m-%d')} | 买入 {amount_to_buy} 元 | 价格: {current_price:.2f} | 剩余资金: {capital:.2f}元 | 持有份额: {shares:.2f}")
        elif current_price < s1 and capital >= 1000:
            # 低于第一支撑位，买入1000元
            amount_to_buy = 1000
            shares_to_buy = amount_to_buy / current_price
            capital -= amount_to_buy
            shares += shares_to_buy
            trade_action = f"买入 {shares_to_buy:.2f} 份"
            trade_amount = amount_to_buy
            decision_reason = f"S2({s2:.2f}) <= 价格({current_price:.2f}) < S1({s1:.2f})"
            if verbose:
                print(
                    f"{current_date.strftime('%Y-%m-%d')} | 买入 {amount_to_buy} 元 | 价格: {current_price:.2f} | 剩余资金: {capital:.2f}元 | 持有份额: {shares:.2f}")

        # 记录每日决策
        daily_decisions.append({
            'date': current_date,
            'price': current_price,
            'pp': pp,
            'action': trade_action if trade_action else "无交易",
            'reason': decision_reason,
            'capital': capital,
            'shares': shares,
            'portfolio_value': current_value
        })

        # 记录交易
        if trade_action:
            trades.append({
                'date': current_date,
                'action': trade_action,
                'price': current_price,
                'amount': trade_amount,
                'capital': capital,
                'shares': shares,
                'portfolio_value': current_value
            })

    # 打印交易频率统计
    if daily_decisions:
        trade_days = sum(1 for d in daily_decisions if d['action'] != "无交易")
        total_days = len(daily_decisions)
        print(f"交易天数: {trade_days}/{total_days} ({trade_days / total_days * 100:.2f}%)")

    # 最终资产价值（考虑剩余持仓）
    final_value = capital + shares * df.iloc[-1]['close']

    return {
        'initial_capital': initial_capital,
        'final_value': final_value,
        'return_rate': (final_value - initial_capital) / initial_capital * 100,
        'trades': trades,
        'portfolio_value': portfolio_value,
        'dates': df['date'].tolist(),
        'daily_decisions': daily_decisions  # 返回每日决策日志
    }


def plot_results(results, data):
    """可视化回测结果"""
    print("正在生成可视化图表...")

    font_prop = None
    if chinese_fonts:
        font_prop = fm.FontProperties(fname=chinese_fonts[0])

    plt.figure(figsize=(16, 10))

    # 资产价值变化图
    plt.subplot(2, 1, 1)
    plt.plot(results['dates'], results['portfolio_value'], label='资产价值', color='blue', linewidth=2)
    plt.title('策略回测结果：资产价值变化', fontproperties=font_prop, fontsize=14)
    plt.xlabel('日期', fontproperties=font_prop, fontsize=12)
    plt.ylabel('资产价值（元）', fontproperties=font_prop, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(prop=font_prop, fontsize=12)

    # 标记交易点
    for trade in results['trades']:
        date = trade['date']
        value = trade['portfolio_value']
        if '买入' in trade['action']:
            plt.scatter(date, value, color='green', marker='^', s=120,
                        label='买入' if '买入' not in [t.get_label() for t in
                                                       plt.gca().get_legend().get_texts()] else "")
        else:
            plt.scatter(date, value, color='red', marker='v', s=120,
                        label='卖出' if '卖出' not in [t.get_label() for t in
                                                       plt.gca().get_legend().get_texts()] else "")

    # 价格与支撑阻力位图
    plt.subplot(2, 1, 2)
    plt.plot(data['date'], data['close'], label='收盘价', color='black', linewidth=2)
    plt.plot(data['date'], data['PP'], label='枢轴点', color='gray', linestyle='--', linewidth=1.5)
    plt.plot(data['date'], data['S1'], label='第一支撑位', color='green', linestyle='--', linewidth=1.5)
    plt.plot(data['date'], data['S2'], label='第二支撑位', color='darkgreen', linestyle='--', linewidth=1.5)
    plt.plot(data['date'], data['S3'], label='第三支撑位', color='lightgreen', linestyle='--', linewidth=1.5)
    plt.plot(data['date'], data['R1'], label='第一阻力位', color='red', linestyle='--', linewidth=1.5)
    plt.plot(data['date'], data['R2'], label='第二阻力位', color='darkred', linestyle='--', linewidth=1.5)
    plt.plot(data['date'], data['R3'], label='第三阻力位', color='pink', linestyle='--', linewidth=1.5)

    plt.title('沪深300指数价格与支撑阻力位', fontproperties=font_prop, fontsize=14)
    plt.xlabel('日期', fontproperties=font_prop, fontsize=12)
    plt.ylabel('指数点位', fontproperties=font_prop, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # 调整图例位置和大小
    plt.legend(prop=font_prop, fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.savefig('backtest_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def print_trade_summary(results):
    """打印交易摘要"""
    print("\n===== 交易策略回测总结 =====")
    print(f"初始资金: {results['initial_capital']} 元")
    print(f"最终资产: {results['final_value']:.2f} 元")
    print(f"总收益率: {results['return_rate']:.2f}%")
    print(f"交易次数: {len(results['trades'])} 次")

    if results['trades']:
        # 分析交易结果
        buy_trades = [t for t in results['trades'] if '买入' in t['action']]
        sell_trades = [t for t in results['trades'] if '卖出' in t['action']]

        print(f"\n买入次数: {len(buy_trades)} | 卖出次数: {len(sell_trades)}")

        if sell_trades:
            total_profit = sum(t['amount'] for t in sell_trades) - sum(t['amount'] for t in buy_trades)
            avg_buy_price = sum(t['price'] for t in buy_trades) / len(buy_trades) if buy_trades else 0
            avg_sell_price = sum(t['price'] for t in sell_trades) / len(sell_trades) if sell_trades else 0

            print(f"总利润: {total_profit:.2f} 元")
            print(f"平均买入价格: {avg_buy_price:.2f}")
            print(f"平均卖出价格: {avg_sell_price:.2f}")
            print(
                f"买卖价差: {avg_sell_price - avg_buy_price:.2f} ({((avg_sell_price / avg_buy_price) - 1) * 100:.2f}%)")

        print("\n最近10笔交易:")
        for trade in results['trades'][-10:]:
            print(
                f"{trade['date'].strftime('%Y-%m-%d')} | {trade['action']} | 价格: {trade['price']:.2f} | 资产: {trade['portfolio_value']:.2f}")


def main():
    """主函数：运行回测策略"""
    print("=" * 50)
    print("沪深300指数枢轴点交易策略回测系统")
    print("=" * 50)

    # 设置回测时间范围
    start_date = '2025-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')

    print(f"回测时间范围: {start_date} 至 {end_date}")

    # 获取数据
    data = get_hs300_data(start_date, end_date)
    if data is None:
        print("数据获取失败，无法进行回测")
        return

    # 计算枢轴点和支撑阻力位
    data_with_pivot = calculate_pivot_points(data)

    # 回测策略
    results = backtest_strategy(data_with_pivot, initial_capital=10000, verbose=True)

    # 打印交易摘要
    print_trade_summary(results)

    # 可视化结果
    plot_results(results, data_with_pivot)


if __name__ == "__main__":
    main()