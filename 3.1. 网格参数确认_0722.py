import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import json
import os
from typing import Dict, List, Tuple, Optional


class GridParameterGenerator:
    """
    动态网格参数生成器
    根据股票历史数据自动计算最优网格交易参数，并保存配置
    """

    def __init__(self, config_path: str = "grid_configs.json"):
        """初始化参数生成器"""
        self.config_path = config_path
        self.configs = self._load_existing_configs()

    def _load_existing_configs(self) -> Dict:
        """安全加载已有的配置文件"""
        if not os.path.exists(self.config_path):
            print(f"配置文件 {self.config_path} 不存在，将创建新文件")
            return {}

        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"配置文件格式错误: {e}")
            print(f"将备份原文件并创建新的空配置文件")

            # 备份损坏的文件
            backup_path = f"{self.config_path}.bak"
            os.rename(self.config_path, backup_path)
            print(f"已将损坏的文件备份至 {backup_path}")

            return {}
        except Exception as e:
            print(f"加载配置文件时发生未知错误: {e}")
            return {}

    def save_config(self, symbol: str, params: Dict, performance: Optional[Dict] = None) -> None:
        """保存参数配置"""
        # 确保所有值都是JSON可序列化的
        if performance:
            performance = self._make_serializable(performance)

        config_entry = {
            "params": params,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "performance": performance or {}
        }

        self.configs[symbol] = config_entry

        # 创建目录（如果不存在）
        config_dir = os.path.dirname(self.config_path)
        if config_dir and not os.path.exists(config_dir):
            os.makedirs(config_dir)

        # 安全保存到JSON文件
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.configs, f, indent=4)
            print(f"参数配置已成功保存到 {self.config_path}")
        except Exception as e:
            print(f"保存配置文件失败: {e}")

    def _make_serializable(self, data: Dict) -> Dict:
        """将字典中的非序列化对象转换为可序列化格式"""
        result = {}
        for key, value in data.items():
            if isinstance(value, pd.Series):
                # 将Series转换为列表
                result[key] = value.tolist() if not value.empty else None
            elif isinstance(value, np.ndarray):
                # 将NumPy数组转换为列表
                result[key] = value.tolist()
            elif isinstance(value, (pd.Timestamp, datetime)):
                # 将时间戳转换为字符串
                result[key] = value.strftime("%Y-%m-%d %H:%M:%S")
            elif isinstance(value, float):
                # 处理特殊浮点数
                if np.isnan(value) or np.isinf(value):
                    result[key] = None
                else:
                    result[key] = float(value)  # 确保普通浮点数正确转换
            elif isinstance(value, dict):
                # 递归处理嵌套字典
                result[key] = self._make_serializable(value)
            else:
                # 基本类型直接使用
                result[key] = value
        return result

    @staticmethod
    def analyze_stock(symbol: str, period: str = "1y") -> Dict:
        """分析股票特性，返回关键指标"""
        print(f"正在分析 {symbol} 的历史数据...")

        try:
            # 下载历史数据（添加auto_adjust=False以消除FutureWarning）
            data = yf.download(symbol, period=period, auto_adjust=False)

            if data.empty:
                raise ValueError(f"无法获取 {symbol} 的历史数据")

            # 计算关键指标
            close_prices = data['Close']

            # 波动率（年化）
            volatility = close_prices.pct_change().std() * np.sqrt(252)

            # 趋势强度（最后30天与90天价格变化率）
            trend_strength = (close_prices.iloc[-1] / close_prices.iloc[-30] - 1) / \
                             (close_prices.iloc[-1] / close_prices.iloc[-90] - 1)

            # 价格范围
            price_range = (close_prices.min(), close_prices.max())
            avg_price = close_prices.mean()

            # 交易量特征
            volume_avg = data['Volume'].mean()
            volume_volatility = data['Volume'].pct_change().std()

            # 确保所有计算结果都是标量
            return {
                "volatility": float(volatility) if not pd.isna(volatility) else 0.2,
                "trend_strength": float(trend_strength) if not pd.isna(trend_strength) else 1.0,
                "price_range": (float(price_range[0]), float(price_range[1])) if not pd.isna(
                    price_range[0]) and not pd.isna(price_range[1]) else (0, 100),
                "avg_price": float(avg_price) if not pd.isna(avg_price) else 50,
                "volume_avg": float(volume_avg) if not pd.isna(volume_avg) else 0,
                "volume_volatility": float(volume_volatility) if not pd.isna(volume_volatility) else 0,
                "data_length": len(data)
            }

        except Exception as e:
            print(f"分析失败: {e}")
            # 返回包含默认值的字典，避免后续处理出错
            return {
                "volatility": 0.2,
                "trend_strength": 1.0,
                "price_range": (0, 100),
                "avg_price": 50,
                "volume_avg": 0,
                "volume_volatility": 0,
                "data_length": 0
            }

    @staticmethod
    def generate_parameters(stock_analysis: Dict) -> Dict:
        """根据股票分析结果生成网格参数"""
        if not stock_analysis:
            return {}

        # 基础参数映射表
        base_params = {
            "grid_lines": 10,
            "grid_range": 0.20,
            "position_size": 0.1,
            "rebalance_period": 20,
            "max_positions": 5
        }

        # 根据波动率调整
        volatility = stock_analysis["volatility"]

        if volatility < 0.15:  # 低波动
            base_params["grid_lines"] = 8
            base_params["grid_range"] = 0.15
            base_params["rebalance_period"] = 30
        elif volatility < 0.3:  # 中等波动
            base_params["grid_lines"] = 12
            base_params["grid_range"] = 0.25
        else:  # 高波动
            base_params["grid_lines"] = 16
            base_params["grid_range"] = 0.35
            base_params["rebalance_period"] = 15
            base_params["position_size"] = 0.08

        # 根据趋势强度调整
        trend_strength = stock_analysis["trend_strength"]

        if abs(trend_strength) > 1.5:  # 强趋势
            base_params["grid_range"] *= 1.2  # 扩大网格范围
            base_params["max_positions"] = 7  # 增加最大持仓

        # 根据平均价格调整每次交易的资金量
        avg_price = stock_analysis["avg_price"]

        if avg_price > 100:  # 高价股
            base_params["position_size"] *= 0.7  # 减少单笔资金
        elif avg_price < 10:  # 低价股
            base_params["position_size"] *= 1.3  # 增加单笔资金

        return base_params

    def run(self, symbol: str, period: str = "1y", save: bool = True) -> Tuple[Dict, Dict]:
        """
        运行完整的参数生成流程

        返回:
            tuple: (股票分析结果, 生成的参数)
        """
        analysis = self.analyze_stock(symbol, period)
        params = self.generate_parameters(analysis)

        if save and analysis and params:
            self.save_config(symbol, params, {
                "volatility": analysis["volatility"],
                "trend_strength": analysis["trend_strength"],
                "avg_price": analysis["avg_price"]
            })

        return analysis, params


# 使用示例
if __name__ == "__main__":
    generator = GridParameterGenerator()

    # 为NVDA生成参数
    try:
        nvda_analysis, nvda_params = generator.run("NVDA")
        print("\nNVDA 参数:")
        for key, value in nvda_params.items():
            print(f"{key}: {value}")
    except Exception as e:
        print(f"生成NVDA参数时出错: {e}")

    # 为TSLA生成参数
    try:
        tsla_analysis, tsla_params = generator.run("TSLA")
        print("\nTSLA 参数:")
        for key, value in tsla_params.items():
            print(f"{key}: {value}")
    except Exception as e:
        print(f"生成TSLA参数时出错: {e}")

    # 为AAPL生成参数（示例）
    try:
        aapl_analysis, aapl_params = generator.run("AAPL")
        print("\nAAPL 参数:")
        for key, value in aapl_params.items():
            print(f"{key}: {value}")
    except Exception as e:
        print(f"生成AAPL参数时出错: {e}")