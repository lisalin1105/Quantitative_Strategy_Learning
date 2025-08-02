import yfinance as yf
from sqlalchemy import create_engine
import pandas as pd

# 设置显示选项，显示所有列
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

# 设置 SQLite 数据库连接
db_path = '/Users/a11111/Desktop/Quantitative_Strategy_Learning/qqq_data/yfinance test.sql'
engine = create_engine(f'sqlite:///{db_path}')

# 获取苹果公司的股票数据
ticker = yf.Ticker("AAPL")

# 获取最近的股票历史数据
historical_data = ticker.history(period="5d")

# 确保历史数据加载成功
print("历史数据：")
print(historical_data.to_csv(sep='\t', na_rep='nan'))

# 将历史数据存储到 SQLite 数据库的 stocks 表中
historical_data.to_sql('stocks', con=engine, if_exists='replace', index=True)

# 获取公司的基本信息
info = ticker.info

# 只选择简单字段
simple_info = {key: value for key, value in info.items() if isinstance(value, (str, int, float))}
info_df = pd.DataFrame([simple_info])

# 存储到 SQLite 数据库的 info 表中
info_df.to_sql('info', con=engine, if_exists='replace', index=False)

# 从数据库中查询存储的数据
stored_historical_data = pd.read_sql('SELECT * FROM stocks', con=engine)
stored_info = pd.read_sql('SELECT * FROM info', con=engine)

# 打印存储的数据
print("存储的历史数据：")
print(stored_historical_data.to_csv(sep='\t', na_rep='nan'))
print("\n存储的公司基本信息：")
print(stored_info.to_csv(sep='\t', na_rep='nan'))

# 将数据保存到 CSV 文件
historical_data.to_csv('historical_data.csv', sep='\t', na_rep='nan')
stored_historical_data.to_csv('stored_historical_data.csv', sep='\t', na_rep='nan')
stored_info.to_csv('stored_info.csv', sep='\t', na_rep='nan')

print("数据已保存到 CSV 文件")