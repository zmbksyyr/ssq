import pandas as pd
import sys
import datetime
import os
import requests
from bs4 import BeautifulSoup
import io
import logging
from contextlib import redirect_stdout, redirect_stderr
import csv  # 导入csv模块

# --- 配置 ---
# 获取脚本目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 构建CSV文件的完整路径
# 假设shuangseqiu.csv与脚本在同一目录，如果您想使用 ssq_results.csv (ssq.py 默认文件名) 请修改此处
CSV_FILE_PATH = os.path.join(SCRIPT_DIR, 'shuangseqiu.csv') # 您可以根据需要修改此处为 'ssq_results.csv'

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # 输出到控制台stdout
    ]
)
logger = logging.getLogger('ssq_data_fetcher')

# 添加进度条显示函数
def show_progress(current, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    """
    显示进度条
    @param current: 当前进度
    @param total: 总进度
    @param prefix: 前缀字符串
    @param suffix: 后缀字符串
    @param decimals: 小数位数
    @param length: 进度条长度
    @param fill: 进度条填充字符
    """
    if total <= 0:
        print(f'\r{prefix} |{fill * length}| 100.0% {suffix}', end='')
        if current >= total:
            print()
        return

    percent = ("{0:." + str(decimals) + "f}").format(100 * (current / float(total)))
    filled_length = int(length * current // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{bar}| {percent}% {suffix}', end='', file=sys.stdout, flush=True)
    if current >= total:
        print(file=sys.stdout, flush=True)


# 创建上下文管理器来暂时重定向输出，并捕获stderr
class SuppressOutput:
    """
    上下文管理器：暂时抑制标准输出，捕获标准错误输出到StringIO对象。
    退出时可以将捕获的stderr写入日志。
    """
    def __init__(self, suppress_stdout=True, capture_stderr=True):
        self.suppress_stdout = suppress_stdout
        self.capture_stderr = capture_stderr
        self.stdout_redirect = None
        self.stderr_redirect = None
        self.old_stdout = None
        self.old_stderr = None
        self.stderr_io = None

    def __enter__(self):
        if self.suppress_stdout:
            self.old_stdout = sys.stdout
            # 将stdout重定向到/dev/null或等效位置
            sys.stdout = open(os.devnull, 'w')

        if self.capture_stderr:
            self.old_stderr = sys.stderr
            self.stderr_io = io.StringIO()
            sys.stderr = self.stderr_io

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 先恢复stderr
        if self.capture_stderr and self.old_stderr:
            sys.stderr = self.old_stderr
            captured_stderr_content = self.stderr_io.getvalue()
            if captured_stderr_content.strip():  # 如果捕获的stderr不为空，则记录日志
                 logger.warning(f"Captured stderr:\n{captured_stderr_content.strip()}")

        # 恢复stdout
        if self.suppress_stdout and self.old_stdout:
            if sys.stdout and not sys.stdout.closed:  # 在关闭前检查重定向对象是否有效
                 sys.stdout.close()
            sys.stdout = self.old_stdout

        # 不抑制异常
        return False


# --- 从网站获取最新数据 (此函数不获取日期) ---

def fetch_latest_data(url: str = "https://www.17500.cn/chart/ssq-tjb.html") -> list:
    """从指定网站获取最新双色球数据 (不含日期)"""
    logger.info("正在从网站获取最新双色球数据...")
    data = []
    try:
        # 发送请求获取页面内容，禁用代理
        session = requests.Session()
        session.trust_env = False  # 禁用环境变量中的代理设置

        response = session.get(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }, timeout=30)  # 添加超时设置

        response.raise_for_status()  # 检查请求是否成功

        # 解析HTML
        soup = BeautifulSoup(response.text, 'html.parser')

        # 找到表格
        table = soup.find('table')
        if not table:
            logger.warning("无法在网页中找到表格数据")
            return []

        # 解析表格数据
        rows = table.find_all('tr')

        # 跳过表头，从第一行数据开始
        for row in rows[1:]:  # 假设第一行是表头
            cells = row.find_all('td')
            # 确保存在足够的单元格且不为空
            if len(cells) >= 3 and all(cell and cell.text.strip() for cell in cells[:3]):
                try:
                    # 提取期号
                    period_cell = cells[0]
                    if '<a' in str(period_cell):
                        period = period_cell.find('a').text.strip()
                    else:
                        period = period_cell.text.strip()

                    period = period.replace("期", "").strip()

                    # 检查期号是否为有效整数
                    if not period.isdigit():
                        continue

                    red_balls_str = cells[1].text.strip().replace(" ", ",")
                    blue_ball_str = cells[2].text.strip()

                    # 红球/蓝球的基本验证
                    if not red_balls_str or not blue_ball_str:
                         continue

                    # 可选：对红球格式进行更严格的检查
                    try:
                        red_numbers = [int(x) for x in red_balls_str.split(',')]
                        if len(red_numbers) != 6:
                            continue
                    except ValueError:
                         continue

                    try:
                        blue_number = int(blue_ball_str)
                        if not (1 <= blue_number <= 16):
                            continue
                    except ValueError:
                         continue

                    data.append({
                        '期号': period,
                        '红球': red_balls_str,
                        '蓝球': blue_ball_str
                    })
                except Exception as e:
                    logger.warning(f"处理行时出错: {cells}. 错误: {e}")
                    continue  # 跳过有问题的行

        logger.info(f"从网站成功获取了 {len(data)} 期双色球数据")
        return data

    except requests.exceptions.ProxyError:
        logger.warning("代理连接错误，尝试直接连接...")
        try:
            # 尝试不使用代理直接连接
            response = requests.get(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }, proxies={"http": None, "https": None}, timeout=30)

            response.raise_for_status()

            # 解析HTML - 与上面相同的逻辑
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table')
            if not table:
                logger.warning("直接连接：无法在网页中找到表格数据")
                return []

            data = []
            rows = table.find_all('tr')
            for row in rows[1:]:
                cells = row.find_all('td')
                if len(cells) >= 3 and all(cell and cell.text.strip() for cell in cells[:3]):
                    try:
                         period_cell = cells[0]
                         if '<a' in str(period_cell):
                              period = period_cell.find('a').text.strip()
                         else:
                              period = period_cell.text.strip()

                         period = period.replace("期", "").strip()
                         if not period.isdigit():
                              continue

                         red_balls_str = cells[1].text.strip().replace(" ", ",")
                         blue_ball_str = cells[2].text.strip()

                         if not red_balls_str or not blue_ball_str:
                              continue

                         try:
                              red_numbers = [int(x) for x in red_balls_str.split(',')]
                              if len(red_numbers) != 6:
                                   continue
                         except ValueError:
                              continue

                         try:
                              blue_number = int(blue_ball_str)
                              if not (1 <= blue_number <= 16):
                                   continue
                         except ValueError:
                              continue

                         data.append({
                             '期号': period,
                             '红球': red_balls_str,
                             '蓝球': blue_ball_str
                         })
                    except Exception as e:
                        logger.warning(f"直接连接处理行时出错: {cells}. 错误: {e}")
                        continue

            logger.info(f"直接连接成功获取了 {len(data)} 期双色球数据")
            return data

        except Exception as e:
            logger.error(f"直接连接也失败: {e}")
            return []

    except Exception as e:
        logger.error(f"获取网站数据时出错: {e}")
        return []

# --- 从txt文件获取并解析数据 (已存在) ---

def fetch_data_from_txt(url='http://data.17500.cn/ssq_asc.txt'):
    """从txt文件下载数据并解析"""
    logger.info(f"尝试从 {url} 下载数据...")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        response.encoding = 'utf-8'
        data_lines = response.text.strip().split('\n')
        logger.info(f"成功下载 {len(data_lines)} 行数据。")
        return data_lines
    except requests.exceptions.HTTPError as err:
        if "429" in str(err):
            logger.error("错误：请求过于频繁，请稍后再试。")
        else:
            logger.error(f"HTTP错误：{err}")
    except Exception as e:
        logger.error(f"数据下载失败：{e}")
    return None


def parse_txt_data(data_lines):
    """解析txt数据，提取所需字段"""
    if not data_lines:
        return []
    logger.info("正在解析txt数据...")
    parsed_data = []
    for line in data_lines:
        fields = line.strip().split()
        if len(fields) < 9:
            # logger.warning(f"忽略无效行：{line}") # 避免日志过多
            continue
        try:
            qihao = fields[0]
            date = fields[1]
            red_balls = ",".join(fields[2:8])
            blue_ball = fields[8]
            parsed_data.append([qihao, date, f'{red_balls}', blue_ball]) # 移除额外的引号，csv writer会处理
        except IndexError:
            logger.warning(f"数据格式异常：{line}")
            continue
    logger.info(f"解析出 {len(parsed_data)} 条有效数据。")
    return parsed_data


def update_csv_with_txt_data(csv_file_path, txt_data_parsed):
    """使用txt数据更新CSV文件，重点更新日期"""
    if not txt_data_parsed:
        logger.info("没有获取到txt解析数据，CSV文件保持不变")
        return False

    try:
        # 将txt解析数据转换为DataFrame
        new_data_df = pd.DataFrame(txt_data_parsed, columns=['期号', '日期', '红球', '蓝球'])
        # 确保期号是字符串类型，以便后续合并
        new_data_df['期号'] = new_data_df['期号'].astype(str)

        # 读取现有CSV文件
        existing_df = pd.DataFrame(columns=['期号', '日期', '红球', '蓝球'])
        if os.path.exists(csv_file_path) and os.path.getsize(csv_file_path) > 0:
            try:
                existing_df = pd.read_csv(csv_file_path, dtype={'期号': str}, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    existing_df = pd.read_csv(csv_file_path, dtype={'期号': str}, encoding='gbk')
                except UnicodeDecodeError:
                    try:
                        existing_df = pd.read_csv(csv_file_path, dtype={'期号': str}, encoding='latin-1')
                    except Exception as e:
                        logger.error(f"尝试多种编码读取CSV失败: {e}")
                        existing_df = pd.DataFrame(columns=['期号', '日期', '红球', '蓝球'], dtype=str)
            except pd.errors.EmptyDataError:
                logger.warning("现有CSV文件为空。")
                existing_df = pd.DataFrame(columns=['期号', '日期', '红球', '蓝球'], dtype=str)
        else:
             logger.info("CSV文件不存在或为空，将创建新文件。")
             existing_df = pd.DataFrame(columns=['期号', '日期', '红球', '蓝球'], dtype=str)

        # 使用merge进行外连接，保留所有期号
        # on='期号' 是合并的关键
        merged_df = pd.merge(existing_df, new_data_df, on='期号', how='outer', suffixes=('_old', '_new'))

        # 更新列：如果 _new 列有数据，则使用 _new 列的数据，否则使用 _old 列的数据
        # 这实现了优先使用txt数据更新
        for col in ['日期', '红球', '蓝球']:
             # 使用fillna合并数据，优先使用_new列
             merged_df[col] = merged_df[f'{col}_new'].fillna(merged_df[f'{col}_old'])
             # 删除旧的和新的临时列
             merged_df.drop(columns=[f'{col}_old', f'{col}_new'], inplace=True)

        # 确保所有需要的列都在最终的DataFrame中
        final_columns = ['期号', '日期', '红球', '蓝球']
        for col in final_columns:
            if col not in merged_df.columns:
                merged_df[col] = None # 或者设置为合适的默认值，如空字符串或 NaN

        # 重新排列列的顺序并排序
        final_df = merged_df[final_columns].sort_values('期号').reset_index(drop=True)

        # 保存到CSV
        try:
            # 使用csv模块写入，以便更好地控制引号和格式，特别是红球列
            final_df.to_csv(csv_file_path, index=False, encoding='utf-8', quoting=csv.QUOTE_MINIMAL)
            logger.info(f"CSV文件已成功更新至：{csv_file_path}。")
            return True
        except Exception as e:
             logger.error(f"保存CSV文件时出错: {e}")
             return False

    except Exception as e:
        logger.error(f"更新CSV文件时出错: {e}")
        return False

# --- 从网站获取最新数据并更新CSV (此函数不再是主要更新日期的方式) ---
# 保留此函数可能用于获取非txt文件中的数据，但用户要求重点更新日期依赖txt

def update_csv_with_latest_data(csv_file_path: str):
    """
    从网站获取最新数据并更新CSV文件。
    注意：此函数获取的数据不包含日期，日期的更新应优先使用txt文件。
    """
    logger.info("正在检查并更新最新双色球数据 (不含日期)...")
    latest_data = fetch_latest_data() # 此函数不获取日期
    if not latest_data:
        logger.info("没有获取到网站新数据，CSV文件保持不变 (非日期部分)。")
        return False

    try:
        # 读取现有CSV文件，保留日期列
        existing_df = pd.DataFrame(columns=['期号', '日期', '红球', '蓝球'])
        if os.path.exists(csv_file_path) and os.path.getsize(csv_file_path) > 0:
            try:
                existing_df = pd.read_csv(csv_file_path, dtype={'期号': str}, encoding='utf-8')
            except UnicodeDecodeError:
                 try:
                     existing_df = pd.read_csv(csv_file_path, dtype={'期号': str}, encoding='gbk')
                 except UnicodeDecodeError:
                      try:
                          existing_df = pd.read_csv(csv_file_path, dtype={'期号': str}, encoding='latin-1')
                      except Exception as e:
                           logger.error(f"尝试多种编码读取CSV失败: {e}")
                           existing_df = pd.DataFrame(columns=['期号', '日期', '红球', '蓝球'], dtype=str)
            except pd.errors.EmptyDataError:
                 logger.warning("现有CSV文件为空。")
                 existing_df = pd.DataFrame(columns=['期号', '日期', '红球', '蓝球'], dtype=str)
        else:
             logger.info("CSV文件不存在或为空，将创建新文件。")
             existing_df = pd.DataFrame(columns=['期号', '日期', '红球', '蓝球'], dtype=str)


        # 创建新数据DataFrame (来自网站，无日期)
        new_df = pd.DataFrame(latest_data)
        # 确保期号是字符串类型
        new_df['期号'] = new_df['期号'].astype(str)
        # 添加一个空的日期列，以便与现有数据合并
        new_df['日期'] = None # 或设置为pd.NA 或 ''

        # 使用merge进行外连接
        merged_df = pd.merge(existing_df, new_df, on='期号', how='outer', suffixes=('_old', '_new'))

        # 更新列：红球和蓝球优先使用新数据，日期保留旧数据（因为网站数据没有日期）
        # 注意这里的优先级逻辑：对于日期，我们希望保留来自txt的（_old），如果_old是空的才考虑其他来源（但这个函数没有其他日期来源）
        # 对于红蓝球，优先使用网站获取的 (_new)
        merged_df['红球'] = merged_df['红球_new'].fillna(merged_df['红球_old'])
        merged_df['蓝球'] = merged_df['蓝球_new'].fillna(merged_df['蓝球_old'])
        # 日期列直接使用_old，因为_new是空的
        merged_df['日期'] = merged_df['日期_old']


        # 删除旧的和新的临时列
        merged_df.drop(columns=[f'{col}_old' for col in ['日期', '红球', '蓝球']], inplace=True)
        merged_df.drop(columns=[f'{col}_new' for col in ['日期', '红球', '蓝球']], inplace=True)

        # 确保所有需要的列都在最终的DataFrame中
        final_columns = ['期号', '日期', '红球', '蓝球']
        for col in final_columns:
            if col not in merged_df.columns:
                merged_df[col] = None # 或者设置为合适的默认值

        # 重新排列列的顺序并排序
        final_df = merged_df[final_columns].sort_values('期号').reset_index(drop=True)

        # 保存到CSV
        try:
            final_df.to_csv(csv_file_path, index=False, encoding='utf-8', quoting=csv.QUOTE_MINIMAL)
            logger.info(f"CSV文件已成功更新非日期部分至：{csv_file_path}。")
            return True
        except Exception as e:
             logger.error(f"保存CSV文件时出错: {e}")
             return False

    except Exception as e:
        logger.error(f"更新CSV文件时出错: {e}")
        return False


# --- 主执行逻辑 ---
if __name__ == "__main__":
    logger.info("开始执行双色球数据处理...")

    # 1. 尝试从 ssq_asc.txt 获取数据并更新CSV (优先，尤其用于更新日期)
    txt_data_lines = fetch_data_from_txt()
    if txt_data_lines:
        txt_parsed_data = parse_txt_data(txt_data_lines)
        if txt_parsed_data:
            logger.info("使用ssq_asc.txt数据更新CSV文件...")
            update_csv_with_txt_data(CSV_FILE_PATH, txt_parsed_data)
        else:
            logger.warning("从ssq_asc.txt解析到的数据为空。")
    else:
        logger.warning("未能从ssq_asc.txt获取到数据，跳过txt更新步骤。")
        # 如果txt获取失败，可以选择是否尝试从网站获取其他数据
        # logger.info("尝试从网站获取最新数据作为补充...")
        # update_csv_with_latest_data(CSV_FILE_PATH) # 如果需要从网站获取红蓝球信息，可以启用此行
        # 注意：从网站获取的数据不含日期，如果txt是日期唯一来源，失败则日期无法更新


    logger.info("双色球数据处理完成。")
