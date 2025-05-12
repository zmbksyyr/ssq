import pandas as pd
import sys
import datetime
import os
import requests
from bs4 import BeautifulSoup
import io
import logging
from contextlib import redirect_stdout, redirect_stderr

# --- 配置 ---
# 获取脚本目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 构建CSV文件的完整路径
CSV_FILE_PATH = os.path.join(SCRIPT_DIR, 'shuangseqiu.csv')  # 假设shuangseqiu.csv与脚本在同一目录

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
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='', file=sys.stdout, flush=True)
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


# --- 从网站获取最新数据 ---

def fetch_latest_data(url: str = "https://www.17500.cn/chart/ssq-tjb.html") -> list:
    """从指定网站获取最新双色球数据"""
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


def update_csv_with_latest_data(csv_file_path: str):
    """获取最新数据并更新CSV文件"""
    logger.info("正在检查并更新最新双色球数据...")
    latest_data = fetch_latest_data()
    if not latest_data:
        logger.info("没有获取到新数据，CSV文件保持不变")
        return False

    try:
        # 读取现有CSV文件
        existing_df = pd.DataFrame(columns=['期号', '红球', '蓝球'])  # 如果文件不存在或为空，则以空DataFrame开始
        if os.path.exists(csv_file_path) and os.path.getsize(csv_file_path) > 0:
            try:
                existing_df = pd.read_csv(csv_file_path, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    existing_df = pd.read_csv(csv_file_path, encoding='gbk')
                except UnicodeDecodeError:
                    try:
                        existing_df = pd.read_csv(csv_file_path, encoding='latin-1')
                    except Exception as e:
                        logger.error(f"尝试多种编码读取CSV失败: {e}")
                        existing_df = pd.DataFrame(columns=['期号', '红球', '蓝球'])  # 读取错误时回退为空DataFrame
            except pd.errors.EmptyDataError:
                 logger.warning("现有CSV文件为空。")
                 existing_df = pd.DataFrame(columns=['期号', '红球', '蓝球'])  # 处理空文件

        # 确保'期号'存在且可以转换为整数
        if '期号' not in existing_df.columns:
             logger.warning("现有CSV没有'期号'列。使用最新数据重新开始。")
             existing_df = pd.DataFrame(columns=['期号', '红球', '蓝球'])
        else:
            try:
                existing_df['期号'] = pd.to_numeric(existing_df['期号'], errors='coerce').astype('Int64')  # 使用可空整数类型
                existing_df.dropna(subset=['期号'], inplace=True)  # 删除无法转换的行
                existing_df['期号'] = existing_df['期号'].astype(int)  # 删除NaN后转换为标准int
            except Exception as e:
                logger.error(f"将现有CSV中的'期号'转换为整数时失败: {e}。重新开始。")
                existing_df = pd.DataFrame(columns=['期号', '红球', '蓝球'])

        # 创建新数据DataFrame
        new_df = pd.DataFrame(latest_data)
        if '期号' not in new_df.columns:
             logger.error("获取的数据没有'期号'列。")
             return False

        # 将新数据的'期号'转换为整数，处理潜在错误
        try:
            new_df['期号'] = pd.to_numeric(new_df['期号'], errors='coerce').astype('Int64')  # 使用可空整数类型
            new_df.dropna(subset=['期号'], inplace=True)
            new_df['期号'] = new_df['期号'].astype(int)
        except Exception as e:
             logger.error(f"将获取数据中的'期号'转换为整数时失败: {e}")
             return False

        # 查找新条目（在new_df中但不在existing_df中，基于'期号'）
        existing_periods = set(existing_df['期号'])
        new_entries_df = new_df[~new_df['期号'].isin(existing_periods)].copy()

        # 如果有新数据，追加并保存
        if not new_entries_df.empty:
            # 合并前确保列匹配
            # 如果existing_df为空或重置，使用new_entries_df的列
            if existing_df.empty:
                combined_df = new_entries_df
            else:
                # 确保两个dataframes具有相同顺序的相同列
                common_cols = list(existing_df.columns.intersection(new_entries_df.columns))
                combined_df = pd.concat([existing_df[common_cols], new_entries_df[common_cols]], ignore_index=True)

            combined_df = combined_df.sort_values(by='期号', ascending=True).drop_duplicates(subset=['期号'], keep='last')  # 排序并去除重复项
            combined_df.reset_index(drop=True, inplace=True)

            # 保存更新的数据 - 始终使用utf-8以保持一致性
            combined_df.to_csv(csv_file_path, index=False, encoding='utf-8')
            logger.info(f"成功添加了 {len(new_entries_df)} 期新数据到CSV文件")
            return True
        else:
            logger.info("没有找到新的期号数据，CSV文件保持不变")
            return False

    except Exception as e:
        logger.error(f"更新CSV文件时出错: {e}")
        return False


# 如果直接运行此脚本
if __name__ == "__main__":
    # 更新数据
    update_result = update_csv_with_latest_data(CSV_FILE_PATH)
    if update_result:
        logger.info("成功更新了CSV数据。")
    else:
        logger.info("CSV数据未更新，或更新过程中出错。")
    
    # 获取当前日期时间作为时间戳
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"数据获取完成，运行时间: {timestamp}")
