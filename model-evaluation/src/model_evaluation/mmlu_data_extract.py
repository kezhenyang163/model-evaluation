import json
import logging
from datetime import datetime, date

import pandas as pd
import numpy as np
import openai
import os
from dotenv import load_dotenv, find_dotenv

from prompt import model_prompts
from src.model_evaluation.answer_cacluate import AnswerAccuracyCalculator, load_json_from_file
from utils.logger import StructuredLogger

_ = load_dotenv(find_dotenv())
# 获取环境变量 OPENAI_API_KEY
client = openai.OpenAI(base_url=os.environ['DEEPSEEK_BASE_URL'], api_key=os.environ['DEEPSEEK_API_KEY'])
big_model = os.environ['DEEPSEEK_BASE_MODEL']

pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.max_rows', None)  # 显示所有行
pd.set_option('display.width', None)  # 不限制显示宽度
pd.set_option('display.max_colwidth', 50)  # 不限制列宽

datadir = 'D:/work/github/model-evaluation/model-evaluation/dataset/mmlu/test-00000-of-00001.parquet'

logger = StructuredLogger(__name__)
logger.log(logging.INFO, "执行日志")


def get_random_data():
    """
    随机从 modelscope 或者huggingface 下载的数据文件中获取随机的10条数据
    """

    df = pd.read_parquet(datadir)
    # np.random.seed(142)  # 设置随机种子以便复现
    numbers = [np.random.randint(0, len(df)) for _ in range(10)]
    init_df = df.iloc[numbers]
    logger.log(logging.INFO, "文件读取成功，获取随机10条数据成功")
    return init_df


def save_json_lines_simple(df, filename):
    json_list = []
    """最简单的方法 - 每行一个 JSON 对象"""
    with (open(filename, 'w', encoding='utf-8') as f):
        # 使用 lines 格式，每行一个 JSON 对象
        json_dict_list = df.to_dict(orient='records')
        for json_dict in json_dict_list:
            json_str = json.dumps(json_dict, default=safe_json_dumps, ensure_ascii=False, indent=2,
                                  separators=(",", ":"))
            result_str = json_str.replace('\n', '').replace('\r', '').replace('\t', '')
            f.write(result_str + '\n')
            json_list.append(result_str)
    logger.log(logging.INFO, "数据转换Json 成功 Json的条数是：{}".format(len(json_list)))
    return json_list


def safe_json_dumps(obj):
    """
    安全的 JSON 序列化函数，自动处理 numpy 和 pandas 数据类型
    """
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    elif isinstance(obj, (pd.Timestamp, datetime, date)):
        return obj.isoformat()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    else:
        raise TypeError(f"无法序列化的类型: {type(obj)}")


def get_response(model=big_model, prompt: str = "请介绍一下你自己", max_tokens=3000):
    """调用模型获取prompt 对应的结果数据"""
    messages = [
        {"role": "user", "content": prompt},
    ]
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_completion_tokens=max_tokens
    )
    return completion.choices[0].message.content


def get_response_with_temperature(model=big_model, prompt: str = "请介绍一下你自己", max_tokens=3000, temperature=0.7):
    """调用模型获取prompt 对应的结果数据"""
    messages = [
        {"role": "user", "content": prompt},
    ]
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_completion_tokens=max_tokens,
        temperature=temperature
    )
    return completion.choices[0].message.content


def transfer_to_chinese():
    """
    将给定的MMLU 数据中的数据进行取样，取样之后转换成为Json 数据，通过提示词工程让大模型转换其中的 key 对应的Value 为Json 数据
    """
    # 从文件读取MMLU 某个Subject 的数据并取样10条数据
    org_sample_df = get_random_data()
    org_sample_json_list = save_json_lines_simple(org_sample_df, "mmlu_sample_data.json")
    prompt = model_prompts.ENGLISH_TO_CHINESE_PROMPT.format('college_biology')
    chinese_json_data = []
    logger.log(logging.INFO, "开始翻译中文")
    for sample_json in org_sample_json_list:
        logger.log(logging.INFO, "开始翻译 第{} 条数据".format(len(chinese_json_data) + 1))
        generate_prompt = f"{prompt}\n json data is : {sample_json}"
        response = get_response(prompt=generate_prompt)
        chinese_question = response.replace('\n', '').replace('\r', '').replace(' ', '')
        chinese_json_data.append(chinese_question)

    logger.log(logging.INFO, "翻译中文成功 数据的条数是：{}".format(len(chinese_json_data)))
    return chinese_json_data


def append_to_json_file(new_data, filename="all_results.json"):
    """
    将新数据追加到JSON文件中（如果文件已存在）

    Args:
        new_data: 要追加的新数据
        filename: 文件名
    """

    # 如果文件不存在，创建新文件并写入数据
    if not os.path.exists(filename):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, ensure_ascii=False, indent=2)
    else:
        # 如果文件存在，读取现有数据，追加新数据，再写回
        with open(filename, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)

        # 确保existing_data是列表
        if not isinstance(existing_data, list):
            existing_data = [existing_data]

        # 追加新数据
        existing_data.append(new_data)

        # 写回文件
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)

    print(f"数据已追加到 {filename}")

if __name__ == '__main__':
    chinese_data = transfer_to_chinese()
    result_path = "mmlu_data_result.json"
    extract_result = []
    for chinese_json in chinese_data:
        json_obj = json.loads(chinese_json)
        question = json_obj['question']
        choices = json_obj['choices']
        answer = json_obj['answer']
        prompt = model_prompts.DATASET_EVALUATION_PROMPT_ENGLISH + " question :{}".format(
            question) + "\n choices :{}".format(choices) + ""
        answer_response = get_response_with_temperature(prompt=prompt, temperature=0)
        response_dict = json.loads(answer_response)
        response_dict['answer'] = answer
        logger.log(logging.INFO, "第{}条数据的获取答案成功".format(len(extract_result) + 1))
        extract_result.append(response_dict)
    append_to_json_file(extract_result,result_path)
    logger.log(logging.INFO, "保存结果到文件成功, 数据条数为{}".format(len(extract_result)))

   # 测评文件中的数据准确率
    calculator = AnswerAccuracyCalculator(load_json_from_file(result_path))
    calculator.run_analysis()
