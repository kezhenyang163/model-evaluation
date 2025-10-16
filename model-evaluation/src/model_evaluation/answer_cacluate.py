import json
from typing import List, Dict, Any

class AnswerAccuracyCalculator:
    def __init__(self, json_data: List[Dict[str, Any]]):
        """
        初始化计算器
        Args:
            json_data: JSON数组数据
        """
        self.json_data = json_data
        self.size = len(json_data)
        self.correct_count = 0
        self.results = []

    def calculate_accuracy(self) -> float:
        """
        计算准确率

        Returns:
            准确率百分比
        """
        self.correct_count = 0

        for i, item in enumerate(self.json_data, 1):
            question = item.get("question", "")
            model_answer = item.get("model_answer", "")
            answer = item.get("answer", "")

            # 记录每个问题的结果
            result = {
                "question_number": i,
                "question": question,
                "model_answer": model_answer,
                "answer": answer,
                "is_correct": False
            }

            # 比较答案是否一致
            if str(model_answer) == str(answer):
                self.correct_count += 1
                result["is_correct"] = True

            self.results.append(result)

        # 计算准确率
        if self.size > 0:
            accuracy = (self.correct_count / self.size) * 100
        else:
            accuracy = 0.0

        return accuracy

    def print_detailed_results(self):
        """打印详细结果"""
        for result in self.results:
            print(f"question{result['question_number']}：\"{result['question']}\"")
            print(f"正确答案：{result['answer']}")
            print(f"模型结果：{result['model_answer']}")
            print()  # 空行分隔

    def print_final_accuracy(self, accuracy: float):
        """打印最终准确率"""
        print(f"准确率为：{accuracy:.2f} %")

    def run_analysis(self):
        """运行完整分析"""
        accuracy = self.calculate_accuracy()
        self.print_detailed_results()
        self.print_final_accuracy(accuracy)

        return accuracy


def load_json_from_file(file_path: str) -> List[Dict[str, Any]]:
    """
    从文件加载JSON数据

    Args:
        file_path: JSON文件路径

    Returns:
        JSON数据列表
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def load_json_from_string(json_str: str) -> List[Dict[str, Any]]:
    """
    从字符串加载JSON数据

    Args:
        json_str: JSON格式的字符串

    Returns:
        JSON数据列表
    """
    return json.loads(json_str)


# 使用示例
if __name__ == "__main__":
    # 示例JSON数据
    json_data_path = 'mmlu_data_result.json'
    calculator = AnswerAccuracyCalculator(load_json_from_file(json_data_path))
    calculator.run_analysis()

    # print("\n" + "=" * 50)
    # print("额外统计信息：")
    # print(f"总问题数: {calculator.size}")
    # print(f"正确回答数: {calculator.correct_count}")

