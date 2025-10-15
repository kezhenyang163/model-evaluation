
ENGLISH_TO_CHINESE_PROMPT = """You are a bilingual translation expert in both Chinese and English, proficient in bilingual translation from English to Chinese, 
and have particularly in-depth experience in the translation of {} subjects
I need your help with the translation from English to Chinese
The following are the requirements
1. For the Json format data I provided, translate the Value part into Chinese. Keep the key in the json as English. When translating, the value corresponding to the question needs to maintain its original meaning
2. Maintain the original format. The text should be concise and not too long for easy browsing
3. There is no need to translate the data in the Answer section
4. The final result only needs to include the translated json data without any other prefixes or suffixes json data is organized into a single line of JSONL format data, with the newline characters remove
Please provide elements in Json format. Do not attach text at the beginning or end. Only translate the value in json. The json format must be accurate and usable"""


DATASET_EVALUATION_PROMPT_ENGLISH = """You are an AI assistant designed to answer multiple-choice questions. You will be given a JSON object containing a "question" and a "choices" array. Your task is to analyze the question, select the correct answer from the choices, and return a JSON object containing both your reasoning process and the selected answer index.
Instructions:
1. Carefully read and understand the question
2. Analyze all choices in the array and explain your reasoning
3. Select the most appropriate answer
4. Return a JSON object with exactly flow keys:
   - "model_thinking": a string containing your reasoning process
   - "model_answer": the numerical index (0-based) of your selected choice
   - "question" : original question must be chinese
   - "Choices" :  original choices array like ["choice one","choice two","choice three"]
5.the Value part Must Be Chinese ,resut choices order must be as same of the given choices order 

Output Requirements:
- The output must be a valid JSON object
- "model_thinking" should contain your complete reasoning process
- "model_answer" must be only the numerical index (0, 1, 2, etc.)
- Do not include any additional text outside the JSON object

Example Input:
question: 大脑检测刺激强度差异的能力，最能用以下哪项随刺激强度变化来解释？
choices": ["动作电位的幅度", "阈值电位", "每秒动作电位的数量", "跨越的突触数量"]

Example Output:
{ 
  "question": "大脑检测刺激强度差异的能力，最能用以下哪项随刺激强度变化来解释？", "choices": ["动作电位的幅度", "阈值电位", "每秒动作电位的数量", "跨越的突触数量"],
  "model_thinking": "这个问题是关于大脑如何检测刺激强度差异的。动作电位的幅度通常是全或无的，不随强度变化。阈值电位相对恒定。每秒动作电位的数量（放电频率）确实会随着刺激强度增加而增加，这是强度编码的关键机制。跨越的突触数量与强度编码没有直接关系。",
  "model_answer": 2
}

Now process the following question and choices, Output only the JSON object:
"""
