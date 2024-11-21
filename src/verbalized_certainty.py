from config_file import certainty_dict,model_dict
from openai import OpenAI as OpenAI_Official  # 导入官方的OpenAI库，用于调用GPT-4模型
import os
from dotenv import load_dotenv
from utils import get_model_and_tokenizer
import torch

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_API_BASE = os.getenv('OPENAI_API_BASE')
official_openai_client = OpenAI_Official(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)  


# 调用GPT-4模型获取信心等级评估
def get_gpt4_confidence_response(prompt):
    """获取gpt模型的接口"""
    # 使用官方OpenAI API调用GPT-4
    response = official_openai_client.chat.completions.create(
        model="gpt-4o-mini",  # 指定GPT-4模型
        messages=[{"role": "user", "content": prompt}],  # 用户输入的内容
        temperature=1
    )
    return response.choices[0].message.content.strip()  # 提取并返回GPT-4的输出文本


# model = AutoModelForCausalLM.from_pretrained(model_id)
# tokenizer = AutoTokenizer.from_pretrained(model_id)
def get_opensource_model_confidence_response(prompt,model_id):
    """
    获取开源模型的置信度
    """
    model,tokenizer=get_model_and_tokenizer(model_id=model_id)
    # 对问题进行 tokenize
    prompt_encoding = tokenizer(prompt, return_tensors="pt")
    prompt_ids = prompt_encoding.input_ids
    anwser=""
    for _ in range(20):  # 生成 2 个 token
        detailed_output = model(prompt_ids)
        logits = detailed_output.logits
        next_token_id = torch.argmax(logits[0, -1, :]).item() # 选择概率最高的 token
        next_token_text = tokenizer.decode(next_token_id)
        anwser += next_token_text+" "
        # 更新 prompt_ids 以继续生成
        prompt_ids = torch.cat([prompt_ids, torch.tensor([[next_token_id]])], dim=-1)
        # print(anwser,"is 换行",next_token_text in ["\n","<|end|>"])
        if next_token_text in ["1","2","3","4","5","6"]:
            return anwser
    # print("置信度解码无选项数字")
    anwser="置信度解码无选项数字"
    return anwser

def generate_verbalized_certainty(previous_prompt:str, previous_response:str, model_id:str, temp: float):
    """
    生成口头化的确定性描述和确定性值。
    参数:
        prompt (str): 提示信息。
        choices (dict): 包含选项的字典。
        response_text (str): 之前生成的答案.
        model_id(str)
        temp（double）温度
    """
    
    # 构建一个提示，要求模型分析其答案的确定性
    confidence_prompt = f"""  
The language model has been asked to complete the following task: ###{previous_prompt}###

Its answer is: {previous_response}  

Now, evaluate the your certainty in its answer by considering the options provided. How confident are you about the model’s answer?Please choose a level of confidence from the following options:


1. Very uncertain
2. Not certain
3. Somewhat certain
4. Moderately certain
5. Fairly certain
6. Very certain

Please only answer with a number from 1 to 6, where 1 indicates low confidence and 6 indicates high confidence.

    """

    # 使用GPT-4模型获取口头信心等级
    gpt4_confidence_text = get_gpt4_confidence_response(confidence_prompt)  # 获取GPT-4模型的信心等级文本
    print("GPT4给的言语置信度",gpt4_confidence_text)

    if model_dict[model_id]["is_open"]:
        # 遍历字典，查找与模型回答相匹配的口头化确定性描述，并获取其数值
        opensourced_model_confidence_text=get_opensource_model_confidence_response(confidence_prompt,model_id=model_id)
        # 定义分数字典，将口头信心等级文本映射为数值
        print(model_id,"给的言语置信度(verbalize_certainty)",opensourced_model_confidence_text)
        
    return gpt4_confidence_text,opensourced_model_confidence_text
    