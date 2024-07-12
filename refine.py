import openai
import logging

# 设置OpenAI API密钥


def refine_input(matching_type,user_input):
    """
    使用GPT-4o API将用户的输入进行精炼。
    
    参数:
    user_input (str): 用户的输入文本
    
    返回:
    str: 精炼后的文本
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": """
                    You are a helpful assistant. Your task is to refine the user's input to be concise, clear and detailed. 
                    The refined input will be used to calculate similarity with a vector database of 

                    Please perform the following steps:
                    1. Understand the matching type;
                    2. Understand the type of companies/Investor/Solution the user is looking for based on the provided text.
                    3. Output a refined version that contains key characteristics of the needed companie/Investor/Solution .
                    
                    The refined input should include:
                    - Type of companies/Investor/Solution 
                    - Services or products offered
                    - Technologies used, including detailed tech, solutions, services, and products, this is must.
                    
                    Examples:
                    1. "A company focusing on 3D printer development and production to promote 3D printing technology by producing high-quality printers at a reasonable price, "
                    2. "An investor forcusing in renewable energy solutions, providing innovative solar panel technology to increase energy efficiency and sustainability,"
                    3. "An AI-driven tools for financial analytics to help businesses make data-driven decisions."
                    except the above info, after the example, add the details of the technology, solution and service that related with the companies/Investor/Solution .....
                """},
                {"role": "user", "content": f"Refine the following input: {matching_type+user_input}"}
            ],
            max_tokens=1000,
            n=1,
            stop=None,
            temperature=0.0,
        )
        
        refined_text = response['choices'][0]['message']['content'].strip()
        return refined_text
    
    except openai.error.OpenAIError as e:
        logging.error(f"OpenAI API request failed: {e}")
        return None

# 示例调用
if __name__ == "__main__":
    user_input = """先进制造"""
    refined_text = refine_input(user_input)
    if refined_text:
        print(f"精炼后的文本: {refined_text}")
    else:
        print("精炼文本失败，请检查日志。")
