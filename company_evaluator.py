from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI
import openai

# 设置 OpenAI API 密钥

# 初始化聊天模型
chat_model = ChatOpenAI(temperature=0, model_name='gpt-4')

def evaluate_company(row, background, business_needs, prompt=None):
    # 如果没有提供 prompt，则使用默认的 prompt_statement
    if prompt is None:
        prompt = (
            "you are a professional industry expert in innovation, technology, investment, consulting. "
            "please consider the background: " + background + "\n\n" +
            "and the business needs: to find " + business_needs +
            " tell if the following company is in the target, if it matches the business needs and give the reason." + "\n\n" +
            "company info: "
        )
    
    question = prompt + "\n" + row.to_string()

    # 定义响应模式
    response_schemas = [
        ResponseSchema(name="answer", description="answer to the user's question"),
        ResponseSchema(name="Yes/No", description="tell if the company is in the target"),
        ResponseSchema(name="Reason", description="reason why you get this answer")
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    # 创建提示模板
    prompt_template = ChatPromptTemplate(
        messages=[
            HumanMessagePromptTemplate.from_template("answer the user's question as best as possible.\n{format_instructions}\n{question}")
        ],
        input_variables=["question"],
        partial_variables={"format_instructions": format_instructions}
    )

    # 格式化提示和获取输出
    _input = prompt_template.format_prompt(question=question)
    output = chat_model(_input.to_messages())

    # 解析输出
    parsed_output = output_parser.parse(output.content)
    return parsed_output
