import gradio as gr

from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from openai import OpenAI


API_KEY=""
lm_model = "gpt-3.5-turbo"
def initialize_sales_bot(vector_store_dir: str="real_estates_sale"):

    with open("real_estate_sales_data.txt","r",encoding='utf8') as f:
        real_estate_sales = f.read()
    text_splitter = CharacterTextSplitter(
        separator = r'\d+\.',
        chunk_size = 100,
        chunk_overlap  = 0,
        length_function = len,
        is_separator_regex = True,
    )
    docs = text_splitter.create_documents([real_estate_sales])
    db = FAISS.from_documents(docs, OpenAIEmbeddings(openai_api_key=API_KEY))
    retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.8})   
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0,openai_api_key=API_KEY)
    
    global SALES_BOT   
    SALES_BOT = RetrievalQA.from_chain_type(llm,retriever=retriever)
    # 返回向量数据库的检索结果
    SALES_BOT.return_source_documents = True

    return SALES_BOT

def initialize_elec_bot(vector_store_dir: str="electronics_sale"):

    with open("electronics_sales_data.txt","r",encoding='utf8') as f:
        reference_data = f.read()
    text_splitter = CharacterTextSplitter(
        separator = r'\d+\.',
        chunk_size = 100,
        chunk_overlap  = 0,
        length_function = len,
        is_separator_regex = True,
    )
    docs = text_splitter.create_documents([reference_data])
    db = FAISS.from_documents(docs, OpenAIEmbeddings(openai_api_key=API_KEY))
    retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.8})   
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0,openai_api_key=API_KEY)
    
    global ELECTRONICS_BOT   
    ELECTRONICS_BOT = RetrievalQA.from_chain_type(llm,retriever=retriever)
    # 返回向量数据库的检索结果
    ELECTRONICS_BOT.return_source_documents = True

    return ELECTRONICS_BOT

def initialize_deco_bot(vector_store_dir: str="decoration_sale"):

    with open("decoration_sales_data.txt","r",encoding='utf8') as f:
        reference_data = f.read()
    text_splitter = CharacterTextSplitter(
        separator = r'\d+\.',
        chunk_size = 100,
        chunk_overlap  = 0,
        length_function = len,
        is_separator_regex = True,
    )
    docs = text_splitter.create_documents([reference_data])
    db = FAISS.from_documents(docs, OpenAIEmbeddings(openai_api_key=API_KEY))
    retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.8})   
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0,openai_api_key=API_KEY)
    
    global DECORATION_BOT   
    DECORATION_BOT = RetrievalQA.from_chain_type(llm,retriever=retriever)
    # 返回向量数据库的检索结果
    DECORATION_BOT.return_source_documents = True

    return DECORATION_BOT

def initialize_edu_bot(vector_store_dir: str="education_sale"):

    with open("education_sales_data.txt","r",encoding='utf8') as f:
        reference_data = f.read()
    text_splitter = CharacterTextSplitter(
        separator = r'\d+\.',
        chunk_size = 100,
        chunk_overlap  = 0,
        length_function = len,
        is_separator_regex = True,
    )
    docs = text_splitter.create_documents([reference_data])
    db = FAISS.from_documents(docs, OpenAIEmbeddings(openai_api_key=API_KEY))
    retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.8})   
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0,openai_api_key=API_KEY)
    
    global EDUCATION_BOT   
    EDUCATION_BOT = RetrievalQA.from_chain_type(llm,retriever=retriever)
    # 返回向量数据库的检索结果
    EDUCATION_BOT.return_source_documents = True

    return EDUCATION_BOT


def sales_chat(message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")
    # TODO: 从命令行参数中获取
    enable_chat = True
    #determine topic with LLM
    topics_in_str = " or ".join([f"'{k}'" for k in bot_dict.keys()])
    try:
        client = OpenAI(api_key=API_KEY)
        completion = client.chat.completions.create(
            model=lm_model,
            messages=[
                {"role": "system", "content": f"Only reply {topics_in_str}"},
                {"role": "user", "content": f"classify the topic among 'real_estate' or 'housing_decoration' or 'electronic_appliances' or 'education':'{message}'"}
            ],
            temperature=0.7,
        )
        specialization = completion.choices[0].message.content
        assert specialization in bot_dict.keys()
        print(f"topic settled to be {specialization}")
    except Exception as e:
        print(f"topic judgement failed: {e}")
        specialization = ""
    # based on LLM determine the bot, defalt to real_estate
    specialist_bot = bot_dict.get(specialization,SALES_BOT)

    ans = specialist_bot({"query": message})
    # 如果检索出结果，或者开了大模型聊天模式
    # 返回 RetrievalQA combine_documents_chain 整合的结果
    if ans["source_documents"] or enable_chat:
        print(f"[result]{ans['result']}")
        print(f"[source_documents]{ans['source_documents']}")
        return ans["result"]
    # 否则输出套路话术
    else:
        return "这个问题我要问问领导"
    

def launch_gradio():
    demo = gr.ChatInterface(
        fn=sales_chat,
        title="房产/电器/家装/教育-销售",
        # retry_btn=None,
        # undo_btn=None,
        chatbot=gr.Chatbot(height=600),
    )

    demo.launch(share=True, server_name="0.0.0.0")

if __name__ == "__main__":
    # 初始化房产销售机器人
    initialize_sales_bot(vector_store_dir="real_estate_sales_data.txt")
    initialize_elec_bot(vector_store_dir="electronics_sales_data.txt")
    initialize_deco_bot(vector_store_dir="decoration_sales_data.txt")
    initialize_edu_bot(vector_store_dir="education_sales_data.txt")
    bot_dict = {
        "real_estate":SALES_BOT,
        "electronic_appliances":ELECTRONICS_BOT,
        "housing_decoration":DECORATION_BOT,
        "education":EDUCATION_BOT,
        }
    # 启动 Gradio 服务
    launch_gradio()
