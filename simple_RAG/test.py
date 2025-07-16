# -*- coding: utf-8 -*-
"""
Local RAG Pipeline - 本地检索增强生成系统
"""

# =============================
# 导入依赖库
# =============================
import random
import torch
import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from sentence_transformers import SentenceTransformer, util
from time import perf_counter as timer
import textwrap

# =============================
# 配置与全局变量设置
# =============================
device = "cuda" if torch.cuda.is_available() else "cpu"

# 模型路径（根据你的实际路径修改）
embedding_model_path = r'D:\python代码调试\大模型实战\simple-local-rag\model\all-mpnet-base-v2'
llm_model_path = r'D:\python代码调试\大模型实战\simple-local-rag\model\gemma-2b-it'

# =============================
# 工具函数定义区
# =============================

def print_wrapped(text, wrap_length=80):
    """
    打印自动换行的文本，提高终端输出可读性。
    
    参数:
        text (str): 要打印的文本。
        wrap_length (int): 每行字符数，默认为 80。
    """
    wrapped_text = textwrap.fill(str(text), wrap_length)
    print(wrapped_text)

# =============================
# 数据加载与预处理
# =============================

def load_text_chunks_and_embeddings(csv_path="simple-local-rag/text_chunks_and_embeddings_df.csv"):
    """
    加载 CSV 文件中的文本块和对应的嵌入向量。
    
    参数:
        csv_path (str): CSV 文件路径。
        
    返回:
        list[dict]: 包含文本块和嵌入向量的字典列表。
    """
    # 读取 CSV 文件
    df = pd.read_csv(csv_path)

    # 将字符串格式的 embedding 列转换回 numpy 数组
    def safe_load_embedding(embedding_str):
        try:
            return np.fromstring(embedding_str.strip('[]'), sep=',', dtype=np.float32)
        except Exception as e:
            print(f"[警告] 解析 embedding 失败：{e}")
            return None

    df["embedding"] = df["embedding"].apply(safe_load_embedding)
    df.dropna(subset=["embedding"], inplace=True)  # 删除解析失败的行

    # 转换为字典列表
    pages_and_chunks = df.to_dict(orient="records")

    return pages_and_chunks

# =============================
# 嵌入模型加载与使用
# =============================

def get_embedding_model(model_name_or_path=embedding_model_path, device=device):
    """
    加载 SentenceTransformer 嵌入模型。
    
    参数:
        model_name_or_path (str): 模型名称或路径。
        device (str): 使用设备 ('cuda' 或 'cpu')。
        
    返回:
        SentenceTransformer: 加载好的嵌入模型。
    """
    return SentenceTransformer(model_name_or_path, device=device)

def convert_embeddings_to_tensor(pages_and_chunks, device=device):
    """
    将嵌入向量列表转换为 PyTorch 张量。
    
    参数:
        pages_and_chunks (list[dict]): 文本块列表，包含 'embedding' 字段。
        device (str): 使用设备 ('cuda' 或 'cpu')。
        
    返回:
        torch.Tensor: 形状为 [N, D] 的嵌入张量。
    """
    embeddings = torch.tensor(
        np.array([item["embedding"] for item in pages_and_chunks]),
        dtype=torch.float32
    ).to(device)
    return embeddings

# =============================
# 语义检索模块
# =============================

def retrieve_relevant_resources(query: str, embeddings: torch.Tensor, model: SentenceTransformer, n_resources_to_return: int = 5):
    """
    根据查询获取最相关的文本块。
    
    参数:
        query (str): 用户输入的自然语言查询。
        embeddings (torch.Tensor): 所有文本块的嵌入向量。
        model (SentenceTransformer): 用于编码查询的嵌入模型。
        n_resources_to_return (int): 返回的 top-k 结果数量。
        
    返回:
        tuple(torch.Tensor, torch.Tensor): 分数和索引。
    """
    query_embedding = model.encode(query, convert_to_tensor=True).to(embeddings.device)
    dot_scores = util.dot_score(query_embedding, embeddings)[0]
    scores, indices = torch.topk(dot_scores, k=n_resources_to_return)
    return scores, indices

# =============================
# LLM 模型加载与生成
# =============================

def load_local_llm(model_path, load_in_4bit=True):
    """
    加载本地大语言模型（支持量化加载）。
    
    参数:
        model_path (str): 模型保存路径。
        load_in_4bit (bool): 是否启用 4-bit 量化。
        
    返回:
        tokenizer, model: 分词器和模型实例。
    """
    quantization_config = BitsAndBytesConfig(load_in_4bit=load_in_4bit, bnb_4bit_compute_dtype=torch.float16)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto"
    )

    return tokenizer, model

def generate_answer(tokenizer, model, prompt):
    """
    使用本地 LLM 生成回答。
    
    参数:
        tokenizer: 分词器。
        model: 本地加载的大语言模型。
        prompt (str): 构造好的提示词。
        
    返回:
        str: LLM 生成的回答。
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.7)
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # if full_response:
    #     # Replace special tokens and unnecessary help message
    #     answer = full_response.replace(prompt, "").replace("<bos>", "").replace("<eos>", "").replace("Sure, here is the answer to the user query:\n\n", "")
    
    # 提取"Answer:"之后的内容
    answer_key = "Answer:"
    if answer_key in full_response:
        answer_start = full_response.index(answer_key) + len(answer_key)
        answer = full_response[answer_start:].strip()
    else:
        answer = full_response
    
    return answer

# =============================
# 提示词构造（Prompt Engineering）
# =============================

def build_prompt_with_context(query: str, context_items: list[dict]):
    """
    构造带有上下文信息的提示词。
    
    参数:
        query (str): 用户问题。
        context_items (list[dict]): 检索出的相关文本块。
        
    返回:
        str: 完整的 prompt。
    """
    """
    构造带有上下文信息的提示词。
    
    参数:
        query (str): 用户问题。
        context_items (list[dict]): 检索出的相关文本块。
        
    返回:
        str: 完整的 prompt。
    """
    context = "- " + "\n- ".join([item["chunk"] for item in context_items])
    
    base_prompt = f"""Based on the following context, please answer the query concisely and accurately.
Only return the answer itself, without any additional explanations or references to the context.

Context:
{context}

Query: {query}
Answer:"""
    
    dialogue_template = [
        {"role": "user",
        "content": base_prompt}
    ]
    
    # Apply the chat template
    prompt = tokenizer.apply_chat_template(conversation=dialogue_template,
                                        tokenize=False,
                                        add_generation_prompt=True)
    return prompt

# =============================
# 主程序执行区
# =============================

if __name__ == "__main__":
    # Step 1: 加载并预处理数据
    print("【Step 1】加载文本块和嵌入向量...")
    pages_and_chunks = load_text_chunks_and_embeddings()
    print(f"成功加载 {len(pages_and_chunks)} 个文本块\n")

    # Step 2: 加载嵌入模型和嵌入向量
    print("【Step 2】加载嵌入模型并转换为张量...")
    embedding_model = get_embedding_model()
    embeddings = convert_embeddings_to_tensor(pages_and_chunks)
    print(f"Embeddings shape: {embeddings.shape}\n")

    # Step 3: 加载本地 LLM 模型
    print("【Step 3】加载本地大语言模型（LLM）...")
    tokenizer, llm_model = load_local_llm(model_path=llm_model_path)
    print("模型加载完成\n")

    # Step 4: 用户输入查询并进行语义检索
    print("【Step 4】请输入你的问题：")
    query = input()

    print("\n【Step 5】正在执行语义搜索...\n")
    scores, indices = retrieve_relevant_resources(
        query=query,
        embeddings=embeddings,
        model=embedding_model,
        n_resources_to_return=5
    )

    # Step 5.1: 收集检索结果
    retrieved_results = []
    print("【检索结果】\n")
    for score, index in zip(scores, indices):
        chunk = pages_and_chunks[index]["chunk"]
        page_number = pages_and_chunks[index]["page_number"]
        retrieved_results.append({
            "score": float(score),
            "chunk": chunk,
            "page_number": int(page_number)
        })
        print(f"Score: {score:.4f}")
        print_wrapped(chunk)
        print(f"Page number: {page_number}")
        print("-" * 80)

    # Step 6: 构造 Prompt 并调用 LLM 生成答案
    print("\n【Step 6】正在生成回答...\n")
    context_items = [pages_and_chunks[i] for i in indices.tolist()]
    prompt = build_prompt_with_context(query=query, context_items=context_items)
    answer = generate_answer(tokenizer=tokenizer, model=llm_model, prompt=prompt)

    # Step 7: 展示最终结构化输出
    print("\n" + "=" * 80 + "\n")
    print("【用户问题】")
    print_wrapped(query)
    print("\n【大模型回答】")
    print_wrapped(answer)
    print("\n" + "=" * 80)

    # 可选：打印原始 prompt（用于调试）
    # print("\n【Prompt】\n")
    # print(prompt)

    # 可选：将结果保存为 JSON（用于后续处理或展示）
    # import json
    # result = {
    #     "query": query,
    #     "retrieved_results": retrieved_results,
    #     "answer": answer
    # }
    # with open("rag_result.json", "w", encoding="utf-8") as f:
    #     json.dump(result, f, ensure_ascii=False, indent=2)