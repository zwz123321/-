# Download PDF file
import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import requests
import fitz 
from tqdm.auto import tqdm # for progress bars, requires !pip install tqdm 
import random
import pandas as pd
from spacy.lang.en import English 
from sentence_transformers import SentenceTransformer
import numpy as np
import torch

# Get PDF document
pdf_path = "simple-local-rag/human-nutrition-text.pdf"
device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available, otherwise CPU
# Download PDF if it doesn't already exist
if not os.path.exists(pdf_path):
    print("File doesn't exist, downloading...")

    # The URL of the PDF you want to download
    url = "https://pressbooks.oer.hawaii.edu/humannutrition2/open/download?type=pdf"

    # The local filename to save the downloaded file
    filename = pdf_path

    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Open a file in binary write mode and save the content to it
        with open(filename, "wb") as file:
            file.write(response.content)
        print(f"The file has been downloaded and saved as {filename}")
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")
else:
    print(f"File {pdf_path} exists.")
    
# Load PDF document
def text_formatter(text: str) -> str:
    """Performs minor formatting on text."""
    cleaned_text = text.replace("\n", " ").strip() # note: this might be different for each doc (best to experiment)

    # Other potential text formatting functions can go here
    return cleaned_text

# Open PDF and get lines/pages
# Note: this only focuses on text, rather than images/figures etc
def open_and_read_pdf(pdf_path: str) -> list[dict]:
    """
    Opens a PDF file, reads its text content page by page, and collects statistics.

    Parameters:
        pdf_path (str): The file path to the PDF document to be opened and read.

    Returns:
        list[dict]: A list of dictionaries, each containing the page number
        (adjusted), character count, word count, sentence count, token count, and the extracted text
        for each page.
    """
    doc = fitz.open(pdf_path)  # open a document
    pages_and_texts = []
    for page_number, page in tqdm(enumerate(doc)):  # iterate the document pages
        text = page.get_text()  # get plain text encoded as UTF-8
        text = text_formatter(text)
        pages_and_texts.append({"page_number": page_number - 41,  # adjust page numbers since our PDF starts on page 42
                                "page_char_count": len(text),
                                "page_word_count": len(text.split(" ")),
                                "page_sentence_count_raw": len(text.split(". ")),
                                "page_token_count": len(text) / 4,  # 1 token = ~4 chars, see: https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
                                "text": text})
    return pages_and_texts

pages_and_texts = open_and_read_pdf(pdf_path=pdf_path)
# print(pages_and_texts[:2])  # Display first two pages of text and stats
# print(random.sample(pages_and_texts, k=3))
df = pd.DataFrame(pages_and_texts)
# print(df.head(3))  # Display first three rows of the DataFrame

# 初始化 spaCy 的英文模型
nlp = English()
sentencizer = nlp.add_pipe("sentencizer")  # 添加句子切分组件

# 对 pages_and_texts 中的每一页进行句子划分
for item in tqdm(pages_and_texts, desc="Splitting sentences"):
    doc = nlp(item["text"])
    item["sentences"] = [sent.text for sent in doc.sents]
    # Make sure all sentences are strings
    item["sentences"] = [str(sentence) for sentence in item["sentences"]]
    
    # Count the sentences 
    item["page_sentence_count_spacy"] = len(item["sentences"])

# 将句子进一步划分为 chunks（例如每组 10 句）
def split_list(input_list: list, slice_size: int) -> list[list]:
    return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]

pages_and_chunks = []
for item in tqdm(pages_and_texts, desc="Creating chunks"):
    page_number = item["page_number"]
    sentences = item["sentences"]

    if len(sentences) > 0:
        chunks = split_list(sentences, slice_size=10)

        for chunk in chunks:
            chunk_dict = {
                "page_number": page_number,
                "chunk": " ".join(chunk),
                "chunk_char_count": sum(len(sentence) for sentence in chunk),
                "chunk_word_count": len(" ".join(chunk).split(" ")),
                "chunk_sentence_count_raw": len(chunk),
                "chunk_token_count": sum(len(sentence) for sentence in chunk) / 4  # 1 token ≈ 4 characters
            }
            pages_and_chunks.append(chunk_dict)
            
df_chunk = pd.DataFrame(pages_and_chunks)
print(df_chunk.head())
print(f"Total number of chunks: {len(pages_and_chunks)}")

# 加载预训练嵌入模型（推荐：all-mpnet-base-v2）
model = SentenceTransformer(r'D:\python代码调试\大模型实战\simple-local-rag\model\all-mpnet-base-v2',device = device)
# model = SentenceTransformer('all-mpnet-base-v2',device = device)
# 提取所有 chunk 文本用于嵌入
texts_for_embedding = [item["chunk"] for item in pages_and_chunks]

# 生成嵌入向量
embeddings = model.encode(texts_for_embedding, 
                          show_progress_bar=True, 
                          convert_to_tensor=True,
                          batch_size=32)

# 查看嵌入结果
print("Embeddings shape:", embeddings.shape)  # (num_chunks, 768)

text_chunks_and_embeddings_df = pd.DataFrame(pages_and_chunks)
text_chunks_and_embeddings_df['embedding'] = [embedding.tolist() for embedding in embeddings]
embeddings_df_save_path = "simple-local-rag/text_chunks_and_embeddings_df.csv"
text_chunks_and_embeddings_df.to_csv(embeddings_df_save_path, index=False)

print("✅ 嵌入向量与 chunk 数据已成功保存！")
