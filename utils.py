import os
import re
from bs4 import BeautifulSoup
import hashlib
import openai
import tiktoken
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity


def filter_markdown_heading2(text: str) -> str:
    text = re.sub(r"^###", "#", text)
    text = re.sub(r'##\s商品图片\n.+?(?=##|\Z)', '', text, flags=re.DOTALL)
    text = re.sub(r'##\s产品服务\n.+?(?=##|\Z)', '', text, flags=re.DOTALL)
    return text

def preprocess_markdown_file(file_path: str, export_raw: bool=False) -> str:
    with open(file_path, 'r') as f:
        text = f.read()
        
    if export_raw:
        return text

    return BeautifulSoup(text, 'html.parser').get_text()
    
def num_tokens_from_string(string: str, model_name: str='gpt-3.5-turbo') -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def generate_checksum(s: str) -> str:
    """生成字符串s的SHA-256哈希值"""
    h = hashlib.sha256(s.encode('utf-8'))
    return h.hexdigest()

def semantic_search(target, labels, top_n=3) -> list:
    similarity = cosine_similarity(np.array([target]).reshape(1, -1), np.array(labels).reshape(len(labels), -1))
    sorted_indexes = sorted(enumerate(similarity[0]), key=lambda x: x[1], reverse=True)
    filtered_sorted_indexes = [x for x in sorted_indexes if x[1] > 0.9]
    
    return filtered_sorted_indexes[:top_n] if len(filtered_sorted_indexes) > 0 else sorted_indexes[:top_n]

    