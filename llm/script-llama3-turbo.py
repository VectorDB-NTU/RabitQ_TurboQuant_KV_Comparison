import os

PATH = '/scratch/md5367/huggingface/dataset/'
os.environ['TRANSFORMERS_CACHE'] = PATH
os.environ['HF_HOME'] = PATH
os.environ['HF_DATASETS_CACHE'] = PATH
os.environ['TORCH_HOME'] = PATH
os.environ['XDG_CACHE_HOME'] = PATH
os.environ['TRANSFORMERS_CACHE'] = PATH

import argparse

parser = argparse.ArgumentParser(description="Write dataset name to a file.")
parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset to write to the file")
parser.add_argument("--key_quantization_bits", type=int, required=True)
parser.add_argument("--key_quantization_bits_initial_layers", type=int, required=True)
parser.add_argument("--initial_layers_count", type=int, required=True)
parser.add_argument("--outlier_count_general", type=int, required=True)

args = parser.parse_args()
print(
    f"The dataset is: {args.dataset}, {args.key_quantization_bits, args.key_quantization_bits_initial_layers, args.initial_layers_count}")

from metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

dataset2prompt = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "multifieldqa_zh": "阅读以下文字并用中文简短回答：\n\n{context}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{input}\n回答：",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "dureader": "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。\n\n问题：{input}\n回答：",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
    "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "vcsum": "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议记录：\n{context}\n\n会议总结：",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "lsht": "请判断给定新闻的类别，下面是一些例子。\n\n{context}\n{input}",
    "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
    "passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: ",
    "passage_retrieval_zh": "以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n{context}\n\n下面是一个摘要\n\n{input}\n\n请输入摘要所属段落的编号。答案格式必须是\"段落1\"，\"段落2\"等格式\n\n答案是：",
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n"
}
dataset2maxlen = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "multifieldqa_zh": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "dureader": 128,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "vcsum": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "lsht": 64,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "passage_retrieval_zh": 32,
    "lcc": 64,
    "repobench-p": 64
}

import numpy as np
import random


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def truncate_input(input: list, max_length: int, manner="middle"):
    if max_length < 0:
        return input
    if len(input) <= max_length:
        return input
    if manner == "middle":
        split = max_length // 2
        return input[0:split] + input[-split:]
    else:
        return None


def truncate_by_tokens(input, tok, max_tokens, manner: str = "middle"):
    tokens = tok.encode(input)
    len_before = len(tokens)
    tokens = truncate_input(tokens, max_length=max_tokens, manner=manner)
    len_after = len(tokens)
    assert len_after <= len_before
    assert len_after <= max_tokens or max_tokens < 0
    return tokens


# from models.llama3_utils_qjl import QJLSketch, QJLKeyQuantizer
# from models.llama3_utils_rqjl_outlier import RQJLSketch, RQJLKeyQuantizer
from models.llama3_utils_turbo_int_outlier import RQJLSketch, RQJLKeyQuantizer

from models.llama3_rqjl_outlier import LlamaForCausalLM_QJL
import time
import numpy as np
import torch
from transformers import LlamaForCausalLM, LlamaConfig, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from fastchat.model import get_conversation_template

seed_everything(42)

model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # change other models if you want
dtype = torch.bfloat16
device = 'cuda'
config = LlamaConfig.from_pretrained(model_name)
config._flash_attn_2_enabled = True
use_qjl = True
tic = time.time()
config.attention_dropout = 0.0
config.use_flash = True
config = LlamaConfig.from_pretrained(model_name)
config._flash_attn_2_enabled = True
config._attn_implementation == "flash_attention_2"
config.use_cache = True

config.key_quantization_bits = args.key_quantization_bits
config.key_quantization_bits_initial_layers = args.key_quantization_bits_initial_layers
config.initial_layers_count = args.initial_layers_count

config.outlier_count_general = args.outlier_count_general
config.outlier_count_initial_layers = args.outlier_count_general

config.value_quantization_bits = 2
config.group_size = 32
config.buffer_size = 128

generator = torch.Generator(device=torch.device(device))
config.qjl_outlier = RQJLSketch(dimension=32, bit_width=args.key_quantization_bits // 128, rng=generator)
config.qjl_residual = RQJLSketch(dimension=96, bit_width=args.key_quantization_bits_initial_layers // 128,
                                 rng=generator)

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = LlamaForCausalLM_QJL.from_pretrained(
    pretrained_model_name_or_path=model_name,
    config=config,
    torch_dtype=dtype,
    device_map="auto",
)
tim_load = time.time() - tic
print(f"model loaded ... {tim_load:.4f} sec")

import os

PATH = '/scratch/md5367/huggingface/dataset/'
os.environ['TRANSFORMERS_CACHE'] = PATH
os.environ['HF_HOME'] = PATH
os.environ['HF_DATASETS_CACHE'] = PATH
os.environ['TORCH_HOME'] = PATH
os.environ['XDG_CACHE_HOME'] = PATH

dataset = args.dataset
data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test')
prompt_format = dataset2prompt[dataset]

total_score = 0.
n_data = len(data)
max_input_length = 100_000
maxlen = dataset2maxlen[dataset]

aa = []
start = time.time()
for i in range(n_data):

    json_obj = data[i]
    prompt = prompt_format.format(**json_obj)

    input_tokens = truncate_by_tokens(prompt, tokenizer, max_input_length)
    input_tensors = {"input_ids": torch.tensor(input_tokens).unsqueeze(0).to(model.device)}
    seq_len = len(input_tokens)
    terminators = [tokenizer.eos_token_id]

    outputs = model.generate(**input_tensors, max_new_tokens=maxlen, eos_token_id=terminators, do_sample=False,
                             temperature=None, top_p=None, use_cache=True, pad_token_id=tokenizer.pad_token_id,
                             return_dict_in_generate=True)
    output = outputs.sequences[0, seq_len:]
    output_token_len = len(output)
    output = tokenizer.decode(output, skip_special_tokens=True)
    pred = output.strip()
    pred = pred.lstrip('\n').split('\n')[0]
    pred = pred.split('  ')[0]

    ground_truths = json_obj['answers']
    all_classes = json_obj['all_classes']
    prediction = pred

    if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
        prediction = prediction.lstrip('\n').split('\n')[0]

    score = 0.
    for ground_truth in ground_truths:
        score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))

    total_score += score

    mem_alloc = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
    mem_reserve = torch.cuda.memory_reserved() / 1024 / 1024 / 1024
    mem_peak = torch.cuda.memory_stats()['active_bytes.all.peak'] / 1024 / 1024 / 1024

    mem_info = f"mem_alloc: {mem_alloc:.5f}, mem_reserved: {mem_reserve:.5f}, mem_peak: {mem_peak:.5f}"
    aa.append(score)
    print(f"[{i:>4}] score: {score:.4f}, avg_score: {total_score / (i + 1):.4f}, | {mem_info}")

memory = ((32) * config.key_quantization_bits + (96) * config.key_quantization_bits_initial_layers) / (
            128 * 128) + 1.0 * (config.outlier_count_general > 0)
with open('result_llama3.txt', 'a') as file:
    file.write(f"qjl, dataset {dataset}, memory {memory} bits, avg score: {np.mean(aa)}\n")