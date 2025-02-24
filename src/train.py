import torch
import os
import matplotlib

import pandas as pd
import matplotlib.pyplot as plt

from datasets import load_dataset
from tqdm import tqdm
from transformers import VisionEncoderDecoderModel
from ..model.model_builder import ModelBuilder
from ..model.extractor_and_tokenizer import Extractor, Tokenizer


def set_device():
    # choose device
    if torch.backends.mps.is_available() and os.name == 'posix' and 'darwin' in os.sys.platform:
        device = torch.device('mps')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device


def load_data():
    # 直接加载 NLMCXR 数据集
    # 默认保存路径：~/.cache/huggingface/datasets
    dataset = load_dataset('Fakhraddin/NLMCXR', split='train', cache_dir='../data')  # 或者根据需要选择 'train'/'test' 等
    # 初始化空的 DataFrame
    data = pd.DataFrame(columns=['path', 'text'])

    # 如果字段不同，可以根据实际字段名称进行调整
    for item in tqdm(dataset):
        image_path = item['path']  # 从数据集中提取图像路径
        caption = item['text']  # 从数据集中提取报告文本

        # 过滤掉不需要的报告（例如侧位X光）
        if '2001' in image_path:
            continue

        # 将 .jpg 替换为 .png，如果需要
        image_path = image_path.replace('.jpg', '.png')

        # 数据清理（如 caption 长度控制）
        if len(caption) < 400:
            # 添加新的一行数据
            new_data = pd.DataFrame([{'path': 'NLMCXR_png/' + image_path, 'text': caption}])
            data = pd.concat([data, new_data], ignore_index=True)

    # 删除重复的 caption
    data = data.drop_duplicates('text').reset_index(drop=True)

    # 打印或返回处理后的数据
    print(data.head())
    return data


def word_count(str):
    words = str.split()
    return len(words)


def check_text(data):
    matplotlib.use('TkAgg')
    plt.ion()  # 启用交互模式
    data['text'].apply(word_count).hist()
    plt.savefig('word_count_histogram.png')


def set_tokenizer_para(tokenizer, flag):
    # GPT-2 原生不支持 pad_token，这里先用 unk_token 作为 pad_token，然后改为 eos_token。
    if flag:
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.eos_token


def set_model_para(model, tokenizer):
    set_tokenizer_para(tokenizer, True)
    # 更新model和config的eos_token(结束标记)
    model.eos_token_id = tokenizer.eos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    model.decoder_start_token_id = tokenizer.bos_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id

    model.pad_token_id = tokenizer.pad_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.max_length = 128
    model.config.max_length = 128
    model.config.decoder.max_length = 128

    model.min_length = 40
    model.config.min_length = 40
    model.config.decoder.min_length = 40

    model.no_repeat_ngram_size = 3
    model.config.no_repeat_ngram_size = 3
    model.config.decoder.no_repeat_ngram_size = 3
    set_tokenizer_para(tokenizer, False)

    # 更新model的config
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

def set_output(model, feature_extractor, tokenizer):
    output_dir = "../vit-gpt-model"
    model.save_pretrained(output_dir)
    feature_extractor.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == '__main__':
    data = load_data()
    check_text(data)
    image_encoder_model = 'google/vit-base-patch16-224-in21k'
    text_decoder_model = 'gpt2'
    model = ModelBuilder(image_encoder_model, text_decoder_model).build()
    feature_extractor = Extractor(image_encoder_model).set_extractor()
    tokenizer = Tokenizer(text_decoder_model).set_tokenizer()
    set_model_para(model, tokenizer)
    set_output(model, feature_extractor, tokenizer)
