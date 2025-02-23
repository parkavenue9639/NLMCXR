import torch
import os
import matplotlib

import pandas as pd
import matplotlib.pyplot as plt

from datasets import load_dataset
from tqdm import tqdm
from transformers import VisionEncoderDecoderModel, AutoTokenizer, ViTImageProcessor



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


def set_model(image_encoder_model, text_decoder_model):
    model = VisionEncoderDecoderModel.from_pretrained(image_encoder_model, text_decoder_model)
    return model


def set_extractor_and_tokenizer(image_encoder_model, text_decoder_model):
    # 特征提取器
    feature_extractor = ViTImageProcessor.from_pretrained(image_encoder_model)
    # 分词器
    tokenizer = AutoTokenizer.from_pretrained(text_decoder_model)
    return feature_extractor, tokenizer



data = load_data()
check_text(data)
image_encoder_model = 'google/vit-base-patch16-224-in21k'
text_decoder_model = 'gpt2'
model = set_model(image_encoder_model, text_decoder_model)
feature_extractor, tokenizer = set_extractor_and_tokenizer(image_encoder_model, text_decoder_model)
