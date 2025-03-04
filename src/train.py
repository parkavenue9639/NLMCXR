import torch
import os
import matplotlib

import matplotlib.pyplot as plt

from trainer import Trainer
from model.model_builder import ModelBuilder
from model.extractor_and_tokenizer import Extractor, Tokenizer
from model.data_set import DataProcess


def word_count(str):
    words = str.split()
    return len(words)


def check_text(data):
    matplotlib.use('TkAgg')
    plt.ion()  # 启用交互模式
    data['text'].apply(word_count).hist()
    plt.savefig('word_count_histogram.png')


if __name__ == '__main__':
    image_encoder_model = 'google/vit-base-patch16-224-in21k'
    text_decoder_model = 'gpt2'

    feature_extractor = Extractor(image_encoder_model).feature_extractor
    tokenizer = Tokenizer(text_decoder_model).tokenizer
    model = ModelBuilder(image_encoder_model, text_decoder_model, tokenizer, feature_extractor).model

    data_process = DataProcess(tokenizer, feature_extractor)
    check_text(data_process.data)

    # model = ModelBuilder(image_encoder_model, text_decoder_model, tokenizer, feature_extractor).model

    trainer = Trainer(model, feature_extractor, tokenizer, data_process.processed_dataset, "BLEU")
    trainer.train()
    trainer.save_model()


