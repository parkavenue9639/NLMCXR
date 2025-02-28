import torch
import os
import matplotlib

import matplotlib.pyplot as plt

from trainer import Trainer
from model.model_builder import ModelBuilder
from model.extractor_and_tokenizer import Extractor, Tokenizer
from model.data_set import DataProcess


def set_device():
    # choose device
    if torch.backends.mps.is_available() and os.name == 'posix' and 'darwin' in os.sys.platform:
        device = torch.device('mps')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device


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
    image_encoder_model = 'google/vit-base-patch16-224-in21k'
    text_decoder_model = 'gpt2'

    feature_extractor = Extractor(image_encoder_model).set_extractor()
    tokenizer = Tokenizer(text_decoder_model).set_tokenizer()

    data_process = DataProcess(tokenizer, feature_extractor)
    check_text(data_process.data)

    model = ModelBuilder(image_encoder_model, text_decoder_model, tokenizer, feature_extractor).model

    trainer = Trainer(model, feature_extractor, tokenizer, data_process.processed_dataset, "BLEU")
    trainer.train()
    trainer.save_model()


