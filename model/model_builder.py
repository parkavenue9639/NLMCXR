import torch
import os
from transformers import VisionEncoderDecoderModel


class ModelBuilder:
    def __init__(self, encoder, decoder, tokenizer, extractor):
        self.encoder = encoder
        self.decoder = decoder
        self.device = None
        self.model = None
        self.set_device()
        self.build(extractor, tokenizer)

    def set_device(self):
        # choose device
        if torch.backends.mps.is_available() and os.name == 'posix' and 'darwin' in os.sys.platform:
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')

    def build(self, extractor, tokenizer):
        self.model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(self.encoder, self.decoder, timeout=60)
        self.model = self.model.to(self.device)
        self.set_model_parameters(tokenizer)
        self.set_output(extractor, tokenizer)
        return

    def set_tokenizer_para(self, tokenizer, flag):
        # GPT-2 原生不支持 pad_token，这里先用 unk_token 作为 pad_token，然后改为 eos_token。
        if flag:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"Tokenizer pad_token: {tokenizer.pad_token}, pad_token_id: {tokenizer.pad_token_id}")
        print(f"Model config pad_token_id: {self.model.config.pad_token_id}")

    def set_model_parameters(self, tokenizer):
        self.set_tokenizer_para(tokenizer, True)
        # 更新model和config的eos_token(结束标记)
        self.model.eos_token_id = tokenizer.eos_token_id
        self.model.config.eos_token_id = tokenizer.eos_token_id

        self.model.decoder_start_token_id = tokenizer.bos_token_id
        self.model.config.decoder_start_token_id = tokenizer.bos_token_id

        self.model.pad_token_id = tokenizer.pad_token_id
        self.model.config.pad_token_id = tokenizer.pad_token_id

        self.model.max_length = 128
        self.model.config.max_length = 128
        self.model.config.decoder.max_length = 128

        self.model.min_length = 40
        self.model.config.min_length = 40
        self.model.config.decoder.min_length = 40

        self.model.no_repeat_ngram_size = 3
        self.model.config.no_repeat_ngram_size = 3
        self.model.config.decoder.no_repeat_ngram_size = 3
        self.set_tokenizer_para(tokenizer, False)

        # 更新model的config
        self.model.config.eos_token_id = tokenizer.eos_token_id
        self.model.config.decoder_start_token_id = tokenizer.bos_token_id
        self.model.config.pad_token_id = tokenizer.pad_token_id

    def set_output(self, extractor, tokenizer):
        output_dir = "../vit-gpt-model"
        self.model.save_pretrained(output_dir)
        extractor.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
