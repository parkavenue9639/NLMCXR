from transformers import AutoTokenizer, ViTImageProcessor


class Extractor:
    def __init__(self, image_encoder_model):
        self.pretrained_extractor = image_encoder_model
        self.feature_extractor = ViTImageProcessor.from_pretrained(self.pretrained_extractor)
        pass


class Tokenizer:
    def __init__(self, text_decoder_model):
        self.pretrained_tokenizer = text_decoder_model
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_tokenizer)

    def set_tokenizer_para(self, flag):
        # GPT-2 原生不支持 pad_token，这里先用 unk_token 作为 pad_token，然后改为 eos_token。
        if flag:
            self.tokenizer.pad_token = self.tokenizer.unk_token
        else:
            self.tokenizer.pad_token = self.tokenizer.eos_token
