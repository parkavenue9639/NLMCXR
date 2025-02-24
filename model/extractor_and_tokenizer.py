from transformers import AutoTokenizer, ViTImageProcessor


class Extractor:
    def __init__(self, image_encoder_model):
        self.pretrained_extractor = image_encoder_model
        pass

    def set_extractor(self):
        feature_extractor = ViTImageProcessor.from_pretrained(self.pretrained_extractor)
        return feature_extractor


class Tokenizer:
    def __init__(self, text_decoder_model):
        self.pretrained_tokenizer = text_decoder_model
        pass

    def set_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.pretrained_tokenizer)
        return tokenizer
