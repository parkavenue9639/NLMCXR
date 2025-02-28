from transformers import VisionEncoderDecoderModel


class ModelBuilder:
    def __init__(self, encoder, decoder, tokenizer, extractor):
        self.encoder = encoder
        self.decoder = decoder
        self.model = None
        self.tokenizer = tokenizer
        self.extractor = extractor
        self.build()

    def build(self):
        self.model = VisionEncoderDecoderModel.from_pretrained(self.encoder, self.decoder)
        self.set_model_parameters()
        self.set_output()
        return

    def set_tokenizer_para(self, flag):
        # GPT-2 原生不支持 pad_token，这里先用 unk_token 作为 pad_token，然后改为 eos_token。
        if flag:
            self.tokenizer.pad_token = self.tokenizer.unk_token
        else:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def set_model_parameters(self):
        self.set_tokenizer_para(True)
        # 更新model和config的eos_token(结束标记)
        self.model.eos_token_id = self.tokenizer.eos_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id

        self.model.decoder_start_token_id = self.tokenizer.bos_token_id
        self.model.config.decoder_start_token_id = self.tokenizer.bos_token_id

        self.model.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.model.max_length = 128
        self.model.config.max_length = 128
        self.model.config.decoder.max_length = 128

        self.model.min_length = 40
        self.model.config.min_length = 40
        self.model.config.decoder.min_length = 40

        self.model.no_repeat_ngram_size = 3
        self.model.config.no_repeat_ngram_size = 3
        self.model.config.decoder.no_repeat_ngram_size = 3
        self.set_tokenizer_para(False)

        # 更新model的config
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.model.config.decoder_start_token_id = self.tokenizer.bos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def set_output(self):
        output_dir = "../vit-gpt-model"
        self.model.save_pretrained(output_dir)
        self.extractor.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
