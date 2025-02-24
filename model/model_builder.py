from transformers import VisionEncoderDecoderModel


class ModelBuilder:
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder

    def build(self):
        model = VisionEncoderDecoderModel.from_pretrained(self.encoder, self.decoder)
        return model
