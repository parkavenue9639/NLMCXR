import json
import logging
import random
import matplotlib.pyplot as plt
from transformers import pipeline
from PIL import Image
from matplotlib import font_manager as fm


class image_captioner:
    def __init__(self, metric_name):
        self.metric_name = metric_name
        self.json_filename = None
        self.captioner = None
        self.set_json_filename()
        self.set_captioner()
        self.set_log()

    def set_json_filename(self):
        self.json_filename = '../vit-gpt-model/preprocessor_config.json'
        return

    def set_captioner(self):
        with open(self.json_filename) as json_file:
            json_decoded = json.load(json_file)

        if 'image_processor_type' in json_decoded:
            json_decoded['feature_extractor_type'] = json_decoded.pop('image_processor_type')

        with open(self.json_filename, 'w') as json_file:
            json.dump(json_decoded, json_file, indent=2, separators=(',', ': '))

        self.captioner = pipeline("image-to-text", model='../vit-gpt-model')

    def set_log(self):
        logging.basicConfig(level=logging.ERROR)

    def demo(self, data):
        print(data.shape)
        row_number = random.randint(1308, len(data) - 1)  # 随机选取数据集中一行
        # image = Image.open(data.iloc[row_number].image_path)
        image = data.iloc[row_number].image
        original = data.iloc[row_number].caption
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(image)
        generated = self.captioner(image)[0]['generated_text']
        fontprop = fm.FontProperties(fname='Roboto-Regular.ttf', size=12)

        #     ax.text(8, 1, impression, fontproperties=fontprop)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        return original, generated
