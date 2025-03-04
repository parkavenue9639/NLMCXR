import pandas as pd
import numpy as np
from tqdm import tqdm

from PIL import Image
from datasets import load_dataset
from datasets import Dataset, DatasetDict


class DataProcess:
    def __init__(self, tokenizer, feature_extractor):
        self.data = None
        self.sum_data = None
        self.image_obj = None
        self.train_dataset = None
        self.validation_dataset = None
        self.test_dataset = None
        self.ds = DatasetDict()
        self.processed_images = []
        self.processed_dataset = None
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.process_data()

    def process_data(self):
        self.load_data()
        self.process_train_dataset()
        self.process_validation_dataset()
        self.process_test_dataset()
        self.process_ds()
        self.processed_dataset = self.ds.map(
            function=self.preprocess_fn,  # 要应用的处理函数
            batched=True,  # 一次处理多个样本
            fn_kwargs={'max_target_length': 128},  # 将max_target_length=128传递给处理函数
            remove_columns=self.ds['train'].column_names  # 预处理后，移除数据集中的原始列，只保留处理后的结果
        )

    def load_data(self):
        # 直接加载 NLMCXR 数据集
        # 默认保存路径：~/.cache/huggingface/datasets
        dataset = load_dataset('Fakhraddin/NLMCXR', split='train', cache_dir='../data')  # 或者根据需要选择 'train'/'test' 等
        #  print(type(dataset[0]['image']))  <class 'PIL.PngImagePlugin.PngImageFile'>
        # 初始化空的 DataFrame
        data = pd.DataFrame(columns=['path', 'text', 'image'])

        # 如果字段不同，可以根据实际字段名称进行调整
        for item in tqdm(dataset):
            image_path = item['path']  # 从数据集中提取图像路径
            caption = item['text']  # 从数据集中提取报告文本
            image_obj = item['image']  # 从数据集中提取图片对象

            # 过滤掉不需要的报告（例如侧位X光）
            if '2001' in image_path:
                continue

            # 将 .jpg 替换为 .png，如果需要
            image_path = image_path.replace('.jpg', '.png')

            # 数据清理（如 caption 长度控制）
            if len(caption) < 400:
                # 添加新的一行数据
                new_data = pd.DataFrame([{'path': image_path, 'text': caption, 'image': image_obj}])
                data = pd.concat([data, new_data], ignore_index=True)

        # 删除重复的 caption,同步删除对应的image对象
        self.data = data.drop_duplicates('text').reset_index(drop=True)
        print('shape of data: {}'.format(self.data.shape[0]))
        self.processed_images = self.data['image'].tolist()
        print('shape of processed_images: {}'.format(len(self.processed_images)))
        self.sum_data = self.data.copy()
        print('shape of sum_data: {}'.format(self.sum_data.shape[0]))
        self.data = self.data.drop(columns=['image'])

        # 打印或返回处理后的数据
        print(self.data.head())
        print(self.sum_data.head())
        return

    def process_train_dataset(self):
        self.train_dataset = Dataset.from_pandas(self.data[:654])
        return

    def process_validation_dataset(self):
        self.validation_dataset = Dataset.from_pandas(self.data[654:1308])
        return

    def process_test_dataset(self):
        self.test_dataset = Dataset.from_pandas(self.data[1308:])
        return

    def process_ds(self):
        self.ds['train'] = self.train_dataset
        self.ds['validation'] = self.validation_dataset
        self.ds['test'] = self.test_dataset
        return

    def tokenization_fn(self, captions, max_target_length):
        # 对输入文本captions进行标记化
        # 将所有文本填充到max_length，指定文本的最大长度为max_target_length，返回的结果是pytorch张量
        labels = self.tokenizer(captions,
                                padding="max_length",
                                max_length=max_target_length, return_tensors='pt').input_ids
        return labels

    def feature_extraction_fn(self, images, check_image=True):
        """
        处理图像特征提取，兼容文件路径和 PIL.Image 对象。
        """
        processed_images = []

        for image in images:
            if isinstance(image, str):  # 如果是路径，尝试打开
                try:
                    print(f"Processing image: {image}")
                    img = Image.open(image)
                    processed_images.append(img)
                except Exception as e:
                    print(f"Error opening image {image}: {e}")
            elif isinstance(image, Image.Image):  # 如果是 PIL.Image，直接使用
                processed_images.append(image)
            else:
                print(f"Skipping invalid image format: {type(image)}")

        if not processed_images:
            raise ValueError("No valid images provided for feature extraction.")
        else:
            print("length of processed_images: {}".format(len(processed_images)))

        encoder_inputs = self.feature_extractor(images=processed_images, return_tensors="pt")
        return encoder_inputs.pixel_values

    def preprocess_fn(self, examples, max_target_length, check_image=True):
        # 将文本标记化和图像特征提取合并到一起，返回一个包含两部分的字典。
        image_paths = examples['path']
        captions = examples['text']

        # 获取当前 batch 对应的 image 对象
        batch_images = []
        for path in image_paths:
            # 通过路径查找对应的 image（因为 self.data 仍然保留了 image）
            image_obj = self.sum_data.loc[self.sum_data['path'] == path, 'image'].values
            if len(image_obj) > 0:
                batch_images.append(image_obj[0])  # 取出 PIL.Image
            else:
                print("invalid image_obj: {}".format(image_obj))
        print("Batch images: {}".format(len(batch_images)))
        # label:文本的标记化处理， pixel_values： 图像的特征
        model_inputs = {'labels': self.tokenization_fn(captions, max_target_length),
                        'pixel_values': self.feature_extraction_fn(batch_images, check_image=check_image)}
        # This contains image path column
        return model_inputs
