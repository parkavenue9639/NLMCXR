import torch.mps
import os

from model.score import Score
from transformers import default_data_collator, Seq2SeqTrainer, Seq2SeqTrainingArguments


class Trainer:
    def __init__(self, model, feature_extractor, tokenizer, processed_dataset, metric_name):
        self.model = model
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.processed_dataset = processed_dataset
        self.metric_name = metric_name
        self.metrics = {"BLEU": Score(tokenizer).compute_BLEU_score, "SPICE": Score(tokenizer).compute_SPICE_score,
                        "CIDER": Score(tokenizer).compute_CIDER_score, "METEOR": Score(tokenizer).compute_METEOR_score,
                        "ROUGE": Score(tokenizer).compute_ROUGE_score}
        self.training_args = None
        self.trainer = None
        self.set_training_args()
        self.set_trainer()

    def set_training_args(self):
        self.training_args = Seq2SeqTrainingArguments(
            predict_with_generate=True,
            evaluation_strategy="epoch",  # 每个epoch评估一次
            save_strategy="epoch",  # 每个epoch保存一次检查点
            save_total_limit=3,  # 保留最近三个检查点
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            output_dir=f"./image-captioning-output-{self.metric_name}",
            num_train_epochs=10
        )
        return

    def set_trainer(self):
        self.trainer = Seq2SeqTrainer(
                        model=self.model,
                        tokenizer=self.feature_extractor,
                        args=self.training_args,
                        compute_metrics=self.metrics[self.metric_name],
                        train_dataset=self.processed_dataset['train'],
                        eval_dataset=self.processed_dataset['validation'],
                        data_collator=default_data_collator,
                        )

    def train(self):
        checkpoint_dir = f"./image-captioning-output-{self.metric_name}"
        print("Checkpoint directory:", checkpoint_dir)
        print("Directory contents:", os.listdir(checkpoint_dir))
        self.trainer.train(resume_from_checkpoint=True)
        torch.mps.empty_cache()

    def save_model(self):
        self.trainer.save_model(f"./image-captioning-output-{self.metric_name}")
        self.tokenizer.save_pretrained(f"./image-captioning-output-{self.metric_name}")