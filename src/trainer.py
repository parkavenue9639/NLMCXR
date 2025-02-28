import torch.mps

from model.score import Score
from transformers import default_data_collator, Seq2SeqTrainer, Seq2SeqTrainingArguments


class Trainer:
    def __init__(self, model, feature_extractor, tokenizer, processed_dataset, metric_name):
        self.model = model
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.processed_dataset = processed_dataset
        self.metric_name = metric_name
        self.metrics = {"BLEU": Score.compute_BLEU_score, "SPICE": Score.compute_SPICE_score,
                       "CIDER": Score.compute_CIDER_score, "METEOR": Score.compute_METEOR_score,
                       "ROUGE": Score.compute_ROUGE_score}
        self.training_args = None
        self.trainer = None

    def set_training_args(self):
        self.training_args = Seq2SeqTrainingArguments(
            predict_with_generate=True,
            evaluation_strategy="epoch",
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            output_dir=f"./image-captioning-output-{self.metric_name}",
            num_train_epochs = 10
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
        self.trainer.train()
        torch.mps.empty_cache()

    def save_model(self):
        self.trainer.save_model(f"./image-captioning-output-{self.metric_name}")
        self.tokenizer.save_pretrained(f"./image-captioning-output-{self.metric_name}")