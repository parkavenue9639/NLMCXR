import nltk
import evaluate
import numpy as np
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice


class Score:
    def __init__(self):

        nltk.download("punkt")  # 下载punkt数据包，这是NLTK库的一个分词工具，专门用于句子分割和标点符号处理
        self.ignore_pad_token_for_loss = True  # 计算损失时是否忽略标记
        pass

    def postprocess_text(self, preds, labels):
        preds = [pred.strip() for pred in preds]  # Get prediction tokens
        labels = [label.strip() for label in labels]  # Get label tokens

        # 确保预测结果和标签只包含有用的文本，不受空字符的影响
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_BLEU_score(self, tokenizer, eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if self.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = self.postprocess_text(decoded_preds,
                                                         decoded_labels)

        # BLEU metric
        bleu_metric = evaluate.load("bleu")
        result = bleu_metric.compute(predictions=decoded_preds,
                                     references=decoded_labels)

        # Average precisions
        result["precisions"] = sum(result["precisions"]) / len(result["precisions"])

        # Finalize predictions
        prediction_lens = [
            np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
        ]
        result["gen_len"] = np.mean(prediction_lens)

        return result

    def compute_ROUGE_score(self, eval_preds, tokenizer):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if self.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = self.postprocess_text(decoded_preds,
                                                         decoded_labels)

        # ROUGE metric
        rouge_metric = evaluate.load("rouge")
        result = rouge_metric.compute(predictions=decoded_preds,
                                      references=decoded_labels)

        # Average precisions
        result["precisions"] = sum(result["precisions"]) / len(result["precisions"])

        # Finalize predictions
        prediction_lens = [
            np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
        ]
        result["gen_len"] = np.mean(prediction_lens)

        return result

    def compute_METEOR_score(self, eval_preds, tokenizer):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if self.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = self.postprocess_text(decoded_preds,
                                                         decoded_labels)

        # METEOR metric
        meteor_metric = evaluate.load("meteor")
        result = meteor_metric.compute(predictions=decoded_preds,
                                       references=decoded_labels)

        # Average precisions
        result["precisions"] = sum(result["precisions"]) / len(result["precisions"])

        # Finalize predictions
        prediction_lens = [
            np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
        ]
        result["gen_len"] = np.mean(prediction_lens)

        return result

    def compute_CIDER_score(self, eval_preds, tokenizer):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if self.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = self.postprocess_text(decoded_preds,
                                                         decoded_labels)

        # Create CIDEr metric object
        cider_metric = Cider()

        # Format decoded to satisfy pycocoevalcap
        dict_decoded_preds = {}
        dict_decoded_labels = {}
        for i in range(len(decoded_preds)):
            dict_decoded_preds[i] = [decoded_preds[i]]
            dict_decoded_labels[i] = [decoded_labels[i]]
        decoded_preds = dict_decoded_preds
        decoded_labels = dict_decoded_labels

        # Store results into metric; not too reliable but works
        result = cider_metric.compute_score(decoded_preds, decoded_labels)
        result = {"CIDEr": result[0], "Scores": result[1]}

        prediction_lens = [
            np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
        ]
        result["gen_len"] = np.mean(prediction_lens)
        return result

    def compute_SPICE_score(self, eval_preds, tokenizer):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if self.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = self.postprocess_text(decoded_preds,
                                                         decoded_labels)

        # Create SPICE metric object
        spice_metric = Spice()

        # Format decoded to satisfy pycocoevalcap
        dict_decoded_preds = {}
        dict_decoded_labels = {}
        for i in range(len(decoded_preds)):
            dict_decoded_preds[i] = [decoded_preds[i]]
            dict_decoded_labels[i] = [decoded_labels[i]]
        decoded_preds = dict_decoded_preds
        decoded_labels = dict_decoded_labels

        # Store results into metric; not too reliable but works
        result = spice_metric.compute_score(decoded_preds, decoded_labels)
        result = {"SPICE": result[0], "Scores": result[1]}

        prediction_lens = [
            np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
        ]

        result["gen_len"] = np.mean(prediction_lens)
        return result