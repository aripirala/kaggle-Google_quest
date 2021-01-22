import transformers
import tokenizers
import os

DEVICE = "gpu"
MAX_LEN = 512
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 8
LR = 3e-5
EPOCHS = 5
BERT_PATH = "bert-base-uncased"
MODEL_PATH = "../models/model.bin"
TRAINING_FILE = "../input/train.csv"
OPTIMIZER = transformers.AdamW
SCHEDULER = transformers.get_linear_schedule_with_warmup
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
BERT_MODEL = transformers.BertModel.from_pretrained(BERT_PATH, return_dict=False)
# TOKENIZER = tokenizers.BertWordPieceTokenizer(
#     os.path.join(BERT_PATH, 'vocab.txt'),
#     do_lower_case=True
# )