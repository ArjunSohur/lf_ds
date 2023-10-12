# library imports
from transformers import LongformerTokenizer
from torch.utils.data import Dataset
import nltk

# variables
tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")


def tokenize(text):
    inputs = tokenizer(
        text, return_tensors="pt", padding="max_length", truncation=True, max_length=512
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    return input_ids, attention_mask


class YelpData(Dataset):
    def __init__(self, train_data):
        self.train_data = train_data

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        text = self.train_data[idx]["text"]

        input_ids, attention_mask = tokenize(text)

        target = self.train_data[idx]["stars"]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target": target
        }
    
class sentence_YelpData(Dataset):
    def __init__(self, train_data):
        max_sentence_length = 0

        for i in range(len(train_data)):
            train_data[i]["text"] = nltk.sent_tokenize(train_data[i]["text"])
            max_sentence_length = max(max_sentence_length, len(train_data[i]["text"]))

        for i in range(len(train_data)):
            while len(train_data[i]["text"]) < max_sentence_length:
                train_data[i]["text"].append("")

        self.train_data = train_data

    def __len__(self):
        return len(self.train_data)
    
    def __getitem__(self, idx):
        text = self.train_data[idx]["text"]

        input_ids, attention_mask = tokenize(text)

        target = self.train_data[idx]["stars"]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target": target
        }
