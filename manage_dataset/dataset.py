# library imports
from transformers import LongformerTokenizer
from torch.utils.data import Dataset

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
            "target": target,
        }
