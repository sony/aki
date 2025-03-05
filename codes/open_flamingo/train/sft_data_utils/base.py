from .templates.templates import IGNORE_INDEX

import time
import traceback

import torch
import torchvision
from PIL import Image
from torch.utils.data.dataset import Dataset


class BaseDataset(Dataset):
    """Base dataset class

    Data loading process:
        (offline) init -> load_data -> (finalize_data)
        (online) __getitem__ -> process_data -> preprocess_data ->
            image_processor -> build_text_from_data -> tokenizer
    """
    def __init__(self, tokenizer, processor, max_length, **kwargs):
        super().__init__()

        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length
        self.templatizer = None

    def set_templatizer(self, templatizer):
        self.templatizer = templatizer

    def __len__(self):
        return len(self.dataset)

    def load_data(self):
        """Load data files and parse data samples with dataset-specific parsing logics

        The result instruction text should follow the shared format example:
            'system message'
            Human: 'prompt'
            Human: <image>
            AI: 'answer'

        Required keys in result dictionary:
            'image': pull path of an image file
            'task_type': used for selecting a processor
            NOTE templatizer parse 'examples' into 'text'; only one or the other is required.
            'text': parsed instruction text like above example
            'examples': a list of examples for template-based instruction generation

        Return:
            Parsed data list
        """
        raise NotImplementedError()

    def preprocess_data(self, data):
        """ perform pre-processing for the given data if required
        Args:
            data: datapoint given from self.dataset
        """
        return data

    def build_text_from_data(self, data):
        return data["text"]

    def encode_prompt(self, sample):
        text = self.tokenizer(
            sample,
            max_length=self.max_length,
            truncation=True,
            padding="longest",
            return_tensors="pt",
        )

        # Get the end token id
        end_token_id = self.tokenizer.added_tokens_encoder["<|assistant|>"]
        text["input_ids"] = text["input_ids"][0]
        text["attention_mask"] = text["attention_mask"][0]

        # if no answers after truncation, then will have labels with all IGNORE_INDEX
        try:
            split_index = (text["input_ids"] == end_token_id).nonzero(as_tuple=True)[0].item() + 1
        except:
            # just treat as no labels
            split_index = len(text["input_ids"])

        labels = text["input_ids"].clone()
        labels[:split_index] = IGNORE_INDEX

        assert text["input_ids"].shape == labels.shape

        return {
            "input_ids": text["input_ids"],
            "attention_mask": text["attention_mask"],
            "labels": labels
        }

    def process_data(self, data):
        data = self.preprocess_data(data)

        # Process Image if exists
        if "image" in data and len(data["image"]) > 0:
            image_urls = data["image"]
            if isinstance(image_urls, str):
                image_urls = [image_urls]

            images = [Image.open(image_path).convert("RGB") for image_path in image_urls]

            images = [self.processor(image) for image in images]
            images = torch.stack(images, dim=0)

            # do augmentation
            image_size = images.shape[-2]
            images = torchvision.transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0), interpolation=Image.BICUBIC)(images)
            images = torchvision.transforms.RandomHorizontalFlip(p=0.5)(images)
        else:
            images = None

        # Process Text
        text = self.build_text_from_data(data)
        text = self.encode_prompt(text)

        return {
            "image": images,
            "text": text,
            "task_type": data["task_type"],
        }

    def __getitem__(self, index):
        data = self.dataset[index]
        while True:
            try:
                data = self.process_data(data)

            except Exception as e:
                traceback.print_exc()
                print("===================================")
                print(f"Error in processing data: {data}")
                print(e)
                print("===================================")
                time.sleep(0.1)
                index = 0 if index == (len(self) - 1) else index + 1
                data = self.dataset[index]
                task_type = data.get("task_type", "dummy_default").split("_")[-1]
                continue
            break

        return data
