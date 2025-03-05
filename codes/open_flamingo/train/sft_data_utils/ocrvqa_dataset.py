import os
import logging
from pathlib import Path

from .base_task import BaseTaskDataset

# Set up basic configuration for logging
logger = logging.getLogger(__name__)


class OCRVQADataset(BaseTaskDataset):
    def __init__(
        self,
        tokenizer,
        processors,
        max_length,
        split="train",
        **kwargs,
    ):
        super().__init__(tokenizer, processors, max_length, **kwargs)

        splits = ["train", "val", "test"]
        split_idx = splits.index(split) + 1
        self.split = split
        self.split_index = split_idx
        
        self.dataset = self.load_data(kwargs["data_path"], kwargs["image_path"])

        logger.info(f"Load OCRVQA {split} dataset. len: {len(self.dataset)}")

    def load_data(self, annotation_path, image_root):
        raw_data = self.load(annotation_path)

        # flatten dataset & filter by split
        image_root = Path(image_root)
        data = []
        for key, dic in raw_data.items():
            if dic["split"] != self.split_index:
                continue

            img_url = dic["imageURL"]
            ext = os.path.splitext(img_url)[1]
            img_path = image_root / f"{key}{ext}"

            for q, a in zip(dic["questions"], dic["answers"]):
                data.append((img_path, {"question": q, "answer": a}))

        # build prompt data
        data = self.finalize_data(data, task_type="ocrvqa_vqa")

        return data
