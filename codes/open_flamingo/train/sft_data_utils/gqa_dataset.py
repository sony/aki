import logging
from pathlib import Path

from .base_task import BaseTaskDataset

# Set up basic configuration for logging
logger = logging.getLogger(__name__)


class GQADataset(BaseTaskDataset):
    """GQA dataset
    """
    def __init__(
        self,
        tokenizer,
        processors,
        max_length,
        split="train",
        balanced=True,
        **kwargs,
    ):
        super().__init__(tokenizer, processors, max_length, **kwargs)

        assert split in ["train", "val", "test"]
        self.split = split

        self.dataset = self.load_data(kwargs["data_path"], split, kwargs["image_path"], balanced)

        logger.info(f"Load GQA {split} dataset. len: {len(self.dataset)}")

    def load_data(self, annotation_path, split, image_root, balanced):
        if not balanced:
            raise NotImplementedError("GQA only supports balanced annotations (1M) for now.")

        js = self.load(annotation_path)

        data = []
        for dic in js.values():
            img_id = dic["imageId"]
            img_path = f"{image_root}{img_id}.jpg"

            # example)
            # Q: Is the sky dark?
            # A: yes
            # FullA: Yes, the sky is dark.
            question = dic["question"]
            answer = dic["answer"]
            full_answer = dic["fullAnswer"]

            data.append((
                str(img_path),
                {
                    "question": question,
                    "answer": answer,
                    "full_answer": full_answer,
                }
            ))

        data = self.finalize_data(data, task_type="gqa_vqa")

        return data
