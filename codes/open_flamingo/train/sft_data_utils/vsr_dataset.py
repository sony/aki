import os
import logging

from .base_task import BaseTaskDataset

# Set up basic configuration for logging
logger = logging.getLogger(__name__)


class VSRDataset(BaseTaskDataset):
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
        assert split in splits
        self.split = split

        # have two paths because VSR uses images from both train and val 2017 for training data
        self.dataset = self.load_data(kwargs["data_path"], kwargs["image_path_train"], kwargs["image_path_val"])

        logger.info(f"Load VSR {split} dataset. len: {len(self.dataset)}")

    def load_data(self, annotation_path, image_root_train, image_root_val):
        raw_data = self.load(annotation_path)

        # flatten dataset
        data = []
        for dic in raw_data:
            # split and img name, e.g., train2017/000000296471.jpg"
            folder_name = dic["image_link"].split("/")[-2:-1][0]
            filename = dic["image_link"].split("/")[-1:][0]
            if folder_name == "train2017":
                img_path = os.path.join(image_root_train, filename)
            elif folder_name == "val2017":
                img_path = os.path.join(image_root_val, filename)
            else:
                raise ValueError(f"Invalid folder name {folder_name}")

            question = dic["caption"]
            question_interro = dic["caption"].split("is")
            question_interro = [str_.lower().replace(".", "?").strip() for str_ in question_interro]
            question_interro = "Is " + " ".join(question_interro)

            answer = "yes" if dic["label"] == 1 else "no"

            data.append(
                (
                    str(img_path),
                    {"question": question, "question_interro": question_interro, "answer": answer},
                )
            )

        # build prompt data
        data = self.finalize_data(data, task_type="vsr_vqa")
        return data
