import os
import logging

from .base_task import BaseTaskDataset, optionize

# Set up basic configuration for logging
logger = logging.getLogger(__name__)


class AOKVQADataset(BaseTaskDataset):
    """A-OKVQA dataset
    """
    def __init__(
        self,
        tokenizer,
        processor,
        max_length,
        split="train",
        **kwargs,
    ):
        super().__init__(tokenizer, processor, max_length, **kwargs)

        assert split in ["train", "val", "test"]
        self.split = split

        self.dataset = self.load_data(kwargs["data_path"], split, kwargs["image_path"])

        logger.info(f"Load A-OKVQA {split} dataset. len: {len(self.dataset)}")

    def load_data(self, annotation_path, split, image_root):
        annotations = self.load(annotation_path, mode="json")

        data = []
        for dic in annotations:
            assert split == dic['split']
            image_id = dic['image_id']
            question = dic['question']
            choices = dic['choices']
            answer_idx = dic['correct_choice_idx']
            rationales = dic['rationales']
            direct_answers = dic['direct_answers']

            image_path = self.get_coco_path(image_id, image_root)
            data.append((
                str(image_path),
                {
                    "question": question,
                    "choices": choices,
                    "answer_idx": answer_idx,
                    "rationales": rationales,
                    "direct_answers": direct_answers,
                }
            ))

        data = self.finalize_data(data, task_type="aokvqa_vqa")

        return data

    def process_example_online(self, example):
        option, answer = optionize(example["choices"], example["answer_idx"])
        rationale = " ".join(example["rationales"])
        example = {
            "question": example["question"],
            "option": option,
            "answer": answer,
            "rationale": rationale,
        }

        return example

    def get_coco_path(self, image_id, coco_dir):
        return os.path.join(coco_dir, f"{image_id:012}.jpg")