import logging
from pathlib import Path

from .base_task import BaseTaskDataset

# Set up basic configuration for logging
logger = logging.getLogger(__name__)


class VQADataset(BaseTaskDataset):
    """VQAv2 dataset"""
    def __init__(
        self,
        tokenizer,
        processor,
        max_length,
        split="train",
        **kwargs,
    ):
        super().__init__(tokenizer, processor, max_length, **kwargs)

        assert split in ["train"]
        self.split = split

        self.dataset = self.load_data(kwargs["data_path"], kwargs["image_path"])

        logger.info(f"Load VQAv2 {split} dataset. len: {len(self.dataset)}")

    def load_data(self, annotation_path, image_path):
        raw_data = self.build_vqa_dataset(annotation_path, image_path)
        data = self.finalize_data(raw_data, task_type="vqa_vqa")

        return data

    def build_vqa_dataset(self, annotation_path, image_path):
        qjs = self.load(f"{annotation_path}/v2_OpenEnded_mscoco_train2014_questions.json", mode="json")
        ajs = self.load(f"{annotation_path}/v2_mscoco_train2014_annotations.json", mode="json")

        questions = qjs["questions"]
        annotations = ajs["annotations"]
        assert len(questions) == len(annotations)
        assert qjs["data_subtype"] == ajs["data_subtype"]

        data_subtype = qjs["data_subtype"]

        data = []
        for q, a in zip(questions, annotations):
            assert q["question_id"] == a["question_id"]
            assert q["image_id"] == a["image_id"]

            image_id = q["image_id"]
            img_fn = 'COCO_' + data_subtype + '_'+ str(image_id).zfill(12) + '.jpg'
            img_path = f"{image_path}{img_fn}"

            data.append((
                str(img_path),
                {
                    "question": q["question"],
                    "answer": a["multiple_choice_answer"],
                },
            ))

        return data