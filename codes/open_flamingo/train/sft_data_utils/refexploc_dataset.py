import os.path as osp
import logging
from pathlib import Path

from .base_task import BaseTaskDataset

# Set up basic configuration for logging
logger = logging.getLogger(__name__)


class RefExpLocDataset(BaseTaskDataset):
    """Visual Genome localization dataset"""

    def __init__(
        self,
        tokenizer,
        processors,
        max_length,
        dataname: str = "refcoco",  # support refcoco, refcoco+, refcocog
        bbox_coord_style=3,
        **kwargs,
    ):
        super().__init__(tokenizer, processors, max_length, **kwargs)

        self.bbox_coord_style = bbox_coord_style
        self.dataset = self.load_data(kwargs["data_path"], kwargs["image_path"], dataname)

        logger.info(f"Load RefExpLoc dataset ({dataname}). len: {len(self.dataset)}")

    def load_data(self, meta_root_path, image_root_path, dataname):
        # We use annotations used in MDETR (https://github.com/ashkamath/mdetr)
        # refer to https://github.com/ashkamath/mdetr/blob/main/.github/refexp.md
        # and download annotations from
        # https://zenodo.org/record/4729015/files/mdetr_annotations.tar.gz?download=1
        meta_root_path = Path(meta_root_path)

        """ Load annotations
        Two types of annotation exists: images / annotations (two are paired; same order)
        Below is an example of annotations:
        1) structure of images:
            'file_name': 'COCO_train2014_000000519404.jpg',
            'height': 480,
            'width': 640,
            'id': 262407,  # id for RefExpLoc
            'original_id': 519404,  # original image id of COCO
            'caption': 'two woman one in black eatting and the other has a white shirt at the desk',
            'dataset_name': 'refcocog',
        2) structure of annotations:
            'area': 57185.38189999998,
            'iscrowd': 0,
            'image_id': 262407,  # same with id of the paired image annotation
            'category_id': 1,
            'id': 262407,
            'bbox': [0.0, 45.95, 238.92, 408.64],
            'original_id': 1241542,
        """
        ann = self.load(meta_root_path)
        img_info = ann["images"]
        ann_info = ann["annotations"]
        assert len(img_info) == len(ann_info)

        # prepare data
        data = []
        for img, ann in zip(img_info, ann_info):
            assert img["id"] == ann["image_id"]
            image_path = osp.join(image_root_path, img["file_name"])
            phrase = img["caption"]  # referring expression

            # get bbox infomation
            x, y, w, h = ann["bbox"]
            W = img["width"]
            H = img["height"]

            # formatting bbox
            bbox_str = self.preprocess_bbox(
                x, y, w, h, W, H,
                bbox_coord_style=self.bbox_coord_style,
            )

            data.append((image_path, {"phrase": phrase, "bbox": bbox_str}))

        task_type = f"{dataname}_loc"
        data = self.finalize_data(data, task_type=task_type)

        return data
