import os.path as osp
import logging
from pathlib import Path

from .base_task import BaseTaskDataset

# Set up basic configuration for logging
logger = logging.getLogger(__name__)


class VGDataset(BaseTaskDataset):
    """Visual Genome localization dataset"""
    def __init__(
        self,
        tokenizer,
        processors,
        max_length,
        bbox_coord_style=3,
        **kwargs,
    ):
        super().__init__(tokenizer, processors, max_length, **kwargs)

        self.bbox_coord_style = bbox_coord_style
        self.dataset = self.load_data(kwargs["data_path"], kwargs["image_metadata_path"], kwargs["image_path"])

        logger.info(f"Load VGLoc dataset. len: {len(self.dataset)}")

    def load_meta(self, path: str | Path):
        js = self.load(path)
        id_key = "image_id" if "image_id" in js[0] else "id"
        meta = {}
        for dic in js:
            image_id = dic.pop(id_key)
            meta[image_id] = dic
        return meta

    def load_data(self, annotation_path, image_metadata_path, image_root):
        images = self.load(image_metadata_path)
        images = {
            dic["image_id"]: dic
            for dic in images
        }
        regions = self.load(annotation_path)

        data = []
        for dic in regions:
            image_id = dic["id"]
            region_list = dic["regions"]
            image_path = osp.join(image_root, f"{image_id}.jpg")
            for rdic in region_list:
                assert image_id == rdic["image_id"]
                x = rdic["x"]
                y = rdic["y"]
                w = rdic["width"]
                h = rdic["height"]

                image_dic = images[image_id]
                W = image_dic["width"]
                H = image_dic["height"]

                # formatting bbox
                bbox_str = self.preprocess_bbox(
                    x, y, w, h, W, H,
                    bbox_coord_style=self.bbox_coord_style,
                )

                phrase = rdic["phrase"]

                data.append((image_path, {"phrase": phrase, "bbox": bbox_str}))

        data = self.finalize_data(data, task_type="vgloc_loc")

        return data
