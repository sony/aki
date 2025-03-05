import os
import re
import json
import logging
from typing import List

from .templates.templates import SYSTEM_MESSAGE_ROLE, ROLE_PATTERNS, MEDIA_TOKENS, HUMAN
from .base import BaseDataset

# Set up basic configuration for logging
logger = logging.getLogger(__name__)


def remove_special_token_from_text(txt, patterns=None):
    if patterns is not None:
        for pattern in patterns:
            if pattern in txt:
                txt = txt.replace(pattern, "")

    # if a special media token in the conversation, replace it to a non-special token.
    for v in MEDIA_TOKENS.values():
        for v_ in v:
            txt = txt.replace(v_, "".join([c for c in v_ if c not in ["<", ">"]]))

    return txt


def chunking_by_keyword(txt, keyword_patterns=["<image>\n", "\n<image>"]):
    pattern = "|".join(map(re.escape, keyword_patterns))
    chunk_strs = re.split(f"({pattern})", txt)
    chunk_strs = [x for x in chunk_strs if len(x) > 0]

    return chunk_strs


class LLaVAInstructDataset(BaseDataset):
    def __init__(
        self,
        tokenizer,
        processors,
        max_length,
        image_tokens: List[str] = [
            "<image>\n",
            "\n<image>",
        ],  # special tokens for an image in this dataset
        **kwargs,
    ):
        super().__init__(tokenizer, processors, max_length, **kwargs)

        self.image_tokens = image_tokens
        self.dataset = self.load_data(kwargs["data_path"], kwargs["image_path"], SYSTEM_MESSAGE_ROLE)

        logger.info(f"Load Llava150k dataset. len: {len(self.dataset)}")

    def get_dataset_name(self):
        return "llava"

    def load_data(self, annotation_path, image_root, system_message):
        """Data is a list where each item is similar to following
        {
            "id": "005116462",
            "image": "00511/005116462.jpg",
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\nRender a clear and concise summary of the photo."
                },
                {
                    "from": "gpt",
                    "value": "$ 10 - cute cheap printed mini dress - khaki multicolor striped floral print peasant short sleeve tunic"
                }
            ]
        },
        """

        with open(annotation_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        parsed_data = []
        for item in raw_data:
            """Prepare data with instruction"""
            temp_text = system_message
            convs = item["conversations"]
            for conv in convs:
                temp_dict = {
                    "image": os.path.join(image_root, item["image"]),
                    "task_type": f"{self.get_dataset_name()}_inst",
                }
                role = conv["from"]
                temp_text += ROLE_PATTERNS[role]  # e.g., '\nHuman: '
                
                # append the image token for each user's turn
                # assume always has an image token
                if ROLE_PATTERNS[role] == f"\n{HUMAN}":
                    temp_text += f"{MEDIA_TOKENS['image'][0]}\n"

                txts = chunking_by_keyword(
                    conv["value"], keyword_patterns=self.image_tokens
                )
                for txt in txts:
                    # no need to add the image token again
                    if txt in self.image_tokens:
                        continue
                    temp_text += txt

                # add end token after appending the query
                if role == "human":
                    temp_text += "<|end|>"

                # split to each human-gpt pair
                # assume always H-G-H-G-...
                if role == "gpt" or role == "\n[|Assistant|] ":
                    temp_dict["text"] = temp_text
                    parsed_data.append(temp_dict)
                    temp_text = system_message    #initialize with instruction

        return parsed_data
