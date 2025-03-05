import os
import re
import json
import logging
from typing import Union

from .base_task import BaseTaskDataset, optionize

# Set up basic configuration for logging
logger = logging.getLogger(__name__)


def load_json_files(input_files: Union[list[str], str], key_pattern=None):
    raw_data_lst = []

    if isinstance(input_files, str):
        input_files = [input_files]

    for input_file in input_files:
        with open(input_file, "r") as f:
            data_temp = json.load(f)
        if key_pattern is not None:
            data_temp = data_temp[key_pattern]
        raw_data_lst.extend(data_temp)

    return raw_data_lst


def parse_question(question):
    """<question>\nContext: <context>\nOptions: (A) <option1> (B) <option2> ..."""
    pattern = r"(?P<question>.+?)\nContext: (?P<context>.*?)\nOptions: (?P<options>.+)"

    # re.DOTALL allows regex pattern `.` to match newlines
    match = re.search(pattern, question, re.DOTALL)
    assert match, f"Question {question} cannot be parsed."

    # remove <image>
    option_str = match.group("options").replace("<image>", "").strip()
    options = re.split(r"\([A-Z]\)", option_str)
    options = [option.strip() for option in options if option.strip()]

    parsed = {
        "question": match.group("question"),
        "context": match.group("context"),
        "options": options,
    }
    return parsed


def parse_answer(answer):
    """Parse an answer string into three groups corresponding to LECTURE, SOLUTION, and ###\nANSWER

    Args:
        answer (String): Sentences consisted of lecture, soution, and answer as one sequence.

    Returns:
        Dict: Parsed sentence dict.
    """
    _ANSWER_PATTERN_DICT = {
        "LECTURE:": "lecture",
        "SOLUTION:": "solution",
        "###\nANSWER:": "answer",
    }

    pattern = re.compile("|".join(_ANSWER_PATTERN_DICT.keys()))
    parts = pattern.split(answer)
    keys = pattern.findall(answer)

    parsed_dict = {value: None for value in _ANSWER_PATTERN_DICT.values()}
    for key, part in zip(keys, parts[1:]):
        parsed_dict[_ANSWER_PATTERN_DICT[key]] = part.lstrip()

    answer_index = ord(parsed_dict["answer"].rstrip(".")) - ord("A")
    parsed_dict["answer_index"] = answer_index

    return parsed_dict


class ScienceQADataset(BaseTaskDataset):
    def __init__(
        self,
        tokenizer,
        processors,
        max_length,
        **kwargs,
    ):
        super().__init__(tokenizer, processors, max_length, **kwargs)

        self.dataset = self.load_data(kwargs["data_path"], kwargs["image_path"])

        logger.info(f"Load ScienceQA instruction train split dataset. len: {len(self.dataset)}")

    def load_data(self, annotation_path, image_root):
        annotations = self.load(annotation_path, mode="json")

        parsed_data = []
        """
        "1":{
            "question":"Which of these states is farthest north?",
            "choices":[
            "West Virginia",
            "Louisiana",
            "Arizona",
            "Oklahoma"
            ],
            "answer":0,
            "hint":"",
            "image":"image.png",
            "task":"closed choice",
            "grade":"grade2",
            "subject":"social science",
            "topic":"geography",
            "category":"Geography",
            "skill":"Read a map: cardinal directions",
            "lecture":"Maps have four cardinal directions, or main directions. Those directions are north, south, east, and west.\nA compass rose is a set of arrows that point to the cardinal directions. A compass rose usually shows only the first letter of each cardinal direction.\nThe north arrow points to the North Pole. On most maps, north is at the top of the map.",
            "solution":"To find the answer, look at the compass rose. Look at which way the north arrow is pointing. West Virginia is farthest north.",
            "split":"train"
        }
        """
        for key, item in annotations.items():
            # only use train split
            if item["split"] != "train":
                continue

            if item["image"] is not None:
                image = os.path.join(image_root, key, item["image"])
            else:
                image = None
            # question, context, options
            question = item["question"]
            options = item["choices"]
            answer_index = item["answer"]
            solution = item["solution"]
            context = item["hint"] if item["hint"] != "" else "N/A"

            parsed_data.append(
                (
                    image,
                    {
                        "question": question,
                        "options": options,
                        "context": context,
                        "solution": solution,
                        "answer_index": answer_index,
                    },
                )
            )

        parsed_data = self.finalize_data(parsed_data, task_type="scienceqa_vqa")

        return parsed_data

    def process_example_online(self, ex):
        option, answer = optionize(ex["options"], ex["answer_index"])
        ex = {
            "question": ex["question"],
            "context": ex["context"],
            "option": option,
            "answer": answer,
            "solution": ex["solution"],
        }

        return ex
