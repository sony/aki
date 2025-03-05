# system messages
# SYSTEM_BASE = "The following is a conversation between a curious human and AI assistant."
# SYSTEM_DETAIL = "The assistant gives helpful, detailed, and polite answers to the user's questions."
SYSTEM_BASE = "A chat between a curious user and an artificial intelligence assistant."
SYSTEM_DETAIL = "The assistant gives helpful, detailed, and polite answers to the user's questions."
SYSTEM_MESSAGE = SYSTEM_BASE + " " + SYSTEM_DETAIL
SYSTEM_MESSAGE_ROLE = '<|system|>' + '\n' + SYSTEM_MESSAGE + '<|end|>'

# special media tokens
IMAGE = "<image>"
END = "<|end|>"

# Role pattern tokens
HUMAN = "<|user|>\n"
AI = "<|assistant|>\n"

ROLE_PATTERNS = {
    "human": f"\n{HUMAN}",
    "user": f"\n{HUMAN}",
    "\n[|Human|] ": f"\n{HUMAN}",
    "gpt": f"\n{AI}",
    "\n[|Assistant|] ": f"\n{AI}",
}
MEDIA_TOKENS = {
    "image": [IMAGE],
}

# constants
IGNORE_INDEX = -100  # default ignore index of CrossEntropyLoss


###############################################################################
# default patterns
###############################################################################

pattern_map = {
    "vqa": ["vqa", "vgqa", "ocrvqa", "okvqa"],
    "vqa-o": ["aokvqa"],  # vqa with options
    "vsr": ["vsr"],
    "kvqa": ["kvqa"],
    "loc": ["vg", "refexploc"],
    "captioning": ["coyo100m", "blip", "textcaps"],
}
pattern_dict = {
    # captioning
    "captioning": [
        # (SYSTEM_DETAIL, "Describe this image in detail.", "{caption}"),
        # ("", "Provide a one-sentence caption for the provided image.", "{caption}"),
        ("[NO_PROMPT]", "", "{caption}"),
    ],
    # VQA
    "vqa": [
        ("", "Answer the question using a single word or phrase. {question}", "{answer}"),
    ],
    # GQA -- VQA with answer & full_answer
    "gqa": [
        ("", "Answer the question using a single word or phrase. {question}", "{answer}"),
        # ("", "Answer the following question with a complete sentence. {question}", "{full_answer}"),
    ],
    # VQA with option
    "vqa-o": [
        ("", "Answer with the option's letter from the given choices directly. {question}\nOptions:\n{option}\n", "{answer}"),
        # ("", "First provide a rationale for your answer, then select the correct option for the following question. {question}\nThere are several options:\n{option}\n", "Rationale: {rationale}\nAnswer: {answer}"),
    ],
    # ScienceQA has context, lecture, and solution also
    "scienceqa": [
        ("", "Answer with the option's letter from the given choices directly. {question}\nContext: {context}\nOptions:\n{option}\n", "{answer}"),
    ],
    "loc": [
        # # visual grounding
        # ("", "Provide the bounding box coordinate of the region this sentence describes. {phrase}", "{bbox}"),
        # grounded captioning
        ("", "Provide a short description for this region. {bbox}", "{phrase}"),
    ],
    "vsr": [
        ("", "Answer the question using a single word or phrase. {question_interro} Please answer yes or no.", "{answer}"),
        # original caption with an additional instruction.
        #("", "Answer whether the following caption is correctly describing the given image with yes or no. {question}", "{answer}"),
    ],
    "kvqa": [
        # use original question
        ("", "Answer the question using a single word or phrase. {question}", "{answer}")
        # use paraphrased question
        #("", "Answer the question using a single word or phrase. {paraQuestion}", "{answer}")
    ],

    # REFTEPS-noname
    "refcoco": [
        # ("", "Provide the bounding box coordinate of the region this sentence describes. {phrase}", "{bbox}"),
        ("", "Provide a short description for this region. {bbox}", "{phrase}"),
    ],
    "refcocop": [
        # ("", "Provide the bounding box coordinate of the region this sentence describes. {phrase}", "{bbox}"),
        ("", "Provide a short description for this region. {bbox}", "{phrase}"),
    ],
    "refcocog": [
        # ("", "Provide the bounding box coordinate of the region this sentence describes. {phrase}", "{bbox}"),
        ("", "Provide a short description for this region. {bbox}", "{phrase}"),
    ],

    # Evaluation
    "mme": [("", "Answer the question using a single word or phrase. {question}", "")],
    "mmb": [("", "Answer with the option's letter from the given choices directly. {question}", "")],
    #
    "eval-vqa": [("", "Answer the question using a single word or phrase. {question}", "")],
    "eval-sqa": [
        ("", "Answer with the option's letter from the given choices directly. {question}\nContext: {context}\nThere are several options:\n{option}\n", "")
    ],
    "eval-refexploc": [("", "Provide the bounding box coordinate of the region this sentence describes. {phrase}", "")],
    "eval-vsr": [("", "Answer the question using a single word or phrase. {question_interro} Please answer yes or no.", "")],
}


# NOTE on template patterns
# Fixed texts: role tokens (Human, AI), image tokens, and system messages.
# You can control (instruction, input, target) using PATTERN_DICT.
#
# ```
# The following is a conversation between a curious human and AI assistant. {instruction}
# Human: <image>
# Human: {input}
# AI: {target}
# ```


def parse_pattern(pattern_dict: dict) -> dict:
    """Parsing patterns with pre-defined rules:
    - `-1` indicates "same as above"
    """
    new_pattern_dict = {}
    for k, patterns in pattern_dict.items():
        if isinstance(patterns, str):
            new_pattern_dict[k] = pattern_dict[patterns]
            continue

        new_patterns = []
        for i, (inst, inputs, targets) in enumerate(patterns):
            if inst == -1:
                assert i > 0
                inst = patterns[i - 1][0]

            new_patterns.append((inst, inputs, targets))

        new_pattern_dict[k] = new_patterns

    return new_pattern_dict


class Template:
    _registry = {}

    def __init__(self, pattern_dict: dict, pattern_map: dict):
        self.pattern_dict = parse_pattern(pattern_dict)
        self.pattern_map = pattern_map
        self.data2pattern = {}
        for pattern_name, dset_names in pattern_map.items():
            for dset_name in dset_names:
                assert dset_name not in self.data2pattern, "Duplicated dset name exists"
                self.data2pattern[dset_name] = pattern_name

    def get_pattern(self, dset_name):
        if dset_name in self.data2pattern:
            pattern_key = self.data2pattern[dset_name]
            return self.pattern_dict[pattern_key]
        elif dset_name in self.pattern_dict:
            return self.pattern_dict[dset_name]
        else:
            return None

    @classmethod
    def register(cls, name: str, pattern_dict: dict, pattern_map: dict):
        template = cls(pattern_dict, pattern_map)
        cls._registry[name] = template

    @classmethod
    def get(cls, name: str):
        return cls._registry[str(name)]


Template.register("default", pattern_dict, pattern_map)