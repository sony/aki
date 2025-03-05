import random
from .templates import Template, SYSTEM_MESSAGE_ROLE, IMAGE, HUMAN, AI, END


IMAGE_PROMPT = HUMAN + IMAGE


def join(items: list[str | None], sep: str) -> str:
    """Join items with separator.
    If item is None or empty, it will be ignored.
    """
    items = [item for item in items if item]
    return sep.join(items)


def match_pattern(pattern: str, items: dict) -> str:
    """Generate text (including input and target) by pattern-item matching.
    """
    return pattern.format(**items)


def templatize_single(
    pattern: tuple[str],
    ex_dict: dict,
    role_sep="\n",
    input_role=HUMAN,
    target_role=AI,
    image_prompt=None,
    end_user_token=END,
) -> str:
    """Templatize single ex_dict.
    NOTE This function cover only single-turn, which means that
    system message and image tokens are not included.
    """
    inputs = match_pattern(pattern[1], ex_dict)
    targets = match_pattern(pattern[2], ex_dict)

    # add role token
    # no need to add the role if having image prompt
    if image_prompt is not None:
        inputs = inputs + end_user_token
    else:
        inputs = input_role + inputs + end_user_token
    targets = target_role + targets

    return role_sep.join([inputs, targets])


def templatize(pattern: tuple[str], examples: list[dict], image_prompt=IMAGE_PROMPT) -> str:
    """One-shot templatize given data, handling both single- and multi-turn.

    Args:
        pattern (tuple[str]): (instruction, input, target)
        examples (list[dict])

    Example structure)
        examples = [
            {
                "question": Q,
                "answer": A,
            },
            ...
        ]

    Templatized text example (single-turn):
        ```
        The following is a conversation between a curious human and AI assistant. {instruction}
        Human: <image>
        Human: {input}
        AI: {target}
        ```
    """
    # separators are not configurable for now.
    isep = " "
    sep = "\n"

    inst = pattern[0]
    instruction = join([SYSTEM_MESSAGE_ROLE, inst], isep)
    inputs_targets = [
        templatize_single(pattern, ex_dict, image_prompt=image_prompt)
        for ex_dict in examples
    ]
    text = join([instruction, image_prompt, *inputs_targets], sep)

    return text


class Templatizer:
    """Recommendation in use of Template
    - Use defaults for N/A augmentation
    - Update your item dict itself for optional item prefixes
        - E.g., if you want to remove prefix `Context:` when null `context` item,
            add `Context:` in your `items` instead of adding `Context:` in patterns.
    """
    @classmethod
    def from_names(cls, template_name: str, dataset_name: str):
        template = Template.get(template_name)
        patterns = template.get_pattern(dataset_name)
        if patterns is None:
            print(
                f"WARNING: dataset name {dataset_name} is not in template {template_name}. "
                "Templatizer will not be applied for this dataset. "
                "Note that some dataset class (e.g., BaseTaskDataset) requires templatizer."
            )
            return None
        else:
            return cls(patterns)

    def __init__(self, patterns: list[str], defaults=None):
        self.patterns = patterns
        self.defaults = defaults or {}

    def sample(self, examples: list[dict], image_prompt=IMAGE_PROMPT):
        """Sample a template from the pattern list and return the templatized items.
        """
        if self.defaults:
            examples = [
                self.defaults | ex_dict
                for ex_dict in examples
            ]
        pattern = random.choice(self.patterns)
        return templatize(pattern, examples, image_prompt=image_prompt)

    def __call__(self, examples: list[dict], image_prompt=IMAGE_PROMPT):
        return self.sample(examples, image_prompt)
