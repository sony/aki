from pathlib import Path

from setuptools import find_packages, setup

if __name__ == "__main__":
    REQUIREMENTS = [
        "einops",
        "einops-exts",
        "transformers>=4.28.1",
        "torch>=2.0.1",
        "pillow",
        "open_clip_torch>=2.16.0",
        "sentencepiece",
    ]

    EVAL = [
        "scipy",
        "torchvision",
        "nltk",
        "inflection",
        "pycocoevalcap",
        "pycocotools",
        "tqdm",
    ]

    TRAINING = [
        "wandb",
        "torchvision",
        "braceexpand",
        "webdataset",
        "tqdm",
    ]

    setup(
        name="AKI",
        packages=find_packages(),
        include_package_data=True,
        version="2.0.1",
        license="MIT",
        description="Unlocking Causal Attention into Modality-Mutual Attention for Multimodal LLMs",
        long_description="None",
        long_description_content_type="text/markdown",
        data_files=[(".", ["README.md"])],
        keywords=["machine learning"],
        install_requires=REQUIREMENTS,
        extras_require={
            "eval": EVAL,
            "training": TRAINING,
            "all": list(set(REQUIREMENTS + EVAL + TRAINING)),
        },
        classifiers=[
            "Programming Language :: Python :: 3.12",
        ],
    )
