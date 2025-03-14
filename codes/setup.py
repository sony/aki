from pathlib import Path

from setuptools import find_packages, setup

if __name__ == "__main__":
    REQUIREMENTS = [
        "einops",
        "einops-exts",
        "transformers==4.41.2",
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
        name="AKIVLM",
        packages=find_packages(),
        include_package_data=True,
        version="0.1.0",
        license="CC-BY-NC 4.0",
        description="Seeing is Understanding: Unlocking Causal Attention into Modality-Mutual Attention for Multimodal LLMs",
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
            'Intended Audience :: Developers',
            'Intended Audience :: Education',
            'Intended Audience :: Science/Research',
        ],
    )
