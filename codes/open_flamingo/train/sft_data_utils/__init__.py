from .aokvqa_dataset import AOKVQADataset
from .vqa_dataset import VQADataset
from .gqa_dataset import GQADataset
from .ocrvqa_dataset import OCRVQADataset
from .vsr_dataset import VSRDataset
from .scienceqa_dataset import ScienceQADataset
from .refexploc_dataset import RefExpLocDataset
from .vg_dataset import VGDataset
from .llava_dataset import LLaVAInstructDataset

from .templates.templatizer import Templatizer
from data_utils import DataInfo
from .multidata_wrapper import MultiDataset
from .loader_utils import batchify, seed_worker

from torch.utils.data import DataLoader
from functools import partial


DATASET_CLASS_LIST = [
    LLaVAInstructDataset,
    ScienceQADataset,
    OCRVQADataset,
    VQADataset,
    AOKVQADataset,
    GQADataset,
    VGDataset,
    VSRDataset,
    RefExpLocDataset,
]
DATASET_CLASS_DICT = {c.__name__: c for c in DATASET_CLASS_LIST}


def get_sft_data(
    dataset_name, tokenizer, processor, max_length, class_name, template_name, **kwargs
):
    dataset_class = DATASET_CLASS_DICT[class_name]
    dataset = dataset_class(tokenizer, processor, max_length, **kwargs)

    if template_name is not None:
        templatizer = Templatizer.from_names(template_name, dataset_name)
        dataset.set_templatizer(templatizer)

    return dataset


def dataset_provider(data_config, training_config, tokenizer, processor, num_gpus):
    datasets = [
        get_sft_data(
            dataset_name=dataset_name, 
            tokenizer=tokenizer, 
            processor=processor, 
            max_length=training_config["max_length"], 
            class_name=dataset["classname"],
            template_name=training_config["template"], 
            **dataset["data_cfg"],
        )
        for dataset_name, dataset in data_config.items()
    ]

    if len(datasets) > 1:
        # wrap with Multidataset class
        dataset = MultiDataset(datasets, sampling_weights=list(training_config["sampling_weights"]), num_gpus=num_gpus, batch_per_device=training_config["batch_size"])
    else:
        dataset = datasets[0]

    collate_fn = partial(
        batchify,
        tokenizer=tokenizer,
        max_length=training_config["max_length"],
    )

    dataloader = DataLoader(
        dataset,
        batch_size=training_config["batch_size"],
        num_workers=training_config["workers"],
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
    )

    # add meta-data to dataloader instance for convenience
    # dataloader.num_batches = num_batches
    dataloader.num_samples = dataset.total_datasets_len

    return DataInfo(name='instruction-finetune-mix', dataloader=dataloader, shared_epoch=None)

