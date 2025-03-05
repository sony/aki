#!/bin/bash

torchrun --nproc-per-node=1 run.py --data LLaVABench MME MMBench_DEV_EN SEEDBench_IMG POPE MMVet MathVista_MINI RealWorldQA MMMU_DEV_VAL --model AKI --work-dir {expected_store_path}
