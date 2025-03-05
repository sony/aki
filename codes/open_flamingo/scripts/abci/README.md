# ABCI instruction

- Manual: https://docs.abci.ai/v3/en/

## Installation

Create vertual environment within interactive job

```sh
# create interactive job
qsub -I -P gcf51138 -q rt_HG -l select=1 -l walltime=12:00:00 -m n

bash create_venv.sh
```

## Submit training job

```sh
# Submit job
qsub scripts/abci/run_train_abci.sh

qsub scripts/abci/run_sft.sh
```