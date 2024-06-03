
set export
export OSTYPE := "linux-gnu"
export TQDM_MININTERVAL := "30"

main:
    . .venv/bin/activate
    echo $(which python)
    python -O scripts/main.py \
        --wandb=online \
        --save \
        --checkpoint './wandb/run-20240601_211958-mv00l07m/files/agent_final.pt'

test:
    . .venv/bin/activate
    # TQDM_MININTERVAL=30 WANDB_MODE=disabled WANDB_SILENT=true python scripts/main.py --config=test --wandb=disabled
    python scripts/main.py --config=test
