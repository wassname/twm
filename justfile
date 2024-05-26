

main:
    . .venv/bin/activate
    TQDM_MININTERVAL=5 python -O scripts/main.py

test:
    . .venv/bin/activate
    TQDM_MININTERVAL=5 WANDB_MODE=disabled WANDB_SILENT=true python scripts/main.py \
        --config=test --wandb=disabled
