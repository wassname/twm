

main:
    . .venv/bin/activate
    TQDM_MININTERVAL=30 python -O scripts/main.py --wandb=online --save

test:
    . .venv/bin/activate
    # TQDM_MININTERVAL=30 WANDB_MODE=disabled WANDB_SILENT=true python scripts/main.py --config=test --wandb=disabled
    python scripts/main.py --config=test
