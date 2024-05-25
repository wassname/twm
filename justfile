

main:
    . .venv/bin/activate
    python -O scripts/main.py

test:
    . .venv/bin/activate
    WANDB_MODE=disabled WANDB_SILENT=true python scripts/main.py \
        --config=test --wandb=disabled
