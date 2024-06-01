import argparse
import random
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
import wandb
from loguru import logger

# experimenting... https://docs.kidger.site/jaxtyping/api/runtime-type-checking/
from jaxtyping import install_import_hook
from beartype import beartype as typechecker
with install_import_hook("main", "beartype.beartype"):
    from twm.config import CONFIGS
    from twm.trainer import Trainer


def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('--game', type=str, default="Craftax-Symbolic-v1")
        parser.add_argument('--config', type=str, default='default')
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--device', type=str, default='cuda')
        parser.add_argument('--buffer_device', type=str, default='cpu')
        parser.add_argument('--cpu_p', type=float, default=0.5)
        parser.add_argument('--wandb', type=str, default='disabled')
        parser.add_argument('--project', type=str, default=None)
        parser.add_argument('--group', type=str, default=None)
        parser.add_argument('--save', action='store_true', default=False)
        parser.add_argument("--checkpoint", type=Path, default=None)
        args = parser.parse_args()
    else:
        args = argparse.Namespace(**args)

    if args.seed is not None:
        seed = args.seed
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        # https://pytorch.org/docs/stable/notes/randomness.html
        # slows down performance
        # torch.backends.cudnn.benchmark = False
        # torch.use_deterministic_algorithms(True)

    # improves performance, but is non-deterministic
    torch.backends.cudnn.benchmark = True

    if __debug__:
        print('Running in debug mode, consider using the -O python flag to improve performance')

    # enable wandb service (experimental, https://github.com/wandb/client/blob/master/docs/dev/wandb-service-user.md)
    # this hopefully fixes issues with multiprocessing
    wandb.require(experiment='service')

    buffer_device = args.buffer_device if args.buffer_device is not None else args.device

    config = deepcopy(CONFIGS[args.config])
    config.update({
        'game': args.game, 'seed': args.seed, 'model_device': args.device, 'buffer_device': buffer_device,
        'cpu_p': args.cpu_p, 'save': args.save
    })
    logger.info("config_name={args.config} config={config}")

    wandb.init(config=config, project=args.project, group=args.group, mode=args.wandb)
    config = dict(wandb.config)

    trainer = Trainer(config)

    if args.checkpoint:
        logger.info(f'Loading checkpoint from {args.checkpoint}')
        state_dict = torch.load(args.checkpoint, map_location=args.device)['state_dict']
        trainer.agent.load_state_dict(state_dict)

    trainer.print_stats()
    try:
        trainer.run()
    finally:
        trainer.close()


if __name__ == '__main__':
    main()
