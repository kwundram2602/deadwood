import logging

from omegaconf import DictConfig, OmegaConf


def setup_logger(name: str = "train") -> logging.Logger:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )
    return logging.getLogger(name)


def init_wandb(cfg: DictConfig, model=None) -> None:
    """Initialise a W&B run. No-op if wandb not installed or use_wandb is false.

    Reads cfg.logging.{use_wandb, project, run_name, log_every_n_steps}.
    """
    log_cfg = cfg.get("logging", None)
    if log_cfg is None or not log_cfg.get("use_wandb", False):
        return

    try:
        import wandb
    except ImportError:
        print("wandb not installed — run: uv add wandb")
        return

    wandb.init(
        project=log_cfg.project,
        name=log_cfg.get("run_name") or None,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    if model is not None:
        wandb.watch(model, log_freq=log_cfg.get("log_every_n_steps", 50))
