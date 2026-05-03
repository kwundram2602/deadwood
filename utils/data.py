from torch.utils.data import Dataset, Subset


def train_val_test_split(
    dataset: Dataset,
    train_ratio: float,
    val_ratio: float,
    seed: int = 42,
) -> tuple[Subset, Subset, Subset]:
    raise NotImplementedError


def kfold_split(
    dataset: Dataset,
    k: int,
    seed: int = 42,
) -> list[tuple[Subset, Subset]]:
    raise NotImplementedError
