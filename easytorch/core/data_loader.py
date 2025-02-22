from typing import Dict

from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import SubsetRandomSampler

from selection.embedding.factory import get_embedding
from selection.factory import get_selection
from selection.outlier_removal.factory import get_outlier_removal
from ..utils import get_rank, get_world_size
from ..utils.data_prefetcher import DataLoaderX


def build_data_loader(dataset: Dataset, data_cfg: Dict, model_cfg: Dict = None) -> DataLoader:
    """Build dataloader from `data_cfg`
    `data_cfg` is part of config which defines fields about data, such as `CFG.TRAIN.DATA`

    structure of `data_cfg` is
    {
        'BATCH_SIZE': (int, optional) batch size of data loader (default: ``1``),
        'SHUFFLE': (bool, optional) data reshuffled option (default: ``False``),
        'NUM_WORKERS': (int, optional) num workers for data loader (default: ``0``),
        'PIN_MEMORY': (bool, optional) pin_memory option (default: ``False``),
        'PREFETCH': (bool, optional) set to ``True`` to use `DataLoaderX` (default: ``False``),
    }

    Args:
        dataset (Dataset): dataset defined by user
        data_cfg (Dict): data config

    Returns:
        data loader
    """

    embedding = get_embedding(data_cfg.get('OUTLIER_REMOVAL_EMBEDDING_STRATEGY'))
    outlier_removal = get_outlier_removal(type=data_cfg.get('OUTLIER_REMOVAL_STRATEGY'),
                                          outlier_removal_ratio=data_cfg.get('OUTLIER_REMOVAL_RATIO', 0.1),
                                          embedding=embedding,
                                          data_config=data_cfg,
                                          model_config=model_cfg)

    if outlier_removal is not None:
        normal_indices = outlier_removal.get_normal_indices(dataset)
        dataset = Subset(dataset, normal_indices)

    embedding = get_embedding(data_cfg.get('EMBEDDING_STRATEGY')) if data_cfg.get('EMBEDDING_STRATEGY') else None
    selection = get_selection(type=data_cfg.get('SELECTION_STRATEGY'),
                              selection_ratio=data_cfg.get('SELECTION_RATIO'),
                              embedding_model=embedding,
                              dataset=dataset
                              )
    sampler = SubsetRandomSampler(selection.select_indices()) if selection is not None else None

    return (DataLoaderX if data_cfg.get('PREFETCH', False) else DataLoader)(
        dataset,
        collate_fn=data_cfg.get('COLLATE_FN', None),
        batch_size=data_cfg.get('BATCH_SIZE', 1),
        shuffle=data_cfg.get('SHUFFLE', False) if sampler is None else False,
        num_workers=data_cfg.get('NUM_WORKERS', 0),
        sampler=sampler,
        pin_memory=data_cfg.get('PIN_MEMORY', False)
    )


def build_data_loader_ddp(dataset: Dataset, data_cfg: Dict):
    """Build ddp dataloader from `data_cfg`
    `data_cfg` is part of config which defines fields about data, such as `CFG.TRAIN.DATA`

    structure of `data_cfg` is
    {
        'BATCH_SIZE': (int, optional) batch size of data loader (default: ``1``),
        'SHUFFLE': (bool, optional) data reshuffled option (default: ``False``),
        'NUM_WORKERS': (int, optional) num workers for data loader (default: ``0``),
        'PIN_MEMORY': (bool, optional) pin_memory option (default: ``False``),
        'PREFETCH': (bool, optional) set to ``True`` to use `BackgroundGenerator` (default: ``False``)
            need to install `prefetch_generator`, see https://pypi.org/project/prefetch_generator/
    }

    Args:
        dataset (Dataset): dataset defined by user
        data_cfg (Dict): data config

    Returns:
        data loader
    """

    ddp_sampler = DistributedSampler(
        dataset,
        get_world_size(),
        get_rank(),
        shuffle=data_cfg.get('SHUFFLE', False)
    )
    return (DataLoaderX if data_cfg.get('PREFETCH', False) else DataLoader)(
        dataset,
        collate_fn=data_cfg.get('COLLATE_FN', None),
        batch_size=data_cfg.get('BATCH_SIZE', 1),
        shuffle=False,
        sampler=ddp_sampler,
        num_workers=data_cfg.get('NUM_WORKERS', 0),
        pin_memory=data_cfg.get('PIN_MEMORY', False)
    )
