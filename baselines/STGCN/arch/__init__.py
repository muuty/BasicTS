from .stgcn_arch import STGCNChebGraphConv as STGCN
from .stgcn_node_identity import STGCNNodeIdentity, STGCNWeakenedMP

__all__ = ["STGCN", "STGCNNodeIdentity", "STGCNWeakenedMP"]
