"""
Heterogeneity modeling modules for ST-SSL.

Reference: https://github.com/Echo-Ji/ST-SSL/blob/master/model/layers.py
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def sinkhorn(out: torch.Tensor, epsilon: float = 0.05, sinkhorn_iterations: int = 3) -> torch.Tensor:
    """
    Sinkhorn-Knopp algorithm for optimal transport.

    Args:
        out: logits tensor
        epsilon: regularization parameter
        sinkhorn_iterations: number of iterations

    Returns:
        soft assignment matrix
    """
    Q = torch.exp(out / epsilon).t()  # Q is K-by-B for consistency with notations from the paper
    B = Q.shape[1]  # number of samples to assign
    K = Q.shape[0]  # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    if sum_Q > 0:
        Q /= sum_Q
    else:
        return Q.t()

    for _ in range(sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        row_sum = torch.sum(Q, dim=1, keepdim=True)
        row_sum = torch.clamp(row_sum, min=1e-8)
        Q /= row_sum
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        col_sum = torch.sum(Q, dim=0, keepdim=True)
        col_sum = torch.clamp(col_sum, min=1e-8)
        Q /= col_sum
        Q /= B

    Q *= B  # the columns must sum to 1 so that Q is an assignment
    return Q.t()


class SpatialHeteroModel(nn.Module):
    """
    Spatial heterogeneity modeling by using a soft-clustering paradigm.

    Uses prototypes with Sinkhorn normalization to compute contrastive loss
    between two views of the spatial representations.
    """

    def __init__(self, c_in: int, nmb_prototype: int, batch_size: int, tau: float = 0.5):
        """
        Args:
            c_in: input feature dimension
            nmb_prototype: number of prototypes
            batch_size: batch size
            tau: temperature for softmax
        """
        super(SpatialHeteroModel, self).__init__()
        self.l2norm = lambda x: F.normalize(x, dim=1, p=2)
        self.prototypes = nn.Linear(c_in, nmb_prototype, bias=False)

        self.tau = tau
        self.d_model = c_in
        self.batch_size = batch_size

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Compute the contrastive loss of batched data.

        Args:
            z1, z2: representations from two views, shape [n, l, v, c]

        Returns:
            loss: contrastive loss
        """
        with torch.no_grad():
            w = self.prototypes.weight.data.clone()
            w = self.l2norm(w)
            self.prototypes.weight.copy_(w)

        # l2norm avoids nan of Q in sinkhorn
        zc1 = self.prototypes(self.l2norm(z1.reshape(-1, self.d_model)))  # nd -> nk
        zc2 = self.prototypes(self.l2norm(z2.reshape(-1, self.d_model)))  # nd -> nk

        with torch.no_grad():
            q1 = sinkhorn(zc1.detach())
            q2 = sinkhorn(zc2.detach())

        l1 = - torch.mean(torch.sum(q1 * F.log_softmax(zc2 / self.tau, dim=1), dim=1))
        l2 = - torch.mean(torch.sum(q2 * F.log_softmax(zc1 / self.tau, dim=1), dim=1))

        return l1 + l2


class AvgReadout(nn.Module):
    """Average readout for graph-level representation."""

    def __init__(self):
        super(AvgReadout, self).__init__()
        self.sigm = nn.Sigmoid()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Apply an average on graph.

        Args:
            h: hidden representation, (batch_size, num_nodes, feat_dim)

        Returns:
            s: summary, (batch_size, feat_dim)
        """
        s = torch.mean(h, dim=1)
        s = self.sigm(s)
        return s


class Discriminator(nn.Module):
    """Discriminator for contrastive learning."""

    def __init__(self, n_h: int):
        super(Discriminator, self).__init__()
        self.net = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, summary: torch.Tensor, h_rl: torch.Tensor, h_fk: torch.Tensor) -> torch.Tensor:
        """
        Args:
            summary: summary vector, (batch_size, feat_dim)
            h_rl: real hidden representation, (batch_size, num_nodes, feat_dim)
            h_fk: fake hidden representation, (batch_size, num_nodes, feat_dim)

        Returns:
            logits: prediction scores, (batch_size, num_nodes * 2)
        """
        s = torch.unsqueeze(summary, dim=1)
        s = s.expand_as(h_rl).contiguous()

        # score of real and fake, (batch_size, num_nodes)
        sc_rl = torch.squeeze(self.net(h_rl, s), dim=2)
        sc_fk = torch.squeeze(self.net(h_fk, s), dim=2)

        logits = torch.cat((sc_rl, sc_fk), dim=1)

        return logits


class TemporalHeteroModel(nn.Module):
    """
    Temporal heterogeneity modeling in a contrastive manner.

    Uses a discriminator to distinguish real vs shuffled representations.
    """

    def __init__(self, c_in: int, batch_size: int, num_nodes: int, device: str = 'cuda'):
        """
        Args:
            c_in: input feature dimension
            batch_size: batch size
            num_nodes: number of nodes
            device: device to use ('cuda' or 'cpu')
        """
        super(TemporalHeteroModel, self).__init__()
        self.W1 = nn.Parameter(torch.FloatTensor(num_nodes, c_in))
        self.W2 = nn.Parameter(torch.FloatTensor(num_nodes, c_in))
        nn.init.kaiming_uniform_(self.W1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W2, a=math.sqrt(5))

        self.read = AvgReadout()
        self.disc = Discriminator(c_in)
        self.b_xent = nn.BCEWithLogitsLoss()

        self.batch_size = batch_size
        self.num_nodes = num_nodes
        self._device = device

        # Pre-compute labels (will be moved to correct device in forward)
        self.register_buffer('lbl', self._create_labels(batch_size, num_nodes))

    def _create_labels(self, batch_size: int, num_nodes: int) -> torch.Tensor:
        """Create labels for discriminator."""
        lbl_rl = torch.ones(batch_size, num_nodes)
        lbl_fk = torch.zeros(batch_size, num_nodes)
        return torch.cat((lbl_rl, lbl_fk), dim=1)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1, z2: representations from two views, shape [n, l, v, c]

        Returns:
            loss: temporal contrastive loss
        """
        # Handle dynamic batch size
        current_batch_size = z1.size(0)
        if current_batch_size != self.batch_size:
            lbl = self._create_labels(current_batch_size, self.num_nodes).to(z1.device)
        else:
            lbl = self.lbl

        h = (z1 * self.W1 + z2 * self.W2).squeeze(1)  # nlvc->nvc
        s = self.read(h)  # s: summary of h, nc

        # select another region in batch
        idx = torch.randperm(current_batch_size, device=z1.device)
        shuf_h = h[idx]

        logits = self.disc(s, h, shuf_h)
        loss = self.b_xent(logits, lbl)

        return loss
