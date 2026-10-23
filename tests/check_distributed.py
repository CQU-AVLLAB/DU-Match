"""CPU/Gloo test for pixel-weighted statistics when local class counts differ."""
import os
import tempfile
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from util.du_match import PrototypeBank


def worker(rank, rendezvous):
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2)
    bank = PrototypeBank(3, 2)
    if rank == 0:
        features = torch.tensor([[[[1., 1.]], [[0., 0.]]]])
        labels = torch.tensor([[[0, 0]]])
    else:
        features = torch.tensor([[[[0., 0.]], [[1., 1.]]]])
        labels = torch.tensor([[[0, 1]]])
    bank.update(features, labels)
    expected_mean = torch.tensor([2/3, 1/3])
    torch.testing.assert_close(bank.centers[0], expected_mean / expected_mean.norm())
    torch.testing.assert_close(bank.variance[0], torch.tensor(4/9))
    assert bank.pixel_counts.tolist() == [3, 1, 0]
    assert bank.updates.tolist() == [1, 1, 0]
    dist.destroy_process_group()


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(worker, args=(os.path.join(directory, 'rendezvous'),), nprocs=2, join=True)
    print('Two-process prototype aggregation: PASS')