import unittest
import torch
from util.du_match import supervised_loss

class EmptyCropTests(unittest.TestCase):
    def test_empty_crop_zero_and_finite_gradient(self):
        x = torch.randn(1, 19, 4, 4, requires_grad=True)
        y = torch.full((1, 4, 4), 255, dtype=torch.long)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
        self.assertTrue(torch.isnan(criterion(x, y)))
        loss = supervised_loss(x, y, criterion)
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertEqual(x.grad.abs().sum().item(), 0)

    def test_nonempty_matches_original_loss_and_gradient(self):
        x = torch.randn(1, 19, 4, 4, requires_grad=True)
        y = torch.zeros((1, 4, 4), dtype=torch.long)
        y[:, :2] = 255
        criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
        a = supervised_loss(x, y, criterion)
        b = criterion(x, y)
        torch.testing.assert_close(a, b)
        torch.testing.assert_close(torch.autograd.grad(a, x)[0], torch.autograd.grad(b, x)[0])

    def test_nonfinite_logits_are_not_hidden(self):
        x = torch.full((1, 19, 4, 4), float('nan'), requires_grad=True)
        y = torch.full((1, 4, 4), 255, dtype=torch.long)
        self.assertFalse(torch.isfinite(supervised_loss(x, y, torch.nn.CrossEntropyLoss(ignore_index=255))))

if __name__ == '__main__': unittest.main()
