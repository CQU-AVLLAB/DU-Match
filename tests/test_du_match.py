import unittest
from copy import deepcopy
import torch
from torch.nn import functional as F
from util.du_match import PrototypeBank, aligned_strips, consistency_loss, cutmix_targets, entropy

OPTIONS = dict(confidence_threshold=.9, prediction_threshold=.3, feature_threshold=.7,
               prototype_temperature=.5, gamma_min=0., gamma_max=.8, alpha=1., beta=1.)


class DualSpaceTests(unittest.TestCase):
    def test_entropy_uniform_one_hot_and_intermediate(self):
        p = torch.tensor([[[[1., .5, .9]], [[0., .5, .1]]]])
        actual = entropy(p)
        expected = torch.tensor([[[0., 1., .4689956]]])
        torch.testing.assert_close(actual, expected)

    def test_alignment_matches_full_interpolation(self):
        torch.manual_seed(4)
        features = torch.randn(2, 5, 3, 4)
        actual = torch.cat([z for _, _, z in aligned_strips(features, (11, 13), 3)], 2)
        expected = F.normalize(F.interpolate(features, (11, 13), mode='bilinear', align_corners=True), dim=1)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)

    def test_statistics_initialization_absent_class_and_resume(self):
        bank = PrototypeBank(3, 2, momentum=.5, variance_momentum=.5)
        features = torch.tensor([[[[1., 0.]], [[0., 1.]]]])
        bank.update(features, torch.tensor([[[0, 0]]]))
        torch.testing.assert_close(bank.centers[0], torch.tensor([2 ** -.5, 2 ** -.5]))
        self.assertAlmostEqual(bank.variance[0].item(), .5)
        saved = deepcopy(bank.state_dict())
        bank.update(features, torch.tensor([[[1, 255]]]))
        torch.testing.assert_close(bank.centers[0], saved['centers'][0])
        torch.testing.assert_close(bank.variance[0], saved['variance'][0])
        self.assertEqual(bank.updates.tolist(), [1, 1, 0])
        recovered = PrototypeBank(3, 2)
        recovered.load_state_dict(bank.state_dict())
        for key, value in bank.state_dict().items():
            torch.testing.assert_close(value, recovered.state_dict()[key])

    def test_four_states_correction_ignore_and_unseen_fallback(self):
        bank = PrototypeBank(3, 2, rows=2)
        bank.centers.copy_(torch.tensor([[1., 0.], [0., 1.], [-1., 0.]]))
        bank.variance.fill_(.1)
        bank.updates.fill_(2)
        p = torch.tensor([[[[.98, .5, .98, .5, .98]], [[.01, .3, .01, .3, .01]], [[.01, .2, .01, .2, .01]]]])
        features = torch.tensor([[[[1., 1., 0., 0., 1.]], [[0., 0., 1., 1., 0.]]]])
        ignore = torch.tensor([[[0, 0, 0, 0, 255]]])
        targets = bank.targets(p, features, ignore, OPTIONS)
        for state, index in [('reliable', 0), ('ambiguous', 1), ('suspicious', 2), ('uncertain', 3)]:
            self.assertEqual(targets[state].sum().item(), 1)
            self.assertTrue(targets[state][0, 0, index])
        self.assertGreater(targets['soft'][0, 1, 0, 2].item(), .7)
        self.assertLess(targets['gamma'][0, 0, 2].item(), .801)
        torch.testing.assert_close(targets['soft'].sum(1), torch.ones_like(ignore).float())
        bank.updates[0] = 0
        unseen = bank.targets(p, features, ignore, OPTIONS)
        self.assertEqual(unseen['fallback'].sum().item(), 4)
        self.assertEqual(unseen['suspicious'].sum().item(), 0)
        bank.updates.zero_()
        empty = bank.targets(p, features, ignore, OPTIONS)
        self.assertTrue(torch.isfinite(empty['soft']).all())
        self.assertEqual(empty['hard_mask'].sum().item(), 2)

    def test_partial_bank_uniform_distribution_has_no_correction(self):
        bank = PrototypeBank(3, 2)
        bank.centers.copy_(torch.tensor([[1., 0.], [1., 0.], [0., 1.]]))
        bank.variance.fill_(.1)
        bank.updates[:2] = 2
        targets = bank.targets(torch.tensor([[[[.98]], [[.01]], [[.01]]]]),
                               torch.tensor([[[[0.]], [[1.]]]]), torch.zeros(1, 1, 1).long(), OPTIONS)
        self.assertTrue(targets['suspicious'].item())
        self.assertAlmostEqual(targets['gamma'].item(), 0., places=6)

    def test_loss_manual_gradient_and_empty(self):
        logits = torch.tensor([[[[2., .2, -.3, 8.]], [[0., -.2, .3, -8.]]]], requires_grad=True)
        target = dict(hard=torch.tensor([[[0, 1, 0, 0]]]),
                      soft=torch.tensor([[[[1., .4, .2, .5]], [[0., .6, .8, .5]]]]),
                      hard_mask=torch.tensor([[[True, False, False, False]]]),
                      ambiguous=torch.tensor([[[False, True, False, False]]]),
                      suspicious=torch.tensor([[[False, False, True, False]]]))
        loss = consistency_loss(logits, target, .5, .7)
        lp = logits.log_softmax(1)
        expected = (-lp[0, 0, 0, 0] + .5 * F.kl_div(lp[:, :, :, 1:2], target['soft'][:, :, :, 1:2], reduction='sum')
                    + .7 * F.kl_div(lp[:, :, :, 2:3], target['soft'][:, :, :, 2:3], reduction='sum')) / 3
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertEqual(logits.grad[:, :, :, 3].abs().sum().item(), 0)
        for key in ('hard_mask', 'ambiguous', 'suspicious'):
            target[key].zero_()
        empty = consistency_loss(logits, target)
        self.assertEqual(empty.item(), 0.)
        self.assertTrue(torch.isfinite(empty))

    def test_cutmix_transports_all_targets(self):
        source = dict(hard=torch.arange(8).reshape(2, 2, 2), soft=torch.arange(24).reshape(2, 3, 2, 2),
                      valid=torch.tensor([[[True, False], [True, True]], [[False, True], [False, False]]]))
        box = torch.tensor([[[1, 0], [0, 1]], [[0, 1], [1, 0]]])
        mixed = cutmix_targets(source, box)
        for name, tensor in source.items():
            for b in range(2):
                for y in range(2):
                    for x in range(2):
                        donor = 1 - b if box[b, y, x] else b
                        torch.testing.assert_close(mixed[name][b, ..., y, x], tensor[donor, ..., y, x])


if __name__ == '__main__':
    unittest.main()