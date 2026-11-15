"""CPU tests for the label-conditioned interaction-structure posterior."""

import unittest

import torch

from rimoe.bayesian_structure import laplace_mixture_statistics


class TestLaplaceMixtureStatistics(unittest.TestCase):
    def test_matches_manual_two_component_mixture(self):
        means = torch.tensor([[0.0, 2.0]])
        log_prior = torch.log(torch.tensor([[0.25, 0.75]]))
        target = torch.tensor([1.0])
        result = laplace_mixture_statistics(means, log_prior, target, 1.0)
        likelihood = torch.exp(-(target[:, None] - means).abs()) / 2.0
        expected_evidence = (torch.exp(log_prior) * likelihood).sum(1)
        self.assertTrue(torch.allclose(result["nll"], -expected_evidence.log().mean()))
        self.assertTrue(torch.allclose(result["posterior"].sum(1), torch.ones(1)))

    def test_validation_rejects_bad_scale_and_target_shape(self):
        means = torch.zeros(2, 4)
        prior = torch.log_softmax(torch.zeros(2, 4), dim=1)
        with self.assertRaises(ValueError):
            laplace_mixture_statistics(means, prior, torch.zeros(2), 0.0)
        with self.assertRaises(ValueError):
            laplace_mixture_statistics(means, prior, torch.zeros(2, 2), 1.0)

    def test_nll_gradient_matches_prior_minus_posterior(self):
        raw = torch.tensor([[0.2, -0.3, 0.7, -0.1]], requires_grad=True)
        means = torch.tensor([[0.0, 0.4, 1.0, -0.2]], requires_grad=True)
        tau = 0.7
        log_prior = torch.log_softmax(raw / tau, dim=1)
        result = laplace_mixture_statistics(means, log_prior, torch.tensor([0.3]), 1.0)
        result["nll"].backward()
        expected = (log_prior.exp() - result["posterior"].detach()) / tau
        self.assertTrue(torch.allclose(raw.grad, expected, atol=1e-6, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
