"""Numerically stable likelihoods for latent interaction structures."""

import torch


def laplace_mixture_statistics(component_means, log_prior, target, scale):
    """Return mixture NLL and the label-conditioned structure posterior.

    ``component_means`` and ``log_prior`` are ``[B, K]`` tensors.  The
    posterior is only a training/diagnostic quantity; it is not needed by the
    model forward pass and therefore does not create a label dependency there.
    """
    if not torch.is_tensor(component_means) or component_means.ndim != 2:
        raise ValueError("component_means must have shape [B,K]")
    if not torch.is_tensor(log_prior) or log_prior.shape != component_means.shape:
        raise ValueError("log_prior must have the same shape as component_means")
    if component_means.shape[1] <= 0:
        raise ValueError("component_means must contain at least one component")
    if not torch.is_tensor(target):
        target = torch.as_tensor(target, device=component_means.device)
    if target.ndim not in (1, 2) or (target.ndim == 2 and target.shape[1] != 1):
        raise ValueError("target must have shape [B] or [B,1]")
    if target.shape[0] != component_means.shape[0]:
        raise ValueError("target batch size must match component_means")
    b = torch.as_tensor(scale, dtype=torch.float32, device=component_means.device)
    if b.ndim != 0 or not torch.isfinite(b).item() or b.item() <= 0:
        raise ValueError("scale must be a finite positive scalar")

    mu = component_means.float()
    log_pi = log_prior.float()
    y = target.float().reshape(-1, 1)
    log_likelihood = -(y - mu).abs() / b - torch.log(2.0 * b)
    log_joint = log_pi + log_likelihood
    log_evidence = torch.logsumexp(log_joint, dim=-1)
    nll_per_sample = -log_evidence
    log_posterior = log_joint - log_evidence.unsqueeze(-1)
    posterior = log_posterior.exp()
    return {
        "nll": nll_per_sample.mean(),
        "nll_per_sample": nll_per_sample,
        "log_likelihood": log_likelihood,
        "log_posterior": log_posterior,
        "posterior": posterior,
    }
