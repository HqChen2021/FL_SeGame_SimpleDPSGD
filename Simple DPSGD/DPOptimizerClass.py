import torch
from torch.optim import Optimizer
from typing import Callable, List, Optional, Union


def _get_flat_grad_sample(p: torch.Tensor):
    """
    Return parameter's per sample gradients as a single tensor.
    By default, per sample gradients (``p.grad_sample``) are stored as one tensor per
    batch basis. Therefore, ``p.grad_sample`` is a single tensor if holds results from
    only one batch, and a list of tensors if gradients are accumulated over multiple
    steps. This is done to provide visibility into which sample belongs to which batch,
    and how many batches have been processed.
    This method returns per sample gradients as a single concatenated tensor, regardless
    of how many batches have been accumulated
    Args:
        p: Parameter tensor. Must have ``grad_sample`` attribute
    Returns:
        ``p.grad_sample`` if it's a tensor already, or a single tensor computed by
        concatenating every tensor in ``p.grad_sample`` if it's a list
    Raises:
        ValueError
            If ``p`` is missing ``grad_sample`` attribute
    """

    if not hasattr(p, "grad_sample"):
        raise ValueError(
            "Per sample gradient not found. Are you using GradSampleModule?"
        )
    if p.grad_sample is None:
        raise ValueError(
            "Per sample gradient is not initialized. Not updated in backward pass?"
        )
    if isinstance(p.grad_sample, torch.Tensor):
        return p.grad_sample
    elif isinstance(p.grad_sample, list):
        return torch.cat(p.grad_sample, dim=0)
    else:
        raise ValueError(f"Unexpected grad_sample type: {type(p.grad_sample)}")

def _generate_noise(std: float, reference: torch.Tensor, generator=None) -> torch.Tensor:
    """
    Generates noise according to a Gaussian distribution with mean 0
    Args:
        std: Standard deviation of the noise
        reference: The reference Tensor to get the appropriate shape and device
            for generating the noise
        generator: The PyTorch noise generator
    """
    zeros = torch.zeros(reference.shape, device=reference.device)
    if std == 0:
        return zeros
    return torch.normal(
        mean=0, std=std, size=reference.shape, device=reference.device, generator=generator)

class DPOptimizer(Optimizer):
    """
    https://github.com/pytorch/opacus/blob/3a7e8f82a8d02cc1ed227f2ef287865d904eff8d/opacus/optimizers/optimizer.py#L209
    ``torch.optim.Optimizer`` wrapper that adds additional functionality to clip per
    sample gradients and add Gaussian noise.
    Can be used with any ``torch.optim.Optimizer`` subclass as an underlying optimizer.
    ``DPOptimzer`` assumes that parameters over which it performs optimization belong
    to GradSampleModule and therefore have the ``grad_sample`` attribute.
    On a high level ``DPOptimizer``'s step looks like this:
    1) Aggregate ``p.grad_sample`` over all parameters to calculate per sample norms
    2) Clip ``p.grad_sample`` so that per sample norm is not above threshold
    3) Aggregate clipped per sample gradients into ``p.grad``
    4) Add Gaussian noise to ``p.grad`` calibrated to a given noise multiplier and
    max grad norm limit (``std = noise_multiplier * max_grad_norm``).
    5) Call underlying optimizer to perform optimization step
    Examples:
        >>> module = MyCustomModel()
        >>> optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
        >>> dp_optimzer = DPOptimizer(
        ...     optimizer=optimizer,
        ...     noise_multiplier=1.0,
        ...     max_grad_norm=1.0,
        ...     expected_batch_size=4,
        ... )
    """

    def __init__(self, optimizer: Optimizer, *, noise_multiplier: float, max_grad_norm: float,
                 expected_batch_size: Optional[int], loss_reduction: str = "mean", generator=None):
        """
        Args:
            optimizer: wrapped optimizer.
            noise_multiplier: noise multiplier
            max_grad_norm: max grad norm used for gradient clipping
            expected_batch_size: batch_size used for averaging gradients. When using
                Poisson sampling averaging denominator can't be inferred from the
                actual batch size. Required is ``loss_reduction="mean"``, ignored if
                ``loss_reduction="sum"``
            loss_reduction: Indicates if the loss reduction (for aggregating the gradients)
                is a sum or a mean operation. Can take values "sum" or "mean"
            generator: torch.Generator() object used as a source of randomness for
                the noise
        """
        if loss_reduction not in ("mean", "sum"):
            raise ValueError(f"Unexpected value for loss_reduction: {loss_reduction}")

        if loss_reduction == "mean" and expected_batch_size is None:
            raise ValueError(
                "You must provide expected batch size of the loss reduction is mean")

        self.original_optimizer = optimizer
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.loss_reduction = loss_reduction
        self.expected_batch_size = expected_batch_size
        self.generator = generator
        self.param_groups = optimizer.param_groups
        self.defaults = self.original_optimizer.defaults
        self.state = optimizer.state
        self.params = self.get_params()

        for p in self.params:
            p.summed_grad = None

    # def __init__(self, max_grad_norm, noise_multiplier, minibatch_size, microbatch_size, *args, **kwargs):
    #     super(DPOptimizerClass, self).__init__(*args, **kwargs)
    #     self.max_grad_norm = max_grad_norm
    #     self.noise_multiplier = noise_multiplier
    #     self.microbatch_size = microbatch_size
    #     self.minibatch_size = minibatch_size
    #
    #     # SGD,Adam has attribute param_groups
    #     for group in self.param_groups:
    #         # add a new element into group, 'params' is the keys of group in self.param_groups,
    #         # the value is parameters of network
    #         group['accum_grads'] = [torch.zeros_like(param.data) if param.requires_grad else None for param in group['params']]

    def get_params(self):
        """
        Return all parameters controlled by the optimizer
        Args:
            optimizer: optimizer
        Returns:
            Flat list of parameters from all ``param_groups``
        """
        ret = []
        for param_group in self.original_optimizer.param_groups:
            ret += [p for p in param_group["params"] if p.requires_grad]
        return ret

    @property
    def grad_samples(self) -> List[torch.Tensor]:
        """
        Returns a flat list of per sample gradient tensors (one per parameter)
        """
        ret = []
        for p in self.params:
            ret.append(_get_flat_grad_sample(p))
        return ret

    def clip_and_accumulate(self):
        """
        Performs gradient clipping.
        Stores clipped and aggregated gradients into `p.summed_grad```
        """
        per_param_norms = [g.view(len(g), -1).norm(2, dim=-1) for g in self.grad_samples]
        per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)
        per_sample_clip_factor = (
                self.max_grad_norm / (per_sample_norms + 1e-6)).clamp(max=1.0)

        for p in self.params:
            grad_sample = _get_flat_grad_sample(p)
            grad = torch.einsum("i,i...", per_sample_clip_factor, grad_sample)

            if p.summed_grad is not None:
                p.summed_grad += grad
            else:
                p.summed_grad = grad

    def add_noise(self):
        """
        Adds noise to clipped gradients. Stores clipped and noised result in ``p.grad``
        """
        for p in self.params:
            noise = _generate_noise(
                std=self.noise_multiplier * self.max_grad_norm,
                reference=p.summed_grad,
                generator=self.generator,
            )
            p.grad = (p.summed_grad + noise).view_as(p.grad)

    def scale_grad(self):
        """
        Applies given ``loss_reduction`` to ``p.grad``.
        Does nothing if ``loss_reduction="sum"``. Divides gradients by
        ``self.expected_batch_size`` if ``loss_reduction="mean"``
        """
        if self.loss_reduction == "mean":
            for p in self.params:
                p.grad /= self.expected_batch_size

    def zero_grad(self, set_to_none: bool = False):
        """
        Clear gradients.
        Clears ``p.grad``, ``p.grad_sample`` and ``p.summed_grad`` for all of it's parameters
        Notes:
            ``set_to_none`` argument only affects ``p.grad``. ``p.grad_sample`` and
            ``p.summed_grad`` is never zeroed out and always set to None.
            Normal grads can do this, because their shape is always the same.
            Grad samples do not behave like this, as we accumulate gradients from different
            batches in a list
        Args:
            set_to_none: instead of setting to zero, set the grads to None. (only
            affects regular gradients. Per sample gradients are always set to None)
        """
        for p in self.params:
            p.grad_sample = None
            p.summed_grad = None
        self.original_optimizer.zero_grad(set_to_none)

    def step(self):
        self.clip_and_accumulate()
        self.add_noise()
        self.scale_grad()
        self.original_optimizer.step()

    def __repr__(self):
        return self.original_optimizer.__repr__()

    def state_dict(self):
        return self.original_optimizer.state_dict()

    def load_state_dict(self, state_dict) -> None:
        self.original_optimizer.load_state_dict(state_dict)
