"""No-grad inference must preserve loaded parameter flags, including mixed flags."""
import numpy as np
import pytest
import torch

import analyze_first_update as analysis


@pytest.mark.parametrize("fail_forward", [False, True])
def test_no_grad_preserves_mixed_parameter_flags_and_state(fail_forward):
    class TinyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.25), requires_grad=True)
            self.bias = torch.nn.Parameter(torch.tensor(0.5), requires_grad=False)
            self.seen = []
        def evaluate_actions(self, obs, actions, action_masks):
            self.seen.append((torch.is_grad_enabled(), {n:p.requires_grad for n,p in self.named_parameters()}))
            assert not torch.is_grad_enabled()
            if fail_forward:
                raise RuntimeError("synthetic forward failure")
            value = self.weight*obs[:,0]+self.bias
            return value, value, value
    policy = TinyPolicy()
    flags = {n:p.requires_grad for n,p in policy.named_parameters()}
    state = {n:p.detach().clone() for n,p in policy.state_dict().items()}
    rng = torch.get_rng_state().clone()
    outer_grad = torch.is_grad_enabled()
    arrays = {"observations":np.zeros((1,4,13012),dtype=np.float32),
              "actions":np.zeros((1,4,1),dtype=np.int64),
              "action_masks":np.ones((1,4,15),dtype=np.float32)}
    if fail_forward:
        with pytest.raises(RuntimeError,match="synthetic forward failure"):
            analysis.infer(policy,arrays)
    else:
        output = analysis.infer(policy,arrays)
        assert all(not value.requires_grad for value in output.values())
    assert policy.seen == [(False,flags)]
    assert {n:p.requires_grad for n,p in policy.named_parameters()} == flags
    assert all(p.grad is None for p in policy.parameters())
    assert all(torch.equal(state[n],p) for n,p in policy.state_dict().items())
    assert torch.equal(rng,torch.get_rng_state())
    assert torch.is_grad_enabled() == outer_grad
