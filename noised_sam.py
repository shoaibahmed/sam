import torch


class NoisedSAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, random_rho=None, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        if random_rho is None:
            random_rho = rho
        assert random_rho >= 0.0, f"Invalid random rho, should be non-negative: {random_rho}"

        defaults = dict(rho=rho, random_rho=random_rho, adaptive=adaptive, **kwargs)
        super(NoisedSAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.iteration = 0

    @torch.no_grad()
    def jitter_weights(self, zero_grad=False):
        # TODO: Ensure that the same logic works for multi-device setting due to randn -- assumes seeding
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        normalization_list = []
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["old_p"] = p.data.clone()
                if p.grad is None: continue
                
                # Compute a random perturbation to be added to the input
                rand_perturb = torch.randn_like(p)
                p.data = rand_perturb
                normalization_list.append(rand_perturb.norm(p=2).to(shared_device))
        
        self.iteration += 1
        if len(normalization_list) == 0:  # Should only happen for the first pass where the grads should be empty
            if self.iteration < 2:
                return
            raise RuntimeError(f"Normalization list empty even for the second pass. Please ensure that zero_grad is not called after the second step.")
        grad_norm = torch.norm(torch.stack(normalization_list), p=2)
        
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                
                # Add perturbation in the weights governed by the rho value
                scale = group["random_rho"] / (grad_norm + 1e-12)
                e_w = p.data * scale.to(p.data)
                p.data = self.state[p]["old_p"]  # get back to "w" from "delta"
                p.add_(e_w)  # climb to the random point within l_2 norm ball "w + delta"
        
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
