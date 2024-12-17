import torch
import torch.nn as nn
import torch.distributions as dists
import utils
import numpy as np


device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')


class LangevinSampler(nn.Module):
    def __init__(self, dim, n_steps=10, approx=False, multi_hop=False, fixed_proposal=False, temp=2., step_size=0.2,
                 mh=True):
        super().__init__()
        self.dim = dim
        self.n_steps = n_steps
        self._ar = 0.
        self._mt = 0.
        self._pt = 0.
        self._hops = 0.
        self._phops = 0.
        self.approx = approx
        self.fixed_proposal = fixed_proposal
        self.multi_hop = multi_hop
        self.temp = temp
        self.step_size = step_size
        self.entropy = False

        if approx:
            self.diff_fn = lambda x, m: utils.approx_difference_function(x, m) / self.temp
        else:
            self.diff_fn = lambda x, m: utils.difference_function(x, m) / self.temp



        self.mh = mh
        self.a_s = []
        self.hops = []


    def step(self, x, model):

        x_cur = x
        # print('instep')
        # print(x_cur)

        m_terms = []
        prop_terms = []

        EPS = 1e-10
        for i in range(self.n_steps):
            forward_delta = self.diff_fn(x_cur, model)
            term2 = 1. / (2 * self.step_size)  # for binary {0,1}, the L2 norm is always 1
            flip_prob = torch.exp(forward_delta - term2) / (torch.exp(forward_delta - term2) + 1)

            rr = torch.rand_like(x_cur)
            # print("rr:", rr)
            ind = (rr < flip_prob) * 1
            x_delta = (1. - x_cur) * ind + x_cur * (1. - ind)
            # print("x_delta", x_delta)

            if self.mh:
                probs = flip_prob * ind + (1 - flip_prob) * (1. - ind)
                lp_forward = torch.sum(torch.log(probs + EPS), dim=-1)

                reverse_delta = self.diff_fn(x_delta, model)
                flip_prob = torch.exp(reverse_delta - term2) / (torch.exp(reverse_delta - term2) + 1)
                probs = flip_prob * ind + (1 - flip_prob) * (1. - ind)
                lp_reverse = torch.sum(torch.log(probs + EPS), dim=-1)

                m_term = (model(x_delta).squeeze() - model(x_cur).squeeze())
                la = m_term + lp_reverse - lp_forward
                a = (la.exp() > torch.rand_like(la)).float()
                self.a_s.append(a.mean().item())
                x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])
            else:
                x_cur = x_delta

        return x_cur


class ELangevinSampler(nn.Module):
    def __init__(self, dim, n_steps=10, approx=False, multi_hop=False, fixed_proposal=False, temp=2., alpha=0.2,
                 alpha_a=0.5, mh=True, eta=10 ** 4):
        super().__init__()
        self.dim = dim
        self.n_steps = n_steps
        self._ar = 0.
        self._mt = 0.
        self._pt = 0.
        self._hops = 0.
        self._phops = 0.
        self.approx = approx
        self.fixed_proposal = fixed_proposal
        self.multi_hop = multi_hop
        self.temp = temp
        self.alpha = alpha
        self.alpha_a = alpha_a
        self.eta = eta  # flatness
        self.entropy = True
        self.gibbs_like_update = False

        if approx:
            self.diff_fn = lambda x, m: utils.approx_difference_function(x, m) / self.temp
        else:
            self.diff_fn = lambda x, m: utils.difference_function(x, m) / self.temp

        self.agrad = lambda eta, x, x_a: utils.auxillary_gradient_function(eta, x, x_a)

        self.mh = mh
        self.a_s = []
        self.hops = []

    def step(self, x, x_a, model):

        x_cur = x

        x_a_cur = x_a

        m_terms = []
        prop_terms = []

        EPS = 1e-10
        for i in range(self.n_steps):

            grad_x_cur = self.diff_fn(x_cur, model)
            grad_x_cur_a = self.agrad(self.eta, x_cur, x_a_cur)
            grad_x_cur_modified = grad_x_cur - (grad_x_cur_a / self.temp) * -(2 * x_cur - 1)

            term2 = 1. / (2 * self.alpha)  # for binary {0,1}, the L2 norm is always 1
            flip_prob = torch.exp(grad_x_cur_modified - term2) / (torch.exp(grad_x_cur_modified - term2) + 1)  # softmax

            rr = torch.rand_like(x_cur)
            # print("rr:", rr)
            ind = (rr < flip_prob) * 1
            x_delta = (1. - x_cur) * ind + x_cur * (1. - ind)
            x_delta_a = x_a_cur + (self.alpha_a / 2.) * grad_x_cur_a + (
                    (1 * self.alpha_a) ** 0.5) * torch.randn_like(x_a_cur, device=x_a_cur.device)

            if self.mh:
                probs = flip_prob * ind + (1 - flip_prob) * (1. - ind)
                lp_forward = torch.sum(torch.log(probs + EPS), dim=-1)

                grad_x_delta = self.diff_fn(x_delta, model)
                grad_x_delta_a = self.agrad(self.eta, x_delta, x_delta_a)

                grad_x_delta_modified = grad_x_delta - (grad_x_delta_a / self.temp) * -(2 * x_delta - 1)

                flip_prob = torch.exp(grad_x_delta_modified - term2) / (torch.exp(grad_x_delta_modified - term2) + 1)
                probs = flip_prob * ind + (1 - flip_prob) * (1. - ind)
                lp_reverse = torch.sum(torch.log(probs + EPS), dim=-1)

                m_term = (model(x_delta).squeeze() - torch.sum(torch.pow(x_delta - x_delta_a, 2), dim=1) / (
                        2 * self.eta)) - (
                                 model(x_cur).squeeze() - torch.sum(torch.pow(x_cur - x_a_cur, 2), dim=1) / (
                                 2 * self.eta))

                additional_m_term_reverse = torch.sum(
                    torch.pow(x_a_cur - (x_delta_a + (self.alpha_a / 2) * grad_x_delta_a), 2), dim=1) / (
                                                    -2 * self.alpha_a)
                additional_m_term_forward = torch.sum(
                    torch.pow(x_delta_a - (x_a_cur + (self.alpha_a / 2) * grad_x_cur_a), 2), dim=1) / (
                                                    -2 * self.alpha_a)

                la = m_term + additional_m_term_reverse - additional_m_term_forward + lp_reverse - lp_forward
                a = (la.exp() > torch.rand_like(la)).float()
                self.a_s.append(a.mean().item())
                x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])
                x_a_cur = x_delta_a * a[:, None] + x_a_cur * (1. - a[:, None])

            else:
                x_cur = x_delta  # Langevin Update
                x_a_cur = x_delta_a

        return (x_cur, x_a_cur)  # Non-GLU Return


class ELangevinSamplerwithGLU(nn.Module):
    def __init__(self, dim, n_steps=10, approx=False, multi_hop=False, fixed_proposal=False, temp=2., alpha=0.2,
                 mh=True, eta=10 ** 4):
        super().__init__()
        self.dim = dim
        self.n_steps = n_steps
        self._ar = 0.
        self._mt = 0.
        self._pt = 0.
        self._hops = 0.
        self._phops = 0.
        self.approx = approx
        self.fixed_proposal = fixed_proposal
        self.multi_hop = multi_hop
        self.temp = temp
        self.alpha = alpha
        self.eta = eta  # flatness
        self.entropy = True

        self.glu_flag = 0

        if approx:
            self.diff_fn = lambda x, m: utils.approx_difference_function(x, m) / self.temp
        else:
            self.diff_fn = lambda x, m: utils.difference_function(x, m) / self.temp

        self.agrad = lambda eta, x, x_a: utils.auxillary_gradient_function(eta, x, x_a)

        self.mh = mh
        self.a_s = []
        self.hops = []

    def step(self, x, x_a, model):

        x_cur = x
        x_a_cur = x_a

        EPS = 1e-10
        for i in range(self.n_steps):

            if (self.glu_flag == 0):

                grad_x_cur = self.diff_fn(x_cur, model)
                grad_x_cur_a = self.agrad(self.eta, x_cur, x_a_cur)
                grad_x_cur_modified = grad_x_cur - (grad_x_cur_a / self.temp) * -(2 * x_cur - 1)

                term2 = 1. / (2 * self.alpha)  # for binary {0,1}, the L2 norm is always 1
                flip_prob = torch.exp(grad_x_cur_modified - term2) / (
                        torch.exp(grad_x_cur_modified - term2) + 1)  # softmax

                rr = torch.rand_like(x_cur)
                ind = (rr < flip_prob) * 1
                x_delta = (1. - x_cur) * ind + x_cur * (1. - ind)

                if self.mh:
                    probs = flip_prob * ind + (1 - flip_prob) * (1. - ind)
                    lp_forward = torch.sum(torch.log(probs + EPS), dim=-1)

                    grad_x_delta = self.diff_fn(x_delta, model)
                    grad_x_delta_a = self.agrad(self.eta, x_delta, x_a_cur)

                    grad_x_delta_modified = grad_x_delta - (grad_x_delta_a / self.temp) * -(2 * x_delta - 1)

                    flip_prob = torch.exp(grad_x_delta_modified - term2) / (
                            torch.exp(grad_x_delta_modified - term2) + 1)
                    probs = flip_prob * ind + (1 - flip_prob) * (1. - ind)
                    lp_reverse = torch.sum(torch.log(probs + EPS), dim=-1)

                    m_term = (model(x_delta).squeeze() - torch.sum(torch.pow(x_delta - x_a_cur, 2), dim=1) / (
                            2 * self.eta)) - (
                                     model(x_cur).squeeze() - torch.sum(torch.pow(x_cur - x_a_cur, 2), dim=1) / (
                                     2 * self.eta))

                    la = m_term + lp_reverse - lp_forward
                    a = (la.exp() > torch.rand_like(la)).float()
                    self.a_s.append(a.mean().item())
                    x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])

                else:
                    x_cur = x_delta

                self.glu_flag = 1
                return (x_cur, x_a_cur)  # Update x, not x_a

            else:

                x_a_cur = x_cur + np.sqrt(self.eta) * torch.randn_like(x_cur, device=x_cur.device)

                self.glu_flag = 0
                return (x_cur, x_a_cur)  # Update x_a, not x


# Gibbs-With-Gradients for binary data
class DiffSampler(nn.Module):
    def __init__(self, dim, n_steps=10, approx=False, multi_hop=False, fixed_proposal=False, temp=2., step_size=1.0):
        super().__init__()
        self.dim = dim
        self.n_steps = n_steps
        self._ar = 0.
        self._mt = 0.
        self._pt = 0.
        self._hops = 0.
        self._phops = 0.
        self.approx = approx
        self.fixed_proposal = fixed_proposal
        self.multi_hop = multi_hop
        self.temp = temp
        self.step_size = step_size
        self.entropy = False
        if approx:
            self.diff_fn = lambda x, m: utils.approx_difference_function(x, m) / self.temp
        else:
            self.diff_fn = lambda x, m: utils.difference_function(x, m) / self.temp

    def step(self, x, model):

        x_cur = x
        a_s = []
        m_terms = []
        prop_terms = []

        if self.multi_hop:
            if self.fixed_proposal:
                delta = self.diff_fn(x, model)
                cd = dists.Bernoulli(probs=delta.sigmoid() * self.step_size)
                for i in range(self.n_steps):
                    changes = cd.sample()
                    x_delta = (1. - x_cur) * changes + x_cur * (1. - changes)
                    la = (model(x_delta).squeeze() - model(x_cur).squeeze())
                    a = (la.exp() > torch.rand_like(la)).float()
                    x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])
                    a_s.append(a.mean().item())
                self._ar = np.mean(a_s)
            else:
                for i in range(self.n_steps):
                    forward_delta = self.diff_fn(x_cur, model)
                    cd_forward = dists.Bernoulli(logits=(forward_delta * 2 / self.temp))
                    changes = cd_forward.sample()

                    lp_forward = cd_forward.log_prob(changes).sum(-1)

                    x_delta = (1. - x_cur) * changes + x_cur * (1. - changes)

                    reverse_delta = self.diff_fn(x_delta, model)
                    cd_reverse = dists.Bernoulli(logits=(reverse_delta * 2 / self.temp))

                    lp_reverse = cd_reverse.log_prob(changes).sum(-1)

                    m_term = (model(x_delta).squeeze() - model(x_cur).squeeze())
                    la = m_term + lp_reverse - lp_forward
                    a = (la.exp() > torch.rand_like(la)).float()
                    x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])
                    a_s.append(a.mean().item())
                    m_terms.append(m_term.mean().item())
                    prop_terms.append((lp_reverse - lp_forward).mean().item())
                self._ar = np.mean(a_s)
                self._mt = np.mean(m_terms)
                self._pt = np.mean(prop_terms)
        else:
            if self.fixed_proposal:
                delta = self.diff_fn(x, model)
                cd = dists.OneHotCategorical(logits=delta)
                for i in range(self.n_steps):
                    changes = cd.sample()

                    x_delta = (1. - x_cur) * changes + x_cur * (1. - changes)
                    la = (model(x_delta).squeeze() - model(x_cur).squeeze())
                    a = (la.exp() > torch.rand_like(la)).float()
                    x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])
                    a_s.append(a.mean().item())
                self._ar = np.mean(a_s)
            else:
                for i in range(self.n_steps):
                    forward_delta = self.diff_fn(x_cur, model)
                    cd_forward = dists.OneHotCategorical(logits=forward_delta)
                    changes = cd_forward.sample()

                    lp_forward = cd_forward.log_prob(changes)

                    x_delta = (1. - x_cur) * changes + x_cur * (1. - changes)

                    reverse_delta = self.diff_fn(x_delta, model)
                    cd_reverse = dists.OneHotCategorical(logits=reverse_delta)

                    lp_reverse = cd_reverse.log_prob(changes)

                    m_term = (model(x_delta).squeeze() - model(x_cur).squeeze())
                    la = m_term + lp_reverse - lp_forward
                    a = (la.exp() > torch.rand_like(la)).float()
                    x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])

        return x_cur


# Gibbs-With-Gradients variant which proposes multiple flips per step
class MultiDiffSampler(nn.Module):
    def __init__(self, dim, n_steps=10, approx=False, temp=1., n_samples=1):
        super().__init__()
        self.dim = dim
        self.n_steps = n_steps
        self._ar = 0.
        self._mt = 0.
        self._pt = 0.
        self._hops = 0.
        self._phops = 0.
        self.approx = approx
        self.temp = temp
        self.n_samples = n_samples
        self.entropy = False
        if approx:
            self.diff_fn = lambda x, m: utils.approx_difference_function(x, m) / self.temp
        else:
            self.diff_fn = lambda x, m: utils.difference_function(x, m) / self.temp
        self.a_s = []
        self.hops = []

    def step(self, x, model):

        x_cur = x
        a_s = []
        m_terms = []
        prop_terms = []

        for i in range(self.n_steps):
            forward_delta = self.diff_fn(x_cur, model)
            cd_forward = dists.OneHotCategorical(logits=forward_delta)
            changes_all = cd_forward.sample((self.n_samples,))

            lp_forward = cd_forward.log_prob(changes_all).sum(0)

            changes = (changes_all.sum(0) > 0.).float()

            x_delta = (1. - x_cur) * changes + x_cur * (1. - changes)
            # self._phops = (x_delta != x).float().sum(-1).mean().item()
            cur_hops = (x_cur[0] != x_delta[0]).float().sum(-1).item()
            self.hops.append(cur_hops)

            reverse_delta = self.diff_fn(x_delta, model)
            cd_reverse = dists.OneHotCategorical(logits=reverse_delta)

            lp_reverse = cd_reverse.log_prob(changes_all).sum(0)

            m_term = (model(x_delta).squeeze() - model(x_cur).squeeze())
            la = m_term + lp_reverse - lp_forward
            a = (la.exp() > torch.rand_like(la)).float()
            x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])
            self.a_s.append(a.mean().item())
            m_terms.append(m_term.mean().item())
            prop_terms.append((lp_reverse - lp_forward).mean().item())
        self._ar = np.mean(a_s)
        self._mt = np.mean(m_terms)
        self._pt = np.mean(prop_terms)
        # print(self._ar)
        self._hops = (x != x_cur).float().sum(-1).mean().item()
        return x_cur


class PerDimGibbsSampler(nn.Module):
    def __init__(self, dim, n, rand=False):
        super().__init__()
        self.dim = dim
        self.n_steps = n
        self.changes = torch.zeros((dim,))
        self.change_rate = 0.
        self.p = nn.Parameter(torch.zeros((dim,)))
        self._i = 0
        self._ar = 0.
        self._hops = 0.
        self._phops = 1.
        self.rand = rand
        self.entropy = False

    def step(self, x, model):
        sample = x.clone()
        for i in range(self.n_steps):
            lp_keep = model(sample).squeeze()
            if self.rand:
                changes = dists.OneHotCategorical(logits=torch.zeros((self.dim,))).sample((x.size(0),)).to(x.device)
            else:
                changes = torch.zeros((x.size(0), self.dim)).to(x.device)
                changes[:, self._i] = 1.

            sample_change = (1. - changes) * sample + changes * (1. - sample)

            lp_change = model(sample_change).squeeze()

            lp_update = lp_change - lp_keep
            update_dist = dists.Bernoulli(logits=lp_update)
            updates = update_dist.sample()
            sample = sample_change * updates[:, None] + sample * (1. - updates[:, None])
            self.changes[self._i] = updates.mean()
            self._i = (self._i + 1) % self.dim
            self._hops = (x != sample).float().sum(-1).mean().item()
            self._ar = self._hops
        return sample

    def logp_accept(self, xhat, x, model):
        # only true if xhat was generated from self.step(x, model)
        return 0.


class PerDimMetropolisSampler(nn.Module):
    def __init__(self, dim, n_out, rand=False):
        super().__init__()
        self.dim = dim
        self.n_out = n_out
        self.changes = torch.zeros((dim,))
        self.change_rate = 0.
        self.p = nn.Parameter(torch.zeros((dim,)))
        self._i = 0
        self._j = 0
        self._ar = 0.
        self._hops = 0.
        self._phops = 0.
        self.rand = rand
        self.entropy = False

    def step(self, x, model):
        if self.rand:
            i = np.random.randint(0, self.dim)
        else:
            i = self._i

        logits = []
        ndim = x.size(-1)

        for k in range(ndim):
            sample = x.clone()
            sample_i = torch.zeros((ndim,))
            sample_i[k] = 1.
            sample[:, i, :] = sample_i
            lp_k = model(sample).squeeze()
            logits.append(lp_k[:, None])
        logits = torch.cat(logits, 1)
        dist = dists.OneHotCategorical(logits=logits)
        updates = dist.sample()
        sample = x.clone()
        sample[:, i, :] = updates
        self._i = (self._i + 1) % self.dim
        self._hops = ((x != sample).float().sum(-1) / 2.).sum(-1).mean().item()
        self._ar = self._hops
        return sample

    def logp_accept(self, xhat, x, model):
        # only true if xhat was generated from self.step(x, model)
        return 0.


class PerDimLB(nn.Module):
    def __init__(self, dim, rand=False):
        super().__init__()
        self.dim = dim
        self.changes = torch.zeros((dim,))
        self.change_rate = 0.
        self.p = nn.Parameter(torch.zeros((dim,)))
        self._i = 0
        self._j = 0
        self._ar = 0.
        self._hops = 0.
        self._phops = 0.
        self.rand = rand
        self.entropy = False

    def step(self, x, model):
        logits = []
        ndim = x.size(-1)
        fx = model(x).squeeze()
        for k in range(ndim):
            sample = x.clone()
            sample[:, k] = 1 - sample[:, k]
            lp_k = (model(sample).squeeze() - fx) / 2.
            logits.append(lp_k[:, None])
        logits = torch.cat(logits, 1)
        Z_forward = torch.sum(torch.exp(logits), dim=-1)
        dist = dists.OneHotCategorical(logits=logits)
        changes = dist.sample()
        x_delta = (1. - x) * changes + x * (1. - changes)
        fx_delta = model(x_delta)
        logits = []
        for k in range(ndim):
            sample = x_delta.clone()
            sample[:, k] = 1 - sample[:, k]
            lp_k = (model(sample).squeeze() - fx_delta) / 2.
            logits.append(lp_k[:, None])
        logits = torch.cat(logits, 1)
        Z_reverse = torch.sum(torch.exp(logits), dim=-1)
        la = Z_forward / Z_reverse
        a = (la > torch.rand_like(la)).float()
        x = x_delta * a[:, None] + x * (1. - a[:, None])
        # a_s.append(a.mean().item())
        # self._ar = np.mean(a_s)
        return x

    def logp_accept(self, xhat, x, model):
        # only true if xhat was generated from self.step(x, model)
        return 0.


# Gibbs-With-Gradients for categorical data
class DiffSamplerMultiDim(nn.Module):
    def __init__(self, dim, n_steps=10, approx=False, temp=1.):
        super().__init__()
        self.dim = dim
        self.n_steps = n_steps
        self._ar = 0.
        self._mt = 0.
        self._pt = 0.
        self._hops = 0.
        self._phops = 0.
        self.approx = approx
        self.temp = temp
        self.entropy = False
        if approx:
            self.diff_fn = lambda x, m: utils.approx_difference_function_multi_dim(x, m) / self.temp
        else:
            self.diff_fn = lambda x, m: utils.difference_function_multi_dim(x, m) / self.temp

    def step(self, x, model):

        x_cur = x
        a_s = []
        m_terms = []
        prop_terms = []

        for i in range(self.n_steps):
            constant = 1.
            forward_delta = self.diff_fn(x_cur, model)

            # make sure we dont choose to stay where we are!
            forward_logits = forward_delta - constant * x_cur
            # print(forward_logits)
            cd_forward = dists.OneHotCategorical(logits=forward_logits.view(x_cur.size(0), -1))
            changes = cd_forward.sample()
            # print(x_cur.shape,forward_delta.shape,changes.shape)
            # exit()
            # compute probability of sampling this change
            lp_forward = cd_forward.log_prob(changes)
            # reshape to (bs, dim, nout)
            changes_r = changes.view(x_cur.size())
            # get binary indicator (bs, dim) indicating which dim was changed
            changed_ind = changes_r.sum(-1)
            # mask out cuanged dim and add in the change
            x_delta = x_cur.clone() * (1. - changed_ind[:, :, None]) + changes_r

            reverse_delta = self.diff_fn(x_delta, model)
            reverse_logits = reverse_delta - constant * x_delta
            cd_reverse = dists.OneHotCategorical(logits=reverse_logits.view(x_delta.size(0), -1))
            reverse_changes = x_cur * changed_ind[:, :, None]

            lp_reverse = cd_reverse.log_prob(reverse_changes.view(x_delta.size(0), -1))

            m_term = (model(x_delta).squeeze() - model(x_cur).squeeze())
            la = m_term + lp_reverse - lp_forward
            a = (la.exp() > torch.rand_like(la)).float()
            x_cur = x_delta * a[:, None, None] + x_cur * (1. - a[:, None, None])
            a_s.append(a.mean().item())
            m_terms.append(m_term.mean().item())
            prop_terms.append((lp_reverse - lp_forward).mean().item())
        self._ar = np.mean(a_s)
        self._mt = np.mean(m_terms)
        self._pt = np.mean(prop_terms)

        self._hops = (x != x_cur).float().sum(-1).sum(-1).mean().item()
        return x_cur


class GibbsSampler(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.changes = torch.zeros((dim,))
        self.change_rate = 0.
        self.p = nn.Parameter(torch.zeros((dim,)))
        self.entropy = False

    def step(self, x, model):
        sample = x.clone()
        for i in range(self.dim):
            lp_keep = model(sample).squeeze()

            xi_keep = sample[:, i]
            xi_change = 1. - xi_keep
            sample_change = sample.clone()
            sample_change[:, i] = xi_change

            lp_change = model(sample_change).squeeze()

            lp_update = lp_change - lp_keep
            update_dist = dists.Bernoulli(logits=lp_update)
            updates = update_dist.sample()
            sample = sample_change * updates[:, None] + sample * (1. - updates[:, None])
            self.changes[i] = updates.mean()
        return sample

    def logp_accept(self, xhat, x, model):
        # only true if xhat was generated from self.step(x, model)
        return 0.


class LangevinSamplerOrdinal(nn.Module):
    def __init__(
            self,
            dim,
            bal,
            max_val=3,
            n_steps=10,
            multi_hop=False,
            temp=1.0,
            step_size=0.2,
            mh=True,
            device=None,
    ):
        super().__init__()
        self.dim = dim
        self.n_steps = n_steps
        self._ar = 0.0
        self._mt = 0.0
        self._pt = 0.0
        self._hops = 0.0
        self._phops = 0.0
        self.multi_hop = multi_hop
        self.temp = temp
        # rbm sampling: accpt prob is about 0.5 with lr = 0.2, update 16 dims per step (total 784 dims). ising sampling: accept prob 0.5 with lr=0.2
        # ising learning: accept prob=0.7 with lr=0.2
        # ebm: statistic mnist: accept prob=0.45 with lr=0.2
        self.a_s = []
        self.bal = bal
        self.mh = mh
        self.max_val = max_val  ### number of classes in each dimension
        self.step_size = (step_size * self.max_val) ** (self.dim ** 2)
        # self.step_size = step_size
        self.entropy = False

    def get_grad(self, x, model):
        x = x.requires_grad_()
        out = model(x)
        gx = torch.autograd.grad(out.sum(), x)[0]
        return gx.detach()

    def _calc_logits(self, x_cur, grad):
        # creating the tensor of discrete values to compute the probabilities for
        batch_size = x_cur.shape[0]
        disc_values = torch.tensor([i for i in range(self.max_val)])[None, None, :]
        disc_values = disc_values.repeat((batch_size, self.dim, 1)).to(x_cur.device)
        term1 = torch.zeros((batch_size, self.dim, self.max_val))
        term2 = torch.zeros((batch_size, self.dim, self.max_val))
        x_expanded = x_cur[:, :, None].repeat((1, 1, self.max_val)).to(x_cur.device)
        grad_expanded = grad[:, :, None].repeat((1, 1, self.max_val)).to(x_cur.device)
        term1 = self.bal * grad_expanded * (disc_values - x_expanded)
        term2 = (disc_values - x_expanded) ** 2 * (1 / (2 * self.step_size))
        return term1 - term2

    def step(self, x, model, use_dula=False):
        """
        input x : bs * dim, every dim contains a integer of 0 to (num_cls-1)
        """
        x_cur = x
        m_terms = []
        prop_terms = []

        EPS = 1e-10
        for i in range(self.n_steps):
            # batch size X dim
            grad = self.get_grad(x_cur.float(), model) / self.temp
            logits = self._calc_logits(x_cur, grad)
            # print(f"x_cur shape: {x_cur.shape}, dtype: {x_cur.dtype}")
            # print(f"logits shape: {logits.shape}, logits values: {logits}")
            cat_dist = torch.distributions.categorical.Categorical(logits=logits)
            x_delta = cat_dist.sample()

            if self.mh:
                lp_forward = torch.sum(cat_dist.log_prob(x_delta), dim=1)
                grad_delta = self.get_grad(x_delta.float(), model) / self.temp

                logits_reverse = self._calc_logits(x_delta, grad_delta)

                cat_dist_delta = torch.distributions.categorical.Categorical(
                    logits=logits_reverse
                )

                lp_reverse = torch.sum(cat_dist_delta.log_prob(x_cur), dim=1)

                m_term = model(x_delta).squeeze() - model(x_cur).squeeze()
                la = m_term + lp_reverse - lp_forward
                a = (la.exp() > torch.rand_like(la)).float()
                self.a_s.append(a.mean().item())
                if use_dula:
                    x_cur = x_delta
                else:
                    x_cur = x_delta * a[:, None] + x_cur * (1.0 - a[:, None])
            else:
                x_cur = x_delta

            x_cur = x_cur.long()
        return x_cur


class ELangevinSamplerOrdinal(nn.Module):
    def __init__(
            self,
            dim,
            bal,
            max_val=3,
            n_steps=10,
            multi_hop=False,
            temp=1.0,
            step_size=0.2,
            step_size_a=0.1,
            eta=1,
            mh=True,
            device=None,
    ):
        super().__init__()
        self.dim = dim
        self.n_steps = n_steps
        self._ar = 0.0
        self._mt = 0.0
        self._pt = 0.0
        self._hops = 0.0
        self._phops = 0.0
        self.multi_hop = multi_hop
        self.temp = temp
        # rbm sampling: accpt prob is about 0.5 with lr = 0.2, update 16 dims per step (total 784 dims). ising sampling: accept prob 0.5 with lr=0.2
        # ising learning: accept prob=0.7 with lr=0.2
        # ebm: statistic mnist: accept prob=0.45 with lr=0.2
        self.a_s = []
        self.bal = bal
        self.mh = mh
        self.max_val = max_val  ### number of classes in each dimension
        self.step_size = (step_size * self.max_val) ** (self.dim ** 2)

        self.step_size_a = step_size_a
        self.eta = eta
        self.entropy = True

    def agrad(self, x, x_a):
        return (x - x_a) / self.eta

    def get_grad(self, x, model):
        x = x.requires_grad_()
        out = model(x)
        gx = torch.autograd.grad(out.sum(), x)[0]
        return gx.detach()

    def _calc_logits(self, x_cur, grad):
        # creating the tensor of discrete values to compute the probabilities for
        batch_size = x_cur.shape[0]
        disc_values = torch.tensor([i for i in range(self.max_val)])[None, None, :]
        disc_values = disc_values.repeat((batch_size, self.dim, 1)).to(x_cur.device)
        term1 = torch.zeros((batch_size, self.dim, self.max_val))
        term2 = torch.zeros((batch_size, self.dim, self.max_val))
        x_expanded = x_cur[:, :, None].repeat((1, 1, self.max_val)).to(x_cur.device)
        grad_expanded = grad[:, :, None].repeat((1, 1, self.max_val)).to(x_cur.device)
        term1 = self.bal * grad_expanded * (disc_values - x_expanded)
        term2 = (disc_values - x_expanded) ** 2 * (1 / (2 * self.step_size))
        return term1 - term2

    def step(self, x, x_a, model, use_dula=False):
        """
        input x : bs * dim, every dim contains a integer of 0 to (num_cls-1)
        """
        x_cur = x
        x_a_cur = x_a
        m_terms = []
        prop_terms = []

        EPS = 1e-10
        for i in range(self.n_steps):
            # batch size X dim
            grad = self.get_grad(x_cur.float(), model) / self.temp
            grad_x_cur_a = self.agrad(x_cur.float(), x_a_cur.float()) / self.temp
            grad_x_cur_modified = grad - grad_x_cur_a

            logits = self._calc_logits(x_cur, grad_x_cur_modified)
            cat_dist = torch.distributions.categorical.Categorical(logits=logits)
            x_delta = cat_dist.sample()
            x_delta_a = x_a_cur + (self.step_size_a / 2.) * grad_x_cur_a + (
                    (1 * self.step_size_a) ** 0.5) * torch.randn_like(x_a_cur, device=x_a_cur.device)

            if self.mh:
                lp_forward = torch.sum(cat_dist.log_prob(x_delta), dim=1)
                grad_delta = self.get_grad(x_delta.float(), model) / self.temp
                grad_delta_a = self.agrad(x_delta.float(), x_delta_a) / self.temp
                grad_delta_modified = grad_delta - grad_delta_a

                logits_reverse = self._calc_logits(x_delta, grad_delta_modified)

                cat_dist_delta = torch.distributions.categorical.Categorical(
                    logits=logits_reverse
                )
                lp_reverse = torch.sum(cat_dist_delta.log_prob(x_cur), dim=1)

                m_term = (model(x_delta).squeeze() - torch.sum(torch.pow(x_delta - x_delta_a, 2), dim=1) / (
                        2 * self.eta)) - (
                                 model(x_cur).squeeze() - torch.sum(torch.pow(x_cur - x_a_cur, 2), dim=1) / (
                                 2 * self.eta))

                additional_m_term_reverse = torch.sum(
                    torch.pow(x_a_cur - (x_delta_a + (self.step_size_a / 2) * grad_delta_a), 2), dim=1) / (
                                                    -2 * self.step_size_a)
                additional_m_term_forward = torch.sum(
                    torch.pow(x_delta_a - (x_a_cur + (self.step_size_a / 2) * grad_x_cur_a), 2), dim=1) / (
                                                    -2 * self.step_size_a)

                la = m_term + additional_m_term_reverse - additional_m_term_forward + lp_reverse - lp_forward

                a = (la.exp() > torch.rand_like(la)).float()
                self.a_s.append(a.mean().item())

                x_cur = x_delta * a[:, None] + x_cur * (1.0 - a[:, None])
                x_a_cur = x_delta_a * a[:, None] + x_a_cur * (1. - a[:, None])
            else:
                x_cur = x_delta
                x_a_cur = x_delta_a

            x_cur = x_cur.long()
            # x_a_cur=x_a_cur.long()

        return (x_cur, x_a_cur)


class ELangevinSamplerOrdinalwithGLU(nn.Module):
    def __init__(
            self,
            dim,
            bal,
            max_val=3,
            n_steps=10,
            multi_hop=False,
            temp=1.0,
            step_size=0.2,
            eta=1,
            mh=True,
            device=None,
    ):
        super().__init__()
        self.dim = dim
        self.n_steps = n_steps
        self._ar = 0.0
        self._mt = 0.0
        self._pt = 0.0
        self._hops = 0.0
        self._phops = 0.0
        self.multi_hop = multi_hop
        self.temp = temp
        # rbm sampling: accpt prob is about 0.5 with lr = 0.2, update 16 dims per step (total 784 dims). ising sampling: accept prob 0.5 with lr=0.2
        # ising learning: accept prob=0.7 with lr=0.2
        # ebm: statistic mnist: accept prob=0.45 with lr=0.2
        self.a_s = []
        self.bal = bal
        self.mh = mh
        self.max_val = max_val  ### number of classes in each dimension
        self.step_size = (step_size * self.max_val) ** (self.dim ** 2)
        self.eta = eta
        self.flag = 0
        self.entropy = True

    def agrad(self, x, x_a):
        return (x - x_a) / self.eta

    def get_grad(self, x, model):
        x = x.requires_grad_()
        out = model(x)
        gx = torch.autograd.grad(out.sum(), x)[0]
        return gx.detach()

    def _calc_logits(self, x_cur, grad):
        # creating the tensor of discrete values to compute the probabilities for
        batch_size = x_cur.shape[0]
        disc_values = torch.tensor([i for i in range(self.max_val)])[None, None, :]
        disc_values = disc_values.repeat((batch_size, self.dim, 1)).to(x_cur.device)
        term1 = torch.zeros((batch_size, self.dim, self.max_val))
        term2 = torch.zeros((batch_size, self.dim, self.max_val))
        x_expanded = x_cur[:, :, None].repeat((1, 1, self.max_val)).to(x_cur.device)
        grad_expanded = grad[:, :, None].repeat((1, 1, self.max_val)).to(x_cur.device)
        term1 = self.bal * grad_expanded * (disc_values - x_expanded)
        term2 = (disc_values - x_expanded) ** 2 * (1 / (2 * self.step_size))
        return term1 - term2

    def step(self, x, x_a, model, use_dula=False):
        """
        input x : bs * dim, every dim contains a integer of 0 to (num_cls-1)
        """

        x_cur = x
        x_a_cur = x_a
        m_terms = []
        prop_terms = []

        EPS = 1e-10
        for i in range(self.n_steps):
            if self.flag == 0:
                # batch size X dim
                grad = self.get_grad(x_cur.float(), model) / self.temp
                grad_x_cur_a = self.agrad(x_cur.float(), x_a_cur.float()) / self.temp
                grad_x_cur_modified = grad - grad_x_cur_a

                logits = self._calc_logits(x_cur, grad_x_cur_modified)
                cat_dist = torch.distributions.categorical.Categorical(logits=logits)
                x_delta = cat_dist.sample()
                # x_delta_a = x_a_cur + (self.step_size_a / 2.) * grad_x_cur_a + ( (1 * self.step_size_a) ** 0.5) * torch.randn_like(x_a_cur, device=x_a_cur.device)

                if self.mh:
                    lp_forward = torch.sum(cat_dist.log_prob(x_delta), dim=1)
                    grad_delta = self.get_grad(x_delta.float(), model) / self.temp
                    grad_delta_a = self.agrad(x_delta.float(), x_a_cur) / self.temp
                    grad_delta_modified = grad_delta - grad_delta_a

                    logits_reverse = self._calc_logits(x_delta, grad_delta_modified)

                    cat_dist_delta = torch.distributions.categorical.Categorical(
                        logits=logits_reverse
                    )
                    lp_reverse = torch.sum(cat_dist_delta.log_prob(x_cur), dim=1)

                    m_term = (model(x_delta).squeeze() - torch.sum(torch.pow(x_delta - x_a_cur, 2), dim=1) / (
                            2 * self.eta)) - (
                                     model(x_cur).squeeze() - torch.sum(torch.pow(x_cur - x_a_cur, 2), dim=1) / (
                                     2 * self.eta))

                    la = m_term + lp_reverse - lp_forward

                    a = (la.exp() > torch.rand_like(la)).float()
                    self.a_s.append(a.mean().item())

                    x_cur = x_delta * a[:, None] + x_cur * (1.0 - a[:, None])
                else:
                    x_cur = x_delta

                self.flag = 1
                x_cur = x_cur.long()
                return (x_cur, x_a_cur)
            else:
                x_a_cur = x_cur + np.sqrt(self.eta) * torch.randn_like(x_a_cur, device=x_cur.device)

                self.flag = 0
                return (x_cur, x_a_cur)  # Update x_a, not x


class LangevinSamplerTSP(nn.Module):
    def __init__(self, dim, n_steps=10, approx=False, multi_hop=False, fixed_proposal=False, temp=2., step_size=0.2,
                 mh=True):
        super().__init__()
        self.dim = dim
        self.n_steps = n_steps
        self._ar = 0.
        self._mt = 0.
        self._pt = 0.
        self._hops = 0.
        self._phops = 0.
        self.approx = approx
        self.fixed_proposal = fixed_proposal
        self.multi_hop = multi_hop
        self.temp = temp
        self.step_size = step_size
        self.entropy = False

        if approx:
            self.diff_fn = lambda x, m: utils.approx_difference_function(x, m) / self.temp
        else:
            self.diff_fn = lambda x, m: utils.difference_function(x, m) / self.temp

        self.mh = mh
        self.a_s = []
        self.hops = []

    def step(self, x, model):

        x_cur = x
        # print('instep')
        # print(x_cur)

        m_terms = []
        prop_terms = []

        EPS = 1e-10
        for i in range(self.n_steps):
            x_cur_binary = torch.stack([model.city_positions[city] for city in x_cur])
            forward_delta = (model.compute_gradient(x_cur)*(1-2. * x_cur_binary ))/ self.temp
            term2 = 1. / (2 * self.step_size)  # for binary {0,1}, the L2 norm is always 1
            flip_prob = torch.exp(forward_delta - term2) / (torch.exp(forward_delta - term2) + 1)
            rr = torch.rand_like(x_cur_binary)
            # print("rr:", rr)
            ind = (rr < flip_prob) * 1
            x_delta_binary = (1. - x_cur_binary) * ind + x_cur_binary * (1. - ind)
            x_delta = tuple(model.reverse_city_positions[tuple(x_delta_binary[i].tolist())] for i  in range(0,len(x_delta_binary)))

            if len(x_delta) != len(set(x_delta)):
                x_delta=x_cur
                x_delta_binary =x_cur_binary

            if self.mh:
                probs = flip_prob * ind + (1 - flip_prob) * (1. - ind)
                lp_forward = torch.sum(torch.log(probs + EPS), dim=-1)

                reverse_delta = model.compute_gradient(x_delta)*-(2. * x_delta_binary - 1)/ self.temp
                flip_prob = torch.exp(reverse_delta - term2) / (torch.exp(reverse_delta - term2) + 1)
                probs = flip_prob * ind + (1 - flip_prob) * (1. - ind)
                lp_reverse = torch.sum(torch.log(probs + EPS), dim=-1)

                m_term = (model(x_delta).squeeze() - model(x_cur).squeeze())
                la = m_term + lp_reverse - lp_forward
                a = (la.exp() > torch.rand_like(la)).float()
                self.a_s.append(a.mean().item())
                x_cur_binary1 = x_delta_binary * a[:, None] + x_cur_binary * (1. - a[:, None])
                x_cur1= tuple(model.reverse_city_positions[tuple(x_cur_binary1[i].tolist())] for i  in range(0,len(x_cur_binary1)))

                if len(x_cur1) == len(set(x_cur1)):
                    x_cur=x_cur1

            else:
                x_cur=x_delta

        return x_cur


class ELangevinSamplerTSP(nn.Module):
    def __init__(self, dim, n_steps=10, approx=False, multi_hop=False, fixed_proposal=False, temp=2., alpha=0.2,
                 alpha_a=0.5, mh=True, eta=10 ** 4):
        super().__init__()
        self.dim = dim
        self.n_steps = n_steps
        self._ar = 0.
        self._mt = 0.
        self._pt = 0.
        self._hops = 0.
        self._phops = 0.
        self.approx = approx
        self.fixed_proposal = fixed_proposal
        self.multi_hop = multi_hop
        self.temp = temp
        self.alpha = alpha
        self.alpha_a = alpha_a
        self.eta = eta  # flatness
        self.entropy = True
        self.gibbs_like_update = False

        if approx:
            self.diff_fn = lambda x, m: utils.approx_difference_function(x, m) / self.temp
        else:
            self.diff_fn = lambda x, m: utils.difference_function(x, m) / self.temp

        self.agrad = lambda eta, x, x_a: utils.auxillary_gradient_function(eta, x, x_a)

        self.mh = mh
        self.a_s = []
        self.hops = []

    def step(self, x,x_a, model):


        x_cur = x

        x_a_cur = x_a

        EPS = 1e-10
        for i in range(self.n_steps):
            x_cur_binary = torch.stack([model.city_positions[city] for city in x_cur])
            forward_delta = (model.compute_gradient(x_cur)*(1-2. * x_cur_binary ))/ self.temp
            term2 = 1. / (2 * self.alpha)  # for binary {0,1}, the L2 norm is always 1
            grad_x_cur_a = self.agrad(self.eta, x_cur_binary, x_a_cur)
            grad_x_cur_modified = forward_delta - (grad_x_cur_a / self.temp) * -(2 * x_cur_binary - 1)
            flip_prob = torch.exp(grad_x_cur_modified - term2) / (torch.exp(grad_x_cur_modified - term2) + 1)
            rr = torch.rand_like(x_cur_binary)
            # print("rr:", rr)
            ind = (rr < flip_prob) * 1
            x_delta_binary = (1. - x_cur_binary) * ind + x_cur_binary * (1. - ind)
            x_delta = tuple(model.reverse_city_positions[tuple(x_delta_binary[i].tolist())] for i  in range(0,len(x_delta_binary)))
            x_delta_a = x_a_cur + (self.alpha_a / 2.) * grad_x_cur_a + (
                    (1 * self.alpha_a) ** 0.5) * torch.randn_like(x_a_cur, device=x_a_cur.device)

            if len(x_delta) != len(set(x_delta)):
                x_delta=x_cur
                x_delta_binary =x_cur_binary
                x_delta_a = x_a_cur



            if self.mh:
                probs = flip_prob * ind + (1 - flip_prob) * (1. - ind)
                lp_forward = torch.sum(torch.log(probs + EPS), dim=-1)

                reverse_delta = model.compute_gradient(x_delta)*-(2. * x_delta_binary - 1)/ self.temp
                grad_x_delta_a = self.agrad(self.eta, x_delta_binary, x_delta_a)
                grad_x_delta_modified = reverse_delta - (grad_x_delta_a / self.temp) * -(2 * x_delta_binary - 1)
                flip_prob = torch.exp(grad_x_delta_modified- term2) / (torch.exp(grad_x_delta_modified - term2) + 1)
                probs = flip_prob * ind + (1 - flip_prob) * (1. - ind)
                lp_reverse = torch.sum(torch.log(probs + EPS), dim=-1)

                m_term = (model(x_delta).squeeze() - torch.sum(torch.pow(x_delta_binary - x_delta_a, 2), dim=1) / (
                        2 * self.eta)) - (
                                 model(x_cur).squeeze() - torch.sum(torch.pow(x_cur_binary - x_a_cur, 2), dim=1) / (
                                 2 * self.eta))

                additional_m_term_reverse = torch.sum(
                    torch.pow(x_a_cur - (x_delta_a + (self.alpha_a / 2) * grad_x_delta_a), 2), dim=1) / (
                                                    -2 * self.alpha_a)
                additional_m_term_forward = torch.sum(
                    torch.pow(x_delta_a - (x_a_cur + (self.alpha_a / 2) * grad_x_cur_a), 2), dim=1) / (
                                                    -2 * self.alpha_a)

                la = m_term + additional_m_term_reverse - additional_m_term_forward + lp_reverse - lp_forward
                a = (la.exp() > torch.rand_like(la)).float()
                self.a_s.append(a.mean().item())


                x_cur_binary1 = x_delta_binary * a[:, None] + x_cur_binary * (1. - a[:, None])
                x_cur1= tuple(model.reverse_city_positions[tuple(x_cur_binary1[i].tolist())] for i  in range(0,len(x_cur_binary1)))
                x_a_cur1 = x_delta_a * a[:, None] + x_a_cur * (1. - a[:, None])

                if len(x_cur1) == len(set(x_cur1)):
                    x_cur=x_cur1
                    x_a_cur=x_a_cur1

            else:
                x_cur=x_delta
                x_a_cur = x_delta_a

        return (x_cur, x_a_cur)








