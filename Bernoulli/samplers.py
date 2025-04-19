import math

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
        # print('Langevin Sampler')

    def step(self, x, model):
        #print(x)


        x_cur = x
        # print('instep')
        # print(x_cur)

        m_terms = []
        prop_terms = []

        EPS = 1e-10
        for i in range(self.n_steps):
            grad_x_cur = self.diff_fn(x_cur, model)
            #print(grad_x_cur)

            term2 = 1. / (2 * self.step_size)  # for binary {0,1}, the L2 norm is always 1
            flip_prob = torch.exp(grad_x_cur - term2) / (torch.exp(grad_x_cur - term2) + 1)

            #print("prob:",flip_prob)

            rr = torch.rand_like(x_cur)
            #print("rr:", rr)
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
    def __init__(self, dim, n_steps=10, approx=False, multi_hop=False, fixed_proposal=False, temp=2., alpha =0.2,
                 alpha_a =0.5, mh=True,eta=10 ** 4):
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
        self.alpha  = alpha 
        self.alpha_a  = alpha_a 
        self.eta = eta  # flatness
        self.entropy = True
        self.gibbs_like_update=False


        if approx:
            self.diff_fn = lambda x, m: utils.approx_difference_function(x, m) / self.temp
        else:
            self.diff_fn = lambda x, m: utils.difference_function(x, m) / self.temp

        self.agrad = lambda eta, x, x_a: utils.auxillary_gradient_function(eta, x, x_a)

        self.mh = mh
        self.a_s = []
        self.hops = []


    def step(self, x, x_a, model):
        #print(x)

        x_cur = x

        x_a_cur = x_a

        m_terms = []
        prop_terms = []

        EPS = 1e-10
        for i in range(self.n_steps):
                grad_x_cur = self.diff_fn(x_cur, model)
                grad_x_cur_a=self.agrad(self.eta, x_cur, x_a_cur)

                grad_x_cur_modified=grad_x_cur-(grad_x_cur_a/self.temp)*-(2*x_cur-1)


                term2 = 1. / (2 * self.alpha )  # for binary {0,1}, the L2 norm is always 1
                flip_prob = torch.exp(grad_x_cur_modified - term2) / (torch.exp(grad_x_cur_modified - term2) + 1)  # softmax
                #print("prob:", flip_prob)

                rr = torch.rand_like(x_cur)
                #rng_state = torch.get_rng_state()

                ind = (rr < flip_prob) * 1
                x_delta = (1. - x_cur) * ind + x_cur * (1. - ind)

                x_delta_a = x_a_cur + (self.alpha_a / 2.) * grad_x_cur_a + (
                        (1 * self.alpha_a) ** 0.5) * torch.randn_like(x_a_cur,device=x_a_cur.device)
                #torch.set_rng_state(rng_state)

                if self.mh:

                    probs = flip_prob * ind + (1 - flip_prob) * (1. - ind)
                    lp_forward = torch.sum(torch.log(probs + EPS), dim=-1)

                    grad_x_delta = self.diff_fn(x_delta, model)
                    grad_x_delta_a=self.agrad(self.eta, x_delta, x_delta_a)

                    grad_x_delta_modified = grad_x_delta - (grad_x_delta_a/self.temp)* -(2 * x_delta - 1)

                    flip_prob = torch.exp(grad_x_delta_modified - term2) / (torch.exp(grad_x_delta_modified - term2) + 1)
                    probs = flip_prob * ind + (1 - flip_prob) * (1. - ind)
                    lp_reverse = torch.sum(torch.log(probs + EPS), dim=-1)


                    m_term = (model(x_delta).squeeze() -torch.sum(torch.pow(x_delta-x_delta_a,2), dim=1)/(2*self.eta)) - (model(x_cur).squeeze()- torch.sum(torch.pow(x_cur-x_a_cur,2),dim=1)/(2*self.eta))

                    additional_m_term_reverse=-torch.sum(torch.pow(x_a_cur-(x_delta_a+(self.alpha_a/2)*grad_x_delta_a),2), dim=1)/(2*self.alpha_a)
                    additional_m_term_forward =-torch.sum( torch.pow( x_delta_a - (x_a_cur + (self.alpha_a / 2) * grad_x_cur_a) ,2 ), dim=1 ) / (2 * self.alpha_a)

                    la = m_term +additional_m_term_reverse -additional_m_term_forward+ lp_reverse - lp_forward



                    a = (la.exp() > torch.rand_like(la)).float()
                    self.a_s.append(a.mean().item())
                    x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])
                    x_a_cur=x_delta_a * a[:, None] + x_a_cur * (1. - a[:, None])

                else:
                    x_cur = x_delta  # Langevin Update
                    x_a_cur = x_delta_a

        return (x_cur, x_a_cur)  # Non-GLU Return








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


    def step(self, x, model):

        x_cur = x
        a_s = []
        m_terms = []
        prop_terms = []

        for i in range(self.n_steps):
            grad_x_cur = self.diff_fn(x_cur, model)
            cd_forward = dists.OneHotCategorical(logits=grad_x_cur)
            changes_all = cd_forward.sample((self.n_samples,))

            lp_forward = cd_forward.log_prob(changes_all).sum(0)

            changes = (changes_all.sum(0) > 0.).float()

            x_delta = (1. - x_cur) * changes + x_cur * (1. - changes)
            self._phops = (x_delta != x).float().sum(-1).mean().item()

            reverse_delta = self.diff_fn(x_delta, model)
            cd_reverse = dists.OneHotCategorical(logits=reverse_delta)

            lp_reverse = cd_reverse.log_prob(changes_all).sum(0)

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

        self._hops = (x != x_cur).float().sum(-1).mean().item()
        return x_cur

class PerDimGibbsSampler(nn.Module):
    def __init__(self, dim, rand=False):
        super().__init__()
        self.dim = dim
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
        #updates=updates.unsqueeze(0)
        sample = sample_change * updates[:, None] + sample * (1. - updates[:, None])
        self.changes[self._i] = updates.mean()
        self._i = (self._i + 1) % self.dim
        self._hops = (x != sample).float().sum(-1).mean().item()
        self._ar = self._hops
        return sample

    def logp_accept(self, xhat, x, model):
        # only true if xhat was generated from self.step(x, model)
        return 0.


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
                    grad_x_cur = self.diff_fn(x_cur, model)
                    cd_forward = dists.Bernoulli(logits=(grad_x_cur * 2 / self.temp))
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
                    grad_x_cur = self.diff_fn(x_cur, model)
                    cd_forward = dists.OneHotCategorical(logits=grad_x_cur)
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




class ELangevinSamplerwithGLU(nn.Module):
    def __init__(self, dim, n_steps=10, approx=False, multi_hop=False, fixed_proposal=False, temp=2., alpha =0.2,
                 mh=True,eta=10 ** 4):
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


        self.gibbs_like_update=True
        self.glu_flag=0


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

            if (self.glu_flag==0):

                grad_x_cur = self.diff_fn(x_cur, model)
                grad_x_cur_a=self.agrad(self.eta, x_cur, x_a_cur)
                grad_x_cur_modified=grad_x_cur-(grad_x_cur_a/self.temp)*-(2*x_cur-1)


                term2 = 1. / (2 * self.alpha )  # for binary {0,1}, the L2 norm is always 1
                flip_prob = torch.exp(grad_x_cur_modified - term2) / (torch.exp(grad_x_cur_modified - term2) + 1)  # softmax

                rr = torch.rand_like(x_cur)
                ind = (rr < flip_prob) * 1
                x_delta = (1. - x_cur) * ind + x_cur * (1. - ind)

                if self.mh:
                    probs = flip_prob * ind + (1 - flip_prob) * (1. - ind)
                    lp_forward = torch.sum(torch.log(probs + EPS), dim=-1)

                    grad_x_delta = self.diff_fn(x_delta, model)
                    grad_x_delta_a=self.agrad(self.eta, x_delta, x_a_cur)

                    grad_x_delta_modified = grad_x_delta - (grad_x_delta_a/self.temp)* -(2 * x_delta - 1)

                    flip_prob = torch.exp(grad_x_delta_modified - term2) / (torch.exp(grad_x_delta_modified - term2) + 1)
                    probs = flip_prob * ind + (1 - flip_prob) * (1. - ind)
                    lp_reverse = torch.sum(torch.log(probs + EPS), dim=-1)


                    m_term =  (model(x_delta).squeeze() -torch.sum(torch.pow(x_delta-x_a_cur,2),dim=1)/(2*self.eta)) - (model(x_cur).squeeze()- torch.sum(torch.pow(x_cur-x_a_cur,2),dim=1)/(2*self.eta))


                    la = m_term  + lp_reverse - lp_forward
                    a = (la.exp() > torch.rand_like(la)).float()
                    self.a_s.append(a.mean().item())
                    x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])

                else:
                    x_cur = x_delta

                self.glu_flag = 1
                return (x_cur, x_a_cur) #Update x, not x_a

            else:

                x_a_cur = x_cur+ np.sqrt(self.eta) * torch.randn_like(x_cur, device=x_cur.device)

                self.glu_flag=0

                return (x_cur, x_a_cur) #Update x_a, not x

