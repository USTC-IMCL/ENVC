import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FactorizedEntropy(nn.Module):
    def __init__(self, channel, init_scale=10, filters=(3, 3, 3),
                 likelihood_bound=1e-6):
        super().__init__()
        self.channel = channel
        self.init_scale = init_scale
        self.filters = filters
        self.likelihood_bound = likelihood_bound
        self.matrices = nn.ParameterList()
        self.biases = nn.ParameterList()
        self.factors = nn.ParameterList()

        _filters = (1,) + self.filters + (1,)
        scale = self.init_scale ** (1 / (len(self.filters) + 1))
        for i in range(len(self.filters) + 1):
            init = np.log(np.expm1(1 / scale / _filters[i + 1]))
            matrix = nn.Parameter(
                torch.Tensor(1, channel, _filters[i + 1], _filters[i]))
            nn.init.constant_(matrix, init)
            self.matrices.append(matrix)

            bias = nn.Parameter(torch.Tensor(1, channel, _filters[i + 1], 1))
            nn.init.uniform_(bias, -0.5, 0.5)
            self.biases.append(bias)

            if i < len(self.filters):
                factor = nn.Parameter(
                    torch.Tensor(1, channel, _filters[i + 1], 1))
                nn.init.constant_(factor, 0)
                self.factors.append(factor)

    def forward(self, input):
        likelihood = self._likelihood(input)
        likelihood = torch.clamp(likelihood, min=self.likelihood_bound)
        return likelihood

    def _logits_cumulative(self, input):
        logits = input
        for i in range(len(self.filters) + 1):
            matrix = self.matrices[i]
            matrix = F.softplus(matrix)
            logits = matrix.matmul(logits)
            bias = self.biases[i]
            logits += bias
            if i < len(self.factors):
                factor = self.factors[i]
                factor = torch.tanh(factor)
                logits += factor * torch.tanh(logits)
        return logits

    def _likelihood(self, input):
        shape = input.shape
        B, C = input.shape[0:2]
        input = input.view(B, C, 1, -1)
        lower = self._logits_cumulative(input - .5)
        upper = self._logits_cumulative(input + .5)
        sign = -torch.sign(lower + upper)
        sign = sign.detach()
        likelihood = torch.abs(
            torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower))
        likelihood = likelihood.view(shape)
        return likelihood


class GaussianEntropy(nn.Module):
    def __init__(self, scale_bound=1e-6, likelihood_bound=1e-6):
        super().__init__()
        self.scale_bound = scale_bound
        self.likelihood_bound = likelihood_bound

    def forward(self, input, mean, scale):
        scale = torch.clamp(scale, min=self.scale_bound)
        likelihood = self._likelihood(input, mean, scale)
        likelihood = torch.clamp(likelihood, min=self.likelihood_bound)
        return likelihood

    def _standardized_cumulative(self, input):
        half = .5
        const = -(2 ** -0.5)
        # Using the complementary error function maximizes numerical precision.
        return half * torch.erfc(const * input)

    def _likelihood(self, input, mean, scale):
        value = input - mean
        value = torch.abs(value)
        upper = self._standardized_cumulative((.5 - value) / scale)
        lower = self._standardized_cumulative((-.5 - value) / scale)
        likelihood = upper - lower
        return likelihood
