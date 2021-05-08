import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class AngularSoftmax(nn.Module):

    def __init__(self, in_features, output_classes, m=1):
        super(AngularSoftmax, self).__init__()
        self.m = m
        self.weights = Parameter(torch.FloatTensor(in_features, output_classes))
        nn.init.xavier_uniform_(self.weights)

    def forward(self, inputs, labels):
        w_normalized = F.normalize(self.weights, dim=0)
        logits = torch.matmul(inputs, w_normalized)

        labels_list = labels.long().tolist()
        yn_indices = ([i for i in range(len(labels_list))], labels_list)
        yn_complement_mask = torch.ones(logits.shape, dtype=torch.float32)
        yn_complement_mask[yn_indices] = 0
        yn_complement_mask = yn_complement_mask.to(inputs.device)

        # for numerical stability
        eps = torch.finfo(torch.float32).eps

        # computing numerators
        x_norms = torch.linalg.norm(inputs, dim=1)
        cosines = torch.clip(logits[yn_indices] / x_norms, -1, +1)
        angles = torch.acos(cosines)
        k = torch.floor(angles / (math.pi / self.m))
        # phi = ((-1) ** k) * torch.cos(angles * self.m) - 2 * k
        cos_m_theta = 4 * (cosines ** 3) - 3 * cosines  # multi-angle formula
        phi = ((-1) ** k) * cosines - 2 * k
        numerators = torch.exp(x_norms * phi) + eps

        # computing denominators
        norm_sums = torch.exp(logits)
        norm_sums = norm_sums * yn_complement_mask
        norm_sums = torch.sum(norm_sums, dim=1)
        denominators = numerators + norm_sums

        total_loss = (-1) * torch.log(numerators / denominators)
        return torch.mean(total_loss)

    # # sequential version:
    # def forward(self, embeddings, labels):
    #     total_loss = 0
    #     count = 0
    #
    #     w_normalizedeights = F.normalize(self.weights)
    #
    #     for emb, label in zip(embeddings, labels):
    #         # computing numerator
    #         emb_magnitude = torch.linalg.norm(emb).item()
    #         cosine = torch.dot(emb, w_normalizedeights[label]) / emb_magnitude
    #         angle = math.acos(cosine.item())
    #         k = int(angle / (math.pi / self.m))
    #         phi = ((-1) ** k) * math.cos(angle * self.m) - 2 * k
    #         numerator = math.exp(emb_magnitude * phi)
    #
    #         # computing denominator
    #         norm_sum = torch.matmul(w_normalizedeights, emb)
    #         norm_sum = torch.exp(norm_sum)
    #         norm_sum[label] = 0
    #         norm_sum = torch.sum(norm_sum).item()
    #         denominator = numerator + norm_sum
    #
    #         total_loss -= math.log(numerator / denominator)
    #         count += 1
    #
    #     return total_loss / count
