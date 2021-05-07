import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class AngularSoftmax(nn.Module):

    def __init__(self, in_features, output_classes, m=3):
        super(AngularSoftmax, self).__init__()
        self.m = m
        self.weights = Parameter(torch.FloatTensor(in_features, output_classes))
        nn.init.xavier_uniform_(self.weights)

    def forward(self, embeddings, labels):
        normalized_weights = F.normalize(self.weights, dim=0)
        yn_mask = ([i for i in range(labels.shape[0])], labels)
        logits = torch.matmul(embeddings, normalized_weights)
        
        # computing numerators
        emb_magnitudes = torch.linalg.norm(embeddings, dim=1)
        dot_products = logits[yn_mask]
        cosines = dot_products / emb_magnitudes
        angles = torch.acos(torch.clip(cosines, -1, +1))
        k = torch.floor(angles / (math.pi / self.m))
        phi = ((-1) ** k) * torch.cos(angles * self.m) - 2 * k
        numerators = torch.exp(emb_magnitudes * phi)
        
        # computing denominators
        norm_sums = torch.exp(logits)
        norm_sums = torch.sum(norm_sums, dim=1) - norm_sums[yn_mask]
        denominators = numerators + norm_sums
        
        # for numerical stability
        eps = torch.finfo(torch.float32).eps
        numerators = torch.abs(numerators) + eps
        denominators = torch.abs(denominators) + eps
        
        total_loss = (-1) * torch.log(numerators / denominators)
        return torch.mean(total_loss)

    # # sequential version:
    # def forward(self, embeddings, labels):
    #     total_loss = 0
    #     count = 0
    #
    #     normalized_weights = F.normalize(self.weights)
    #
    #     for emb, label in zip(embeddings, labels):
    #         # computing numerator
    #         emb_magnitude = torch.linalg.norm(emb).item()
    #         cosine = torch.dot(emb, normalized_weights[label]) / emb_magnitude
    #         angle = math.acos(cosine.item())
    #         k = int(angle / (math.pi / self.m))
    #         phi = ((-1) ** k) * math.cos(angle * self.m) - 2 * k
    #         numerator = math.exp(emb_magnitude * phi)
    #
    #         # computing denominator
    #         norm_sum = torch.matmul(normalized_weights, emb)
    #         norm_sum = torch.exp(norm_sum)
    #         norm_sum[label] = 0
    #         norm_sum = torch.sum(norm_sum).item()
    #         denominator = numerator + norm_sum
    #
    #         total_loss -= math.log(numerator / denominator)
    #         count += 1
    #
    #     return total_loss / count
