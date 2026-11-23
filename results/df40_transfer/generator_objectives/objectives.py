"""Feature adapter and optional generator-adversarial or contrastive losses."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features):
        return features.view_as(features)

    @staticmethod
    def backward(ctx, gradient):
        return -gradient


class Adapter(nn.Module):
    def __init__(self, width, domains):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(width, 128), nn.ReLU())
        self.classifier = nn.Linear(128, 1)
        self.domain = nn.Linear(128, domains)

    def forward(self, features):
        representation = self.projection(features)
        return representation, self.classifier(representation).flatten()


def contrastive(representation, labels):
    features = F.normalize(representation, dim=1)
    logits = features @ features.T / .1
    diagonal = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    matches = (labels[:, None] == labels[None, :]) & ~diagonal
    positives = matches.sum(dim=1)
    valid = positives > 0
    if not valid.any():
        return representation.sum() * 0
    log_probs = F.log_softmax(logits.masked_fill(diagonal, -1e9), dim=1)
    loss = -(log_probs * matches).sum(dim=1)[valid] / positives[valid]
    return loss.mean()


def loss_for(model, features, labels, domains, objective):
    representation, logits = model(features)
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    if objective == 'adversarial':
        fake = labels == 1
        if fake.any():
            logits_domain = model.domain(Reverse.apply(representation[fake]))
            loss = loss + .1 * F.cross_entropy(logits_domain, domains[fake])
    elif objective == 'contrastive':
        loss = loss + .1 * contrastive(representation, labels)
    return loss
