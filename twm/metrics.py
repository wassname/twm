import collections
import torch
from wandb.sdk.data_types.base_types.wb_value import WBValue

def update_metrics(metrics, new_metrics, prefix=None):
    def process(key, t):
        if isinstance(t, (int, float)):
            return t
        assert torch.is_tensor(t), key
        assert not t.requires_grad, key
        assert t.ndim == 0 or t.shape == (1,), key
        return t.clone()

    if prefix is None:
        metrics.update({key: process(key, value) for key, value in new_metrics.items()})
    else:
        metrics.update({f'{prefix}{key}': process(key, value) for key, value in new_metrics.items()})
    return metrics


def combine_metrics(metrics, prefix=None):
    result = {}
    if prefix is None:
        for met in metrics:
            update_metrics(result, met)
    else:
        for met, pre in zip(metrics, prefix):
            update_metrics(result, met, pre)
    return result


def mean_metrics(metrics_history, except_keys=None):
    if len(metrics_history) == 0:
        return {}
    if len(metrics_history) == 1:
        return metrics_history[0]
    except_keys = set() if except_keys is None else set(except_keys)
    result = {}
    value_history = collections.defaultdict(lambda: [])
    for metrics in metrics_history:
        for key, value in metrics.items():
            if key in except_keys or isinstance(value, WBValue):
                result[key] = value  # use last value
            else:
                value_history[key].append(value)
    result.update({key: compute_mean(values) for key, values in value_history.items()})
    return result


class MetricsSummarizer:

    def __init__(self, except_keys=None):
        self.metrics_history = []
        self.except_keys = set() if except_keys is None else set(except_keys)

    def append(self, metrics):
        self.metrics_history.append(metrics)

    def summarize(self):
        summary = mean_metrics(self.metrics_history, except_keys=self.except_keys)
        self.metrics_history = []
        # logger.debug(summary)
        return summary


def compute_mean(values):
    if torch.is_tensor(values):
        return values.float().mean()
    if isinstance(values, (tuple, list)):
        return torch.stack([torch.as_tensor(x).detach() for x in values]).float().mean()
    raise ValueError()
