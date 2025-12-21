from __future__ import annotations

from .registry import Registry

# Core registries for plug-in style components.
MODEL_REGISTRY = Registry("model")
TRAINER_REGISTRY = Registry("trainer")
CALLBACK_REGISTRY = Registry("callback")

LOSS_REGISTRY = Registry("loss")
OPTIM_REGISTRY = Registry("optimizer")
SCHED_REGISTRY = Registry("scheduler")
METRIC_REGISTRY = Registry("metric")
TRANSFORM_REGISTRY = Registry("transform")
DATASET_REGISTRY = Registry("dataset")
