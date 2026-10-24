import itertools
from typing import Dict, Any, List

from models import NeckType, HeadType
from losses import LossType


class ExperimentGenerator:
    """Generates a list of atomic configurations for grid search."""

    def __init__(self, grid_spec: Dict[str, Any]):
        self.grid_spec = grid_spec

    @staticmethod
    def _expand_params(params_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Expands a dictionary of parameter options into a list of atomic parameter combinations.
        Returns [{}] if params_spec is empty.
        """
        if not params_spec:
            return [{}]

        keys = list(params_spec.keys())
        # Wrap scalar values in a list if not already formatted as a list in YAML
        values = [v if isinstance(v, list) else [v] for v in params_spec.values()]

        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    def generate(self) -> List[Dict[str, Any]]:
        experiments = []

        neck_specs = self.grid_spec["neck"]
        head_specs = self.grid_spec["head"]
        loss_specs = self.grid_spec["loss"]
        optimizer_spec = self.grid_spec["optimizer"]

        # Cartesian expansion across all sections
        opt_configs = self._expand_params(optimizer_spec)

        for neck_type_raw, neck_params in neck_specs.items():
            neck_configs = self._expand_params(neck_params)

            for head_type_raw, head_params in head_specs.items():
                head_configs = self._expand_params(head_params)

                for loss_type_raw, loss_params in loss_specs.items():
                    loss_configs = self._expand_params(loss_params)

                    for n_cfg in neck_configs:
                        for h_cfg in head_configs:
                            for l_cfg in loss_configs:
                                for opt_cfg in opt_configs:
                                    experiments.append({
                                        "neck_type": NeckType.from_str(neck_type_raw),
                                        "neck_kwargs": n_cfg,
                                        "head_type": HeadType.from_str(head_type_raw),
                                        "head_kwargs": h_cfg,
                                        "loss_type": LossType.from_str(loss_type_raw),
                                        "loss_kwargs": l_cfg,
                                        "lr": float(opt_cfg["lr"]),
                                        "weight_decay": float(opt_cfg["weight_decay"]),
                                    })

        return experiments