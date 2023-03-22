import logging
from tempfile import mkdtemp
from typing import Any, Tuple, Dict

import networkx
import numpy as np
import pandas as pd
import scipy
import torch
from causica.datasets.dataset import Dataset
from causica.datasets.variables import Variable, Variables
from causica.models.deci.deci import DECI
from causica.utils.torch_utils import get_torch_device

from actableai.causal.discover.algorithms.commons.base_runner import (
    CausalDiscoveryRunner,
    CausalGraph,
    ProgressCallback,
)
from actableai.causal.discover.algorithms.interventions.deci import (
    DeciInterventionModel,
)
from actableai.causal.discover.algorithms.payloads import DeciPayload, ATEDetails
from actableai.causal.discover.model.causal_discovery import (
    map_to_causica_var_type,
)

torch.set_default_dtype(torch.float32)

TRAINING_PROGRESS_PROPORTION = 0.7

ATE_CALC_PROGRESS_PROPORTION = 0.3


class DeciRunner(CausalDiscoveryRunner):
    name = "DECI"

    def __init__(self, p: DeciPayload, progress_callback: ProgressCallback = None):
        super().__init__(p, progress_callback)
        self._model_options = p.model_options
        self._training_options = p.training_options
        self._ate_options = p.ate_options
        # make sure every run has its own folder
        self._is_dag = None
        self._device = get_torch_device("gpu")
        self._deci_save_dir = p.model_save_dir

        if self._deci_save_dir is None:
            self._deci_save_dir = mkdtemp(prefix="actableai_model")

    def _build_causica_dataset(self) -> Dataset:
        self._encode_categorical_as_integers()
        numpy_data = self._prepared_data.to_numpy()
        data_mask = np.ones(numpy_data.shape)

        variables = Variables.create_from_data_and_dict(
            numpy_data,
            data_mask,
            {
                "variables": [
                    {
                        "name": name,
                        # TODO: this is currently mapping categorical to continuous
                        #       we need to update the interventions code to properly handle
                        #       one-hot encoded values
                        "type": map_to_causica_var_type(self._nature_by_variable[name]),
                        "lower": self._prepared_data[name].min(),
                        "upper": self._prepared_data[name].max(),
                    }
                    for name in self._prepared_data.columns
                ]
            },
        )

        return Dataset(train_data=numpy_data, train_mask=data_mask, variables=variables)

    def _build_model(self, causica_dataset: Dataset) -> DECI:
        logging.info(
            f"Creating DECI model with '{self._model_options.base_distribution_type}' base distribution type"
        )

        constraints = self._build_constraint_matrix(
            causica_dataset.variables.name_to_idx,
            tabu_child_nodes=self._constraints.causes,
            tabu_parent_nodes=self._constraints.effects,
            tabu_edges=self._constraints.forbiddenRelationships,
            edge_hints=self._constraints.potentialRelationships,
        ).astype(np.float32)

        # causica expects NaN for edges that need to be discovered
        constraints[constraints == -1] = np.nan

        deci_model = DECI(
            "CauseDisDECI",
            causica_dataset.variables,
            self._deci_save_dir,
            self._device,
            **self._model_options.dict(),
            graph_constraint_matrix=constraints,
        )

        return deci_model

    def _check_if_is_dag(self, deci_model: DECI, adj_matrix: np.ndarray) -> bool:
        return (
            np.trace(scipy.linalg.expm(adj_matrix.round())) - deci_model.num_nodes
        ) == 0

    def _get_adj_matrix(self, deci_model: DECI) -> np.ndarray:
        adj_matrix = deci_model.get_adj_matrix(
            do_round=False,
            samples=1,
            most_likely_graph=True,
            squeeze=True,
        )
        self._is_dag = self._check_if_is_dag(deci_model, adj_matrix)

        return adj_matrix

    def _infer_intervention_reference_values(
        self, train_data: pd.DataFrame, var: Variable
    ) -> Tuple[float, float]:
        # TODO: perhaps allow the user to set the (intervention, reference) values per variable in the frontend
        if var.type_ == "binary":
            return (1, 0)
        elif var.type_ == "categorical":
            value_counts = train_data[var.name].value_counts()
            return (value_counts.idxmax(), value_counts.idxmin())
        else:
            mean = train_data[var.name].mean()
            std = train_data[var.name].std()
            return (mean + std, mean - std)

    def _get_or_infer_intervention_reference_values(
        self, train_data: pd.DataFrame, var: Variable
    ) -> Tuple[float, float]:
        intervention, reference = self._infer_intervention_reference_values(
            train_data, var
        )
        provided_ate_details = self._ate_options.ate_details_by_name.get(var.name, None)

        if provided_ate_details is not None:
            if isinstance(provided_ate_details.intervention, (int, float)):
                intervention = provided_ate_details.intervention
            if isinstance(provided_ate_details.reference, (int, float)):
                reference = provided_ate_details.reference

        return (intervention, reference)

    def _apply_group_mask_to_ate_array(
        self, ate_array: np.ndarray, group_mask: np.ndarray
    ):
        # categorical columns are one-hot encoded, so we apply the mask to
        # go back to the same dimensionality in the original data
        return [ate_array[mask].mean() for mask in group_mask]

    def _compute_average_treatment_effect(
        self, model: DECI, train_data: pd.DataFrame
    ) -> Tuple[np.ndarray, Dict[str, ATEDetails]]:
        ate_matrix = []
        n_variables = train_data.shape[1]
        progress_step = ATE_CALC_PROGRESS_PROPORTION / n_variables

        used_ate_details_by_name = dict()

        if self._ate_options.most_likely_graph and self._ate_options.Ngraphs != 1:
            logging.warning(
                "Adjusting Ngraphs parameter to 1 because most_likely_graph is set to true"
            )
            self._ate_options.Ngraphs = 1

        for variable_index in range(n_variables):
            variable = model.variables[variable_index]
            (
                intervention_value,
                reference_value,
            ) = self._get_or_infer_intervention_reference_values(train_data, variable)

            used_ate_details_by_name[variable.name] = ATEDetails(
                reference=reference_value,
                intervention=intervention_value,
                nature=self._nature_by_variable[variable.name],
            )

            logging.info(
                f"Computing the ATE between {variable.name}={intervention_value} and {variable.name}={reference_value}"
            )

            ate_array, _ = model.cate(
                intervention_idxs=torch.tensor([variable_index]),
                intervention_values=torch.tensor([intervention_value]),
                reference_values=torch.tensor([reference_value]),
                Ngraphs=self._ate_options.Ngraphs,
                Nsamples_per_graph=self._ate_options.Nsamples_per_graph,
                most_likely_graph=self._ate_options.most_likely_graph,
            )
            ate_matrix.append(
                self._apply_group_mask_to_ate_array(
                    ate_array, model.variables.group_mask
                )
            )

            self._report_progress(
                (TRAINING_PROGRESS_PROPORTION + ((variable_index + 1) * progress_step))
                * 100.0
            )

        return np.stack(ate_matrix), used_ate_details_by_name

    def _build_labeled_graph(
        self,
        name_to_idx: Dict[str, int],
        adj_matrix: np.ndarray,
        ate_matrix: np.ndarray,
    ) -> Any:
        deci_graph = networkx.convert_matrix.from_numpy_matrix(
            adj_matrix, create_using=networkx.DiGraph
        )
        deci_ate_graph = networkx.convert_matrix.from_numpy_matrix(
            ate_matrix, create_using=networkx.DiGraph
        )
        labels = {idx: name for name, idx in name_to_idx.items()}

        for n1, n2, d in deci_graph.edges(data=True):
            ate = deci_ate_graph.get_edge_data(n1, n2, default={"weight": 0})["weight"]
            d["confidence"] = d.pop("weight", None)
            d["weight"] = ate

        return networkx.relabel_nodes(deci_graph, labels)

    def _do_causal_discovery(self) -> CausalGraph:
        # if the data contains only a single column,
        # let's return an empty graph
        if self._prepared_data.columns.size == 1:
            return self._get_empty_graph_json(self._prepared_data)

        causica_dataset = self._build_causica_dataset()

        train_data = pd.DataFrame(
            causica_dataset._train_data, columns=self._prepared_data.columns
        )

        deci_model = self._build_model(causica_dataset)

        training_options_dict = self._training_options.dict()

        logging.info(f"running training with options: {training_options_dict}")

        deci_model.run_train(
            causica_dataset,
            training_options_dict,
            lambda model_id, step, max_steps: self._report_progress(
                (step * 100.0 / max_steps) * TRAINING_PROGRESS_PROPORTION
            ),
        )

        adj_matrix = self._get_adj_matrix(deci_model)

        ate_matrix, used_ate_details_by_name = self._compute_average_treatment_effect(
            deci_model, train_data
        )

        deci_model.eval()

        self._intervention_model = DeciInterventionModel(
            deci_model, adj_matrix, ate_matrix, train_data
        )

        causal_graph = self._build_causal_graph(
            labeled_graph=self._build_labeled_graph(
                deci_model.variables.name_to_idx, adj_matrix, ate_matrix
            ),
            has_weights=True,
            has_confidence_values=True,
            columns=self._get_column_names(),
            is_dag=bool(self._is_dag),
            intervention_model_id=self._intervention_model.id,
            ate_details_by_name=used_ate_details_by_name,
        )

        self._report_progress(100.0)

        return causal_graph

    def do_causal_discovery(self) -> CausalGraph:
        return self._do_causal_discovery()
