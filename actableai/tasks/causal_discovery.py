from typing import Union, Callable
import pandas as pd

from actableai.tasks.base import AAITask
from actableai.tasks import TaskType
from actableai.causal.discover.algorithms.commons.base_runner import CausalGraph
from actableai.causal.discover.model.causal_discovery import CausalDiscoveryPayload
from actableai.causal.discover.algorithms.deci import DeciRunner
from actableai.causal.discover.algorithms.notears import NotearsRunner
from actableai.causal.discover.algorithms.direct_lingam import DirectLiNGAMRunner
from actableai.causal.discover.algorithms.pc import PCRunner


class AAICausalDiscoveryTask(AAITask):
    @AAITask.run_with_ray_remote(TaskType.CAUSAL_INFERENCE)
    def run(
        self,
        algo: str,
        payload: CausalDiscoveryPayload,
        progress_callback: Union[Callable, None] = None,
    ) -> CausalGraph:
        """Run a causal discovery algorithm.

        Args:
            algo (str): The name of the algorithm to run. Must be either "deci", "notears",
                "direct-lingamp" or "pc".

            payload (CausalDiscoveryPayload): The payload to use for the algorithm. Use
                actableai.causal.discovery.algorithms.deci.DeciPayload for "deci",
                actableai.causal.discovery.algorithms.notears.NotearsPayload for "notears",
                actableai.causal.discovery.algorithms.direct_lingam.DirectLiNGAMPayload for
                "direct-lingam" and actableai.causal.discovery.algorithms.pc.PCPayload for "pc".

            progress_callback (Union[Callable, None], optional): A callback to use for progress reporting. Defaults to None.

        Return:
            CausalGraph: The causal graph produced by the algorithm.

        Raises:
            ValueError: If the algorithm is not supported.
        """
        if algo == "deci":
            return DeciRunner(payload, progress_callback).run()
        elif algo == "notears":
            return NotearsRunner(payload, progress_callback).run()
        elif algo == "direct-lingam":
            return DirectLiNGAMRunner(payload, progress_callback).run()
        elif algo == "pc":
            return PCRunner(payload, progress_callback).run()
        else:
            raise ValueError(f"Unknown algorithm: {algo}")
