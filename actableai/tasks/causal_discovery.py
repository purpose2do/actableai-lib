from typing import Union, Callable, Dict

from actableai.causal.discover.algorithms.payloads import CausalDiscoveryPayload
from actableai.tasks.base import AAITask
from actableai.tasks import TaskType


class AAICausalDiscoveryTask(AAITask):
    @AAITask.run_with_ray_remote(TaskType.CAUSAL_DISCOVERY)
    def run(
        self,
        algo: str,
        payload: CausalDiscoveryPayload,
        progress_callback: Union[Callable, None] = None,
    ) -> Dict:
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
        import time
        from copy import deepcopy

        from actableai.causal.discover import algorithms
        from actableai.data_validation.base import CheckLevels
        from actableai.data_validation.params import CausalDiscoveryDataValidator

        start = time.time()

        payload.dataset = deepcopy(payload.dataset)

        data_validation_results = CausalDiscoveryDataValidator().validate(
            algo=algo,
        )
        failed_checks = [
            check for check in data_validation_results if check is not None
        ]

        if CheckLevels.CRITICAL in [x.level for x in failed_checks]:
            return {
                "status": "FAILURE",
                "data": {},
                "validations": [
                    {"name": check.name, "level": check.level, "message": check.message}
                    for check in failed_checks
                ],
                "runtime": time.time() - start,
            }

        runner = algorithms.runners.get(algo)
        graph = runner(p=payload, progress_callback=progress_callback).run()

        data = {
            "causal_graph": graph,
        }

        return {
            "status": "SUCCESS",
            "messenger": "",
            "validations": [
                {"name": x.name, "level": x.level, "message": x.message}
                for x in failed_checks
            ],
            "data": data,
            "runtime": time.time() - start,
        }
