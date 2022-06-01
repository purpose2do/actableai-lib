import pytest

from actableai import AAIAssociationRuleTask


@pytest.fixture(scope="function")
def intervention_task():
    yield AAIAssociationRuleTask(use_ray=False)


class TestAssociationRule:
    def test_association_rule(self):
        raise NotImplementedError()
