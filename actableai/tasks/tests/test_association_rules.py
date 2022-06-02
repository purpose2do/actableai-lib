import pytest

from actableai import AAIAssociationRulesTask


@pytest.fixture(scope="function")
def association_rules_task():
    yield AAIAssociationRulesTask(use_ray=False)


class TestAssociationRules:
    def test_association_rules(self):
        raise NotImplementedError()
