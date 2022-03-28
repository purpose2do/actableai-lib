import pytest

from actableai.data_imputation.error_detector.constraint import (
    Condition,
    ConditionGroup,
    Constraint,
    Constraints,
)


@pytest.mark.parametrize(
    "string, expect",
    [
        ("a>b", "a>b"),
        ("a2>b1", "a2>b1"),
        ("aaa1222_232 >    apple9091", "aaa1222_232>apple9091"),
        ("a >= b", "a>=b"),
        (" b     <> a ", "b<>a"),
        ("b <> b", "b<>b"),
        ("b >> a", "NotImplemented"),
        ("b >c > a", "NotImplemented"),
        ("b > b", "b>b"),
        ("be < be", "be<be"),
        ("b >< b", "NotImplemented"),
    ],
)
def test_condition_parse(string, expect):
    assert str(Condition.parse(string)) == expect


@pytest.mark.parametrize(
    "string, expect",
    [
        ("a>b", "a>b"),
        ("a>=b", "a>=b"),
        ("b<a&a>c", "b<a&a>c"),
        (" b< a & a>c ", "b<a&a>c"),
        (" b< a && a>c ", "b<a&a>c"),
        (" b< a & a>c & a>> b", "b<a&a>c"),
        (" b< a & a>c & a> a", "b<a&a>c&a>a"),
    ],
)
def test_condition_group_parse(string, expect):
    assert str(ConditionGroup.parse(string)) == expect


@pytest.mark.parametrize(
    "string, expect",
    [
        ("a>b->c>d", "a>b -> c>d"),
        ("b<a&a>c->c>d", "b<a&a>c -> c>d"),
        (" b< a & a>c   -> d>e", "b<a&a>c -> d>e"),
        (" b< a && a>c    -> d    >e", "b<a&a>c -> d>e"),
    ],
)
def test_constrain_parse(string, expect):
    assert str(Constraint.parse(string)) == expect


@pytest.mark.parametrize(
    "string, expect",
    [
        ("a>b->c>d OR b>c -> d>e", "a>b -> c>d OR b>c -> d>e"),
        (
            """
        a>b->c>d
        b>c -> d>e""",
            "a>b -> c>d OR b>c -> d>e",
        ),
        (
            """
            a>b->c>d
            
            
            b>c -> d>e OR e>aa->c>e
            a>c-> d>e""",
            "a>b -> c>d OR b>c -> d>e OR e>aa -> c>e OR a>c -> d>e",
        ),
    ],
)
def test_constraints_parse(string, expect):
    assert str(Constraints.parse(string)) == expect
