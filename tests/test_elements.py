import numpy as np
import pytest

import pysdm as sd


class PortfolioState(sd.State):
    cash: float
    mu: np.ndarray


class PriceChange(sd.ExogenousInfo):
    change: float


def test_declared_state_is_dataclass_with_helpers():
    s = PortfolioState(cash=10.0, mu=np.array([1.0, 2.0]))
    assert s.cash == 10.0
    assert PortfolioState.field_names() == ("cash", "mu")
    assert set(s.to_dict()) == {"cash", "mu"}


def test_replace_returns_new_object():
    s = PortfolioState(cash=10.0, mu=np.zeros(2))
    s2 = s.replace(cash=5.0)
    assert s2 is not s
    assert s.cash == 10.0 and s2.cash == 5.0


def test_empty_declaration_fails_loudly():
    with pytest.raises(sd.ModelDefinitionError, match="declares no fields"):

        class Empty(sd.State):
            pass


def test_exogenous_info_composite():
    w = PriceChange(change=-0.5)
    assert w.change == -0.5
