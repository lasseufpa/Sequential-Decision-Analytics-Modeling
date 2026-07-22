import pytest

import pysdm as sd


class InventoryState(sd.State):
    resource: float


class InventoryModel(sd.Model):
    """Pizza inventory problem (chapter 1)."""

    def __init__(self, price=45.0, cost=30.0, demand_mean=60.0, demand_std=10.0):
        self.price = price
        self.cost = cost
        self.demand_mean = demand_mean
        self.demand_std = demand_std

    def initial_state(self):
        return InventoryState(resource=0.0)

    def exogenous_info(self, state, decision, rng):
        return max(0.0, float(rng.normal(self.demand_mean, self.demand_std)))

    def transition(self, state, decision, exog_info):
        return state.replace(resource=max(0.0, state.resource + decision - exog_info))

    def objective(self, state, decision, exog_info):
        sold = min(state.resource + decision, exog_info)
        return self.price * sold - self.cost * decision


@pytest.fixture
def inventory_model():
    return InventoryModel()


@pytest.fixture
def threshold_policy():
    return sd.ThresholdPolicy(theta_min=80, theta_max=110)
