"""pysdm — sequential decision modeling under uncertainty.

A small, composable implementation of Warren B. Powell's Universal Modeling
Framework: describe the *problem* once (:class:`Model` + declarative
:class:`State`/:class:`ExogenousInfo`), plug in any *strategy*
(:class:`Policy`), and let the :class:`Engine` run, measure and record it.

Quick start::

    import pysdm as sd

    class InventoryState(sd.State):
        resource: float

    class InventoryModel(sd.Model):
        def __init__(self, price, cost, demand_mean, demand_std):
            self.price, self.cost = price, cost
            self.demand_mean, self.demand_std = demand_mean, demand_std
        def initial_state(self):
            return InventoryState(resource=0.0)
        def exogenous_info(self, state, decision, rng):
            return max(0.0, rng.normal(self.demand_mean, self.demand_std))
        def transition(self, state, decision, exog_info):
            return state.replace(resource=max(0.0, state.resource + decision - exog_info))
        def objective(self, state, decision, exog_info):
            sold = min(state.resource + decision, exog_info)
            return self.price * sold - self.cost * decision

    engine = sd.Engine(
        model=InventoryModel(price=45, cost=30, demand_mean=60, demand_std=10),
        policy=sd.ThresholdPolicy(theta_min=80, theta_max=110),
        horizon=30,
        random_state=42,
    )
    result = engine.run(episodes=1000)
    print(result.mean_reward, result.quantiles([0.05, 0.95]))
"""

from ._random import check_random_state, spawn_generators
from .callbacks import Callback, ProgressLogger
from .elements import Decision, ExogenousInfo, State
from .engine import Engine
from .exceptions import (
    ExogenousSourceError,
    InvalidDecisionError,
    MissingStateAttributeError,
    ModelDefinitionError,
    NotFittedError,
    PysdmError,
)
from .exogenous import CallableSource, DatasetSource, ExogenousSource, ModelSource
from .history import History, StepRecord
from .metrics import CumulativeByStep, DecisionTime, EpisodeReward, Metric, MetricSet
from .model import Model
from .policies import (
    ForwardADPPolicy,
    GreedyPolicy,
    IEPolicy,
    Policy,
    TabularValueFunction,
    ThompsonSamplingPolicy,
    ThresholdPolicy,
    UCBPolicy,
)
from .result import RunResult

__version__ = "0.1.0"

__all__ = [
    # elements
    "State",
    "Decision",
    "ExogenousInfo",
    # problem
    "Model",
    # exogenous data sources
    "ExogenousSource",
    "ModelSource",
    "DatasetSource",
    "CallableSource",
    # policies
    "Policy",
    "ThresholdPolicy",
    "GreedyPolicy",
    "UCBPolicy",
    "IEPolicy",
    "ThompsonSamplingPolicy",
    "ForwardADPPolicy",
    "TabularValueFunction",
    # engine & results
    "Engine",
    "RunResult",
    "History",
    "StepRecord",
    # metrics & callbacks
    "Metric",
    "MetricSet",
    "EpisodeReward",
    "CumulativeByStep",
    "DecisionTime",
    "Callback",
    "ProgressLogger",
    # utilities
    "check_random_state",
    "spawn_generators",
    # exceptions
    "PysdmError",
    "ModelDefinitionError",
    "MissingStateAttributeError",
    "InvalidDecisionError",
    "ExogenousSourceError",
    "NotFittedError",
    "__version__",
]
