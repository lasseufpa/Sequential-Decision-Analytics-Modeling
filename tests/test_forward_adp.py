"""Forward ADP (section 8.6.4) on a small energy storage problem.

State S_t = (R_t, p_t); decision x in {-1, 0, +1} (sell / hold / buy);
contribution -p_t * x (buying costs money, selling earns it). Profit requires
buying at low prices and selling at high prices — which only a policy that
values the *future* (via the post-decision state) can discover.
"""

import numpy as np
import pytest

import pysdm as sd


class StorageState(sd.State):
    resource: int
    price: int


class EnergyStorageModel(sd.Model):
    def __init__(self, r_max=3, p_min=1, p_max=10, p_init=5):
        self.r_max = r_max
        self.p_min = p_min
        self.p_max = p_max
        self.p_init = p_init

    def initial_state(self):
        return StorageState(resource=0, price=self.p_init)

    def exogenous_info(self, state, decision, rng):
        return int(rng.integers(-1, 2))  # price change in {-1, 0, +1}

    def transition(self, state, decision, exog_info):
        return StorageState(
            resource=state.resource + decision,
            price=int(np.clip(state.price + exog_info, self.p_min, self.p_max)),
        )

    def objective(self, state, decision, exog_info):
        return -state.price * decision  # buy: pay p; sell: earn p

    def decision_space(self, state):
        return [x for x in (-1, 0, 1) if 0 <= state.resource + x <= self.r_max]

    def post_decision_state(self, state, decision):
        return state.replace(resource=state.resource + decision)


@pytest.fixture
def storage_model():
    return EnergyStorageModel()


def test_unfitted_policy_refuses_to_decide(storage_model):
    policy = sd.ForwardADPPolicy()
    with pytest.raises(sd.NotFittedError, match="fit"):
        policy.decide(storage_model.initial_state(), storage_model, t=0)


def test_fit_learns_and_records_training(storage_model):
    policy = sd.ForwardADPPolicy(exploration=0.2, random_state=0)
    policy.fit(storage_model, horizon=20, n_iterations=10, n_samples=5)
    assert policy.training_rewards_.shape == (10, 5)
    assert len(policy._vf) > 0


def test_adp_beats_doing_nothing(storage_model):
    class DoNothing(sd.Policy):
        def decide(self, state, model=None):
            return 0

    policy = sd.ForwardADPPolicy(exploration=0.2, random_state=0)
    policy.fit(storage_model, horizon=20, n_iterations=30, n_samples=10)

    adp = sd.Engine(storage_model, policy, horizon=20, random_state=123).run(episodes=200)
    idle = sd.Engine(storage_model, DoNothing(), horizon=20, random_state=123).run(episodes=200)

    assert idle.mean_reward == 0.0
    assert adp.mean_reward > 0.0  # learned to buy low / sell high


def test_value_function_stepsize_is_harmonic():
    vf = sd.TabularValueFunction()
    vf.update(0, "s", 10.0)  # visit 1 -> alpha 1 -> V=10
    assert vf.predict(0, "s") == pytest.approx(10.0)
    vf.update(0, "s", 0.0)  # visit 2 -> alpha 1/2 -> V=5
    assert vf.predict(0, "s") == pytest.approx(5.0)


def test_default_key_rejects_unhashable_states():
    class ArrayState(sd.State):
        values: np.ndarray

    vf = sd.TabularValueFunction()
    with pytest.raises(TypeError, match="key"):
        vf.predict(0, ArrayState(values=np.zeros(3)))
