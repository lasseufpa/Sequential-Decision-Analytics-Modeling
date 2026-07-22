import numpy as np
import pytest

import pysdm as sd


class BeliefState(sd.State):
    """Bayesian beliefs over alternatives: follows the naming convention
    required by the bandit policies (mu/sigma/counts/n)."""

    mu: np.ndarray
    beta: np.ndarray
    counts: np.ndarray
    n: int

    @property
    def sigma(self):
        return 1.0 / np.sqrt(self.beta)


class BanditModel(sd.Model):
    """Gaussian bandit with beliefs in the state (chapter 4 pattern)."""

    def __init__(self, mu_prior, sigma_prior, sigma_W):
        self.mu_prior = np.asarray(mu_prior, dtype=float)
        self.sigma_prior = np.asarray(sigma_prior, dtype=float)
        self.sigma_W = sigma_W

    def initial_state(self):
        return BeliefState(
            mu=self.mu_prior.copy(),
            beta=1.0 / self.sigma_prior**2,
            counts=np.zeros(len(self.mu_prior), dtype=int),
            n=0,
        )

    def exogenous_info(self, state, decision, rng):
        # predictive distribution of the chosen alternative
        std = np.sqrt(state.sigma[decision] ** 2 + self.sigma_W**2)
        return float(rng.normal(state.mu[decision], std))

    def transition(self, state, decision, exog_info):
        beta_W = 1.0 / self.sigma_W**2
        mu, beta, counts = state.mu.copy(), state.beta.copy(), state.counts.copy()
        mu[decision] = (beta[decision] * mu[decision] + beta_W * exog_info) / (
            beta[decision] + beta_W
        )
        beta[decision] += beta_W
        counts[decision] += 1
        return BeliefState(mu=mu, beta=beta, counts=counts, n=state.n + 1)

    def objective(self, state, decision, exog_info):
        return exog_info


@pytest.fixture
def bandit_model():
    return BanditModel(
        mu_prior=[0.32, 0.28, 0.30, 0.26, 0.21],
        sigma_prior=[0.12, 0.09, 0.17, 0.15, 0.11],
        sigma_W=0.05,
    )


@pytest.mark.parametrize(
    "policy",
    [
        sd.GreedyPolicy(),
        sd.UCBPolicy(theta=0.5),
        sd.IEPolicy(theta=1.5),
        sd.ThompsonSamplingPolicy(theta=1.0, random_state=7),
    ],
    ids=["greedy", "ucb", "ie", "thompson"],
)
def test_bandit_policies_run(bandit_model, policy):
    engine = sd.Engine(bandit_model, policy, horizon=30, random_state=0)
    result = engine.run(episodes=30)
    decisions = result.to_frame()["decision"]
    assert decisions.between(0, 4).all()
    assert np.isfinite(result.mean_reward)


def test_threshold_policy_orders_up_to():
    policy = sd.ThresholdPolicy(theta_min=80, theta_max=110)
    state = type("S", (), {"resource": 50.0})()
    assert policy.decide(state) == 60.0
    state.resource = 90.0
    assert policy.decide(state) == 0.0


def test_threshold_policy_validates_thetas():
    with pytest.raises(ValueError, match="theta_max"):
        sd.ThresholdPolicy(theta_min=100, theta_max=50)


def test_get_set_params_roundtrip():
    policy = sd.UCBPolicy(theta=1.0)
    assert policy.get_params() == {"theta": 1.0}
    policy.set_params(theta=2.5)
    assert policy.theta == 2.5
    with pytest.raises(ValueError, match="Unknown parameter"):
        policy.set_params(gamma=1.0)


def test_tags_declare_policy_class():
    assert sd.UCBPolicy().tags()["policy_class"] == "CFA"
    assert sd.ThresholdPolicy(0, 1).tags()["policy_class"] == "PFA"
