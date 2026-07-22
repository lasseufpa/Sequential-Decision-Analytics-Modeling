"""Chapter 4 (diabetes medication bandits) rebuilt on top of pysdm.

What this example demonstrates, mapped to the library design:

* Beliefs live in a declarative ``State`` (mu/beta/counts/n) — the model's
  transition *is* the Bayesian update.
* The hidden truth (each patient's real drug response) is **not** in the
  model: it lives in an ``ExogenousSource`` that samples a new truth per
  episode. Swapping simulated truth for a real clinical dataset would only
  change the source.
* A custom metric (``CorrectIdentification``) plugs into the engine.
* Policies are compared under common random numbers (same seed).

Run:  python examples/ch04_diabetes_bandits.py
"""

import numpy as np

import pysdm as sd

DRUGS = ["Metformin", "Sensitizers", "Secretagogues", "Alpha-glucosidase", "Peptide analogs"]
MU_PRIOR = np.array([0.32, 0.28, 0.30, 0.26, 0.21])
SIGMA_PRIOR = np.array([0.12, 0.09, 0.17, 0.15, 0.11])
SIGMA_W = 0.05


class BeliefState(sd.State):
    """S_n = (mu, beta, counts, n): Bayesian beliefs about each drug."""

    mu: np.ndarray
    beta: np.ndarray
    counts: np.ndarray
    n: int

    @property
    def sigma(self):
        return 1.0 / np.sqrt(self.beta)


class DiabetesModel(sd.Model):
    """The doctor's problem: which drug to try next for this patient."""

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
        # the model's own uncertainty view: the belief's predictive distribution
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
        return exog_info  # observed A1c reduction


class PatientTruthSource(sd.ExogenousSource):
    """Hidden truth outside the model: each episode (patient) draws real drug
    effects; observations are truth + measurement noise. Replace this with a
    ``DatasetSource``/``CallableSource`` to plug in real clinical data."""

    def __init__(self, mu_prior, sigma_prior, sigma_W):
        self.mu_prior = np.asarray(mu_prior, dtype=float)
        self.sigma_prior = np.asarray(sigma_prior, dtype=float)
        self.sigma_W = sigma_W
        self.mu_true = None

    def reset(self, episode, rng):
        self.mu_true = rng.normal(self.mu_prior, self.sigma_prior)

    def generate(self, t, state, decision, rng):
        return float(self.mu_true[decision] + rng.normal(0.0, self.sigma_W))


class CorrectIdentification(sd.Metric):
    """Fraction of episodes whose final belief points at the truly best drug."""

    name = "correct_identification"

    def __init__(self, truth_source):
        self.truth_source = truth_source

    def reset(self):
        self.hits = 0
        self.episodes = 0
        self._last = None

    def update(self, record):
        self._last = record

    def on_episode_end(self, episode):
        self.episodes += 1
        believed = int(np.argmax(self._last.next_state.mu))
        actual = int(np.argmax(self.truth_source.mu_true))
        self.hits += believed == actual

    def result(self):
        return self.hits / self.episodes if self.episodes else float("nan")


def main():
    model = DiabetesModel(MU_PRIOR, SIGMA_PRIOR, SIGMA_W)
    policies = {
        "Greedy": sd.GreedyPolicy(),
        "UCB (theta=0.5)": sd.UCBPolicy(theta=0.5),
        "IE (theta=1.5)": sd.IEPolicy(theta=1.5),
        "Thompson (theta=1.0)": sd.ThompsonSamplingPolicy(theta=1.0, random_state=7),
    }

    print(f"{'policy':22s} {'mean reward':>12s} {'std':>8s} {'p5':>8s} {'p95':>8s} {'correct':>8s}")
    results = {}
    for name, policy in policies.items():
        source = PatientTruthSource(MU_PRIOR, SIGMA_PRIOR, SIGMA_W)
        engine = sd.Engine(
            model,
            policy,
            horizon=30,
            random_state=0,  # common random numbers: same patients for everyone
            exogenous_source=source,
            metrics=[CorrectIdentification(source)],
        )
        r = engine.run(episodes=2000)
        results[name] = r
        q5, q95 = r.quantiles([0.05, 0.95])
        print(
            f"{name:22s} {r.mean_reward:12.3f} {r.std_reward:8.3f} {q5:8.3f} {q95:8.3f} "
            f"{r.metrics['correct_identification']:8.1%}"
        )

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    ax = None
    for name, r in results.items():
        ax = r.plot_cumulative(ax=ax, label=name)
    ax.set_title("Chapter 4 — cumulative A1c reduction by policy (2000 patients)")
    ax.figure.savefig("examples/ch04_cumulative.png", dpi=120, bbox_inches="tight")
    print("saved examples/ch04_cumulative.png")


if __name__ == "__main__":
    main()
