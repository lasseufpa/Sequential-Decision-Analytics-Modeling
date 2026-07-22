"""Chapter 8 (energy storage) — forward approximate dynamic programming.

Implements section 8.6.4 / figure 8.7 with pysdm:

* ``S_t = (R_t, p_t)``; decision x in {-1, 0, +1} (sell / hold / buy);
* post-decision state ``S^x_t = (R_t + x, p_t)`` (``post_decision_state``);
* ``ForwardADPPolicy.fit`` runs the N×M×T forward/backward loop of fig. 8.7;
* evaluation compares ADP against myopic baselines under common random
  numbers, and re-evaluates on *recorded* price paths via ``DatasetSource``
  (the "external data" path — a real price feed would use CallableSource).

Run:  python examples/ch08_energy_storage_adp.py
"""

import numpy as np

import pysdm as sd

HORIZON = 24  # one decision per hour


class StorageState(sd.State):
    resource: int  # R_t: units in storage
    price: int     # p_t: current market price


class EnergyStorageModel(sd.Model):
    """Buy/hold/sell one unit per period against a mean-reverting-ish price."""

    def __init__(self, r_max=5, p_min=1, p_max=20, p_init=10):
        self.r_max = r_max
        self.p_min = p_min
        self.p_max = p_max
        self.p_init = p_init

    def initial_state(self):
        return StorageState(resource=0, price=self.p_init)

    def exogenous_info(self, state, decision, rng):
        return int(rng.integers(-2, 3))  # price shock in {-2..+2}

    def transition(self, state, decision, exog_info):
        return StorageState(
            resource=state.resource + decision,
            price=int(np.clip(state.price + exog_info, self.p_min, self.p_max)),
        )

    def objective(self, state, decision, exog_info):
        return -state.price * decision  # buy: -p, sell: +p

    def decision_space(self, state):
        return [x for x in (-1, 0, 1) if 0 <= state.resource + x <= self.r_max]

    def post_decision_state(self, state, decision):
        return state.replace(resource=state.resource + decision)


class DoNothing(sd.Policy):
    def decide(self, state, model=None):
        return 0


class MyopicPolicy(sd.Policy):
    """Argmax of the immediate contribution only — sells whenever possible."""

    def decide(self, state, model=None):
        return max(model.decision_space(state), key=lambda x: model.contribution(state, x))


class BandPolicy(sd.Policy):
    """Plausible hand-made rule: buy below ``low``, sell above ``high``.

    Loses money here: it ignores the end of the horizon and gets stuck holding
    inventory it paid for — exactly the failure the time-indexed value
    function of forward ADP learns to avoid.
    """

    def __init__(self, low=6, high=14):
        self.low = low
        self.high = high

    def decide(self, state, model=None):
        space = model.decision_space(state)
        if state.price <= self.low and 1 in space:
            return 1
        if state.price >= self.high and -1 in space:
            return -1
        return 0


def main():
    model = EnergyStorageModel()

    # ---- train (figure 8.7) ------------------------------------------- #
    adp = sd.ForwardADPPolicy(exploration=0.2, random_state=0)
    adp.fit(model, horizon=HORIZON, n_iterations=50, n_samples=20)
    per_iteration = adp.training_rewards_.mean(axis=1)
    print(f"training: iteration 1 mean path reward = {per_iteration[0]:7.2f}")
    print(f"training: iteration {len(per_iteration)} mean path reward = {per_iteration[-1]:7.2f}")
    print(f"value table size: {len(adp._vf)} entries\n")

    # ---- evaluate under common random numbers -------------------------- #
    policies = {
        "Forward ADP": adp,
        "Band 6/14": BandPolicy(low=6, high=14),
        "Myopic": MyopicPolicy(),
        "Do nothing": DoNothing(),
    }
    print(f"{'policy':14s} {'mean reward':>12s} {'p5':>8s} {'p95':>8s} {'decide ms':>10s}")
    for name, policy in policies.items():
        r = sd.Engine(model, policy, horizon=HORIZON, random_state=123).run(episodes=500)
        q5, q95 = r.quantiles([0.05, 0.95])
        print(f"{name:14s} {r.mean_reward:12.2f} {q5:8.2f} {q95:8.2f} {r.mean_decision_time_ms:10.3f}")

    # ---- evaluate on recorded data (external exogenous source) --------- #
    rng = np.random.default_rng(2024)
    recorded_shocks = rng.integers(-2, 3, size=(200, HORIZON))  # e.g. loaded from a CSV
    replay = sd.DatasetSource(recorded_shocks, builder=int)
    r = sd.Engine(model, adp, horizon=HORIZON, exogenous_source=replay).run(episodes=200)
    print(f"\nForward ADP on 200 recorded price paths: mean reward = {r.mean_reward:.2f}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(np.arange(1, len(per_iteration) + 1), per_iteration)
    ax1.set(xlabel="iteration n", ylabel="mean path reward", title="Forward ADP learning curve")
    ax1.grid(alpha=0.3)
    for name, policy in policies.items():
        r = sd.Engine(model, policy, horizon=HORIZON, random_state=123).run(episodes=500)
        r.plot_cumulative(ax=ax2, label=name)
    ax2.set_title("Cumulative reward over the day")
    fig.savefig("examples/ch08_forward_adp.png", dpi=120, bbox_inches="tight")
    print("saved examples/ch08_forward_adp.png")


if __name__ == "__main__":
    main()
