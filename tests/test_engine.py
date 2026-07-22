import numpy as np
import pytest

import pysdm as sd


def make_engine(inventory_model, threshold_policy, **kwargs):
    defaults = dict(horizon=30, random_state=42)
    defaults.update(kwargs)
    return sd.Engine(inventory_model, threshold_policy, **defaults)


def test_run_produces_result(inventory_model, threshold_policy):
    result = make_engine(inventory_model, threshold_policy).run(episodes=50)
    assert result.per_episode_reward.shape == (50,)
    assert np.isfinite(result.mean_reward)
    assert result.cumulative_by_step.shape == (30,)
    assert result.max_decision_time_ms >= result.mean_decision_time_ms >= 0
    assert len(result.history) == 50 * 30


def test_reproducibility_common_random_numbers(inventory_model, threshold_policy):
    r1 = make_engine(inventory_model, threshold_policy).run(episodes=20)
    r2 = make_engine(inventory_model, threshold_policy).run(episodes=20)
    np.testing.assert_allclose(r1.per_episode_reward, r2.per_episode_reward)


def test_missing_state_attribute_fails_before_running(inventory_model):
    engine = sd.Engine(inventory_model, sd.GreedyPolicy(), horizon=5, random_state=0)
    with pytest.raises(sd.MissingStateAttributeError, match="mu"):
        engine.run(episodes=1)


def test_history_to_frame(inventory_model, threshold_policy):
    result = make_engine(inventory_model, threshold_policy).run(episodes=3)
    frame = result.to_frame()
    assert {"episode", "t", "state.resource", "decision", "exog", "contribution"} <= set(
        frame.columns
    )
    assert len(frame) == 3 * 30


def test_custom_metric(inventory_model, threshold_policy):
    class OrderCount(sd.Metric):
        name = "order_count"

        def reset(self):
            self.count = 0

        def update(self, record):
            self.count += record.decision > 0

        def result(self):
            return self.count

    engine = make_engine(inventory_model, threshold_policy)
    engine.add_metric(OrderCount())
    result = engine.run(episodes=10)
    assert result.metrics["order_count"] > 0


def test_callbacks_observe_run(inventory_model, threshold_policy):
    seen = []

    class Spy(sd.Callback):
        def on_episode_end(self, engine, episode):
            seen.append(engine.metrics["episode_reward"]["mean"])

    make_engine(inventory_model, threshold_policy, callbacks=[Spy()]).run(episodes=5)
    assert len(seen) == 5
    assert all(np.isfinite(v) for v in seen)


def test_dataset_source_replays_exact_data(inventory_model, threshold_policy):
    paths = np.full((4, 30), 60.0)  # constant demand
    engine = make_engine(
        inventory_model, threshold_policy, exogenous_source=sd.DatasetSource(paths)
    )
    result = engine.run(episodes=4)
    assert np.all(result.to_frame()["exog"] == 60.0)
    with pytest.raises(sd.ExogenousSourceError, match="sample paths"):
        engine.run(episodes=5)  # more episodes than data


def test_callable_source_feeds_external_data(inventory_model, threshold_policy):
    calls = []

    def feed(t, state, decision):
        calls.append(t)
        return 55.0

    engine = make_engine(
        inventory_model, threshold_policy, exogenous_source=sd.CallableSource(feed)
    )
    result = engine.run(episodes=2)
    assert len(calls) == 2 * 30
    assert np.all(result.to_frame()["exog"] == 55.0)


def test_mutating_transition_warns(threshold_policy):
    class MutatingModel(sd.Model):
        def initial_state(self):
            return type("S", (), {"resource": 0.0})()

        def exogenous_info(self, state, decision, rng):
            return 1.0

        def transition(self, state, decision, exog_info):
            state.resource += decision - exog_info
            return state  # same object!

        def objective(self, state, decision, exog_info):
            return 0.0

    engine = sd.Engine(MutatingModel(), threshold_policy, horizon=3, random_state=0)
    with pytest.warns(UserWarning, match="same"):
        engine.run(episodes=1)
