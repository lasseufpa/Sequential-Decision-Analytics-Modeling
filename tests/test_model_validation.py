import pytest

import pysdm as sd


def test_wrong_signature_fails_at_class_definition():
    with pytest.raises(sd.ModelDefinitionError, match="transition"):

        class BadModel(sd.Model):
            def initial_state(self):
                return 0

            def exogenous_info(self, state, decision, rng):
                return 0.0

            def transition(self, state):  # missing decision, exog_info
                return state

            def objective(self, state, decision, exog_info):
                return 0.0


def test_missing_method_fails_at_instantiation():
    class Incomplete(sd.Model):
        def initial_state(self):
            return 0

        def exogenous_info(self, state, decision, rng):
            return 0.0

        def transition(self, state, decision, exog_info):
            return state

        # objective missing

    with pytest.raises(TypeError, match="objective"):
        Incomplete()


def test_optional_hooks_have_helpful_defaults(inventory_model):
    with pytest.raises(sd.ModelDefinitionError, match="decision_space"):
        inventory_model.decision_space(None)
    with pytest.raises(sd.ModelDefinitionError, match="post_decision_state"):
        inventory_model.post_decision_state(None, None)


def test_engine_rejects_class_instead_of_instance(inventory_model):
    with pytest.raises(TypeError, match="instance"):
        sd.Engine(model=inventory_model, policy=sd.GreedyPolicy, horizon=5)


def test_engine_rejects_invalid_model(threshold_policy):
    class NotAModel:
        pass

    with pytest.raises(sd.ModelDefinitionError, match="missing callable"):
        sd.Engine(model=NotAModel(), policy=threshold_policy, horizon=5)
