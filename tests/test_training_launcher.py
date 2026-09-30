"""The launcher must execute declared seeds once and never silently retry."""
import pytest

from scripts import run_training_plan as launcher


def test_launcher_refuses_existing_attempt_before_training(tmp_path, monkeypatch):
    plan = tmp_path / 'plan.json'
    plan.write_text('{}')
    attempts = tmp_path / 'attempts'
    attempts.mkdir()
    (attempts / 'seed42.json').write_text('{}')
    monkeypatch.setattr(launcher, 'repository_root', lambda: tmp_path)
    monkeypatch.setattr(launcher, 'read_plan', lambda _: {'recipes': {'recipe': {}}, 'seeds': [42]})
    monkeypatch.setattr(launcher, 'train_run', lambda *_: pytest.fail('Must not repeat a claimed seed'))
    with pytest.raises(ValueError, match='already has attempts'):
        launcher.execute_plan('plan.json')


def test_launcher_selects_only_after_all_declared_runs(tmp_path, monkeypatch):
    events = []
    plan = {'recipes': {'recipe_a': {}, 'recipe_b': {}}, 'seeds': [42, 43], 'experiment_id': 'test'}
    monkeypatch.setattr(launcher, 'repository_root', lambda: tmp_path)
    monkeypatch.setattr(launcher, 'read_plan', lambda _: plan)
    def train(root, path, recipe, seed):
        events.append((recipe, seed))
        return tmp_path / f'{recipe}_{seed}'
    def select(*_):
        assert events == [('recipe_a', 42), ('recipe_a', 43), ('recipe_b', 42), ('recipe_b', 43)]
        events.append('selection')
        return {'candidate_run_id': 'test_candidate'}
    monkeypatch.setattr(launcher, 'train_run', train)
    monkeypatch.setattr(launcher, 'select_candidate', select)
    launcher.execute_plan('plan.json')
    assert events[-1] == 'selection'


def test_failed_run_stops_remaining_seeds_and_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, 'repository_root', lambda: tmp_path)
    monkeypatch.setattr(launcher, 'read_plan', lambda _: {'recipes': {'r': {}}, 'seeds': [42, 43]})
    def fail(*_):
        raise RuntimeError('training failed')
    monkeypatch.setattr(launcher, 'train_run', fail)
    monkeypatch.setattr(launcher, 'select_candidate', lambda *_: pytest.fail('Cannot select after failure'))
    with pytest.raises(RuntimeError, match='training failed'):
        launcher.execute_plan('plan.json')
