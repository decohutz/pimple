"""Execute an explicit registered CNN plan. No freeze, promotion or holdout access."""
from __future__ import annotations

import argparse

from scripts.v2_artifacts import relative_path, repository_root, run_id
from scripts.v2_training import read_plan, select_candidate, train_run


def execute_plan(plan_relative: str):
    root = repository_root()
    path = relative_path(root, plan_relative)
    plan = read_plan(path)
    # Fail before starting if any attempt exists. No silent retries/seed replacement.
    if any((path.parent / 'attempts').glob('*.json')):
        raise ValueError('Plan already has attempts; review them explicitly before another execution')
    for recipe in plan['recipes']:
        for seed in plan['seeds']:
            print(f'START recipe={recipe} seed={seed}', flush=True)
            directory = train_run(root, path, recipe, seed)
            print(f'COMPLETED {directory.relative_to(root).as_posix()}', flush=True)
    output = root / 'reports/experimental_v2/selections' / run_id(plan['experiment_id'])
    selection = select_candidate(root, path, output)
    print(f'SELECTION {output.relative_to(root).as_posix()}', flush=True)
    print(f'CANDIDATE {selection["candidate_run_id"]}', flush=True)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, help='Repository-relative plan.json path')
    args = parser.parse_args()
    execute_plan(args.plan)
