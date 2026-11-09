from pathlib import Path

import pandas as pd
import pytest

from shot_probability import columns
from shot_probability.benchmark import (
    BASELINE_MODEL_NAME,
    PRIOR_MODEL_NAME,
    load_benchmark_table,
    run_benchmark,
)
from synthetic import make_benchmark_table, make_test_config, write_benchmark_table

EXPECTED_MODELS = {
    "prior",
    "logistic",
    "logistic_context",
    "gradient_boosting",
    "gradient_boosting_context",
    "logistic_freeze_frame",
    "gradient_boosting_freeze_frame",
    "transformer",
}


def test_loader_joins_shots_with_their_context_one_to_one(tmp_path: Path) -> None:
    config = make_test_config(tmp_path)
    table = make_benchmark_table(shot_count=50)
    write_benchmark_table(table, config)

    loaded = load_benchmark_table(config.shots_path, config.context_path, config.freeze_frame_path)

    assert loaded[columns.EVENT_ID].tolist() == table[columns.EVENT_ID].tolist()
    assert loaded[columns.N_PRIOR_EVENTS].notna().all()
    assert loaded[columns.HAS_FREEZE_FRAME].notna().all()


def test_loader_fails_loudly_when_a_shot_has_no_context(tmp_path: Path) -> None:
    config = make_test_config(tmp_path)
    write_benchmark_table(make_benchmark_table(shot_count=50), config)
    pd.read_parquet(config.context_path).iloc[1:].to_parquet(config.context_path, index=False)

    with pytest.raises(ValueError, match="no pre-shot context"):
        load_benchmark_table(config.shots_path, config.context_path, config.freeze_frame_path)


def test_loader_asks_for_make_data_when_a_file_is_missing(tmp_path: Path) -> None:
    config = make_test_config(tmp_path)

    with pytest.raises(FileNotFoundError, match="make data"):
        load_benchmark_table(config.shots_path, config.context_path, config.freeze_frame_path)


def test_loader_fails_loudly_when_a_shot_has_no_freeze_frame(tmp_path: Path) -> None:
    config = make_test_config(tmp_path)
    write_benchmark_table(make_benchmark_table(shot_count=50), config)
    pd.read_parquet(config.freeze_frame_path).iloc[1:].to_parquet(
        config.freeze_frame_path, index=False
    )

    with pytest.raises(ValueError, match="no freeze-frame"):
        load_benchmark_table(config.shots_path, config.context_path, config.freeze_frame_path)


def test_benchmark_scores_every_model_on_the_same_shots_and_writes_outputs(tmp_path: Path) -> None:
    config = make_test_config(tmp_path)
    write_benchmark_table(make_benchmark_table(), config)

    result = run_benchmark(config)

    assert set(result.metrics["model"]) == EXPECTED_MODELS
    shots_per_split = result.metrics.groupby("split")["n_shots"].nunique()
    assert (shots_per_split == 1).all()
    assert set(result.comparison_with_baseline["model"]) == EXPECTED_MODELS - {
        PRIOR_MODEL_NAME,
        BASELINE_MODEL_NAME,
    }
    for output_name in ("metrics.csv", "log_loss_vs_logistic.csv", "calibration.png"):
        assert (config.outputs_dir / output_name).exists()


def test_baseline_metrics_do_not_change_when_context_features_change(tmp_path: Path) -> None:
    config = make_test_config(tmp_path)
    table = make_benchmark_table()
    write_benchmark_table(table, config)
    original = run_benchmark(config).metrics

    altered_table = table.assign(
        **{
            column: table[column].sample(frac=1.0, random_state=2).to_numpy()
            for column in columns.CONTEXT_SCALAR_COLUMNS
        }
    )
    write_benchmark_table(altered_table, config)
    altered = run_benchmark(config).metrics

    phase_one_models = {PRIOR_MODEL_NAME, BASELINE_MODEL_NAME}
    pd.testing.assert_frame_equal(
        original[original["model"].isin(phase_one_models)].reset_index(drop=True),
        altered[altered["model"].isin(phase_one_models)].reset_index(drop=True),
    )
