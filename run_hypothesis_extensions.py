from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.io import arff
from river import drift
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "micro" / "dataset_31_credit-g.arff"
OUT = ROOT / "outputs" / "research_playbook"
SEED = 42
N_STREAMS = 100
STREAM_N = 300
FALSE_STREAM_BUDGET = 0.05
COOLDOWN = 20
EVENTS = ((60, 35), (150, 35), (240, 35))
LABEL_ONSET = 100
LABEL_FLIP_PROBABILITY = 0.80
LABEL_DELAYS = (0, 25, 50)
DELTA_GRID = (0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50)
AE_QUANTILES = (0.90, 0.95, 0.975, 0.99, 0.995, 0.999)
CALIBRATION_STREAMS = 100


class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
        )

    def forward(self, values):
        return self.network(values)


def fit_autoencoder(matrix, seed):
    torch.manual_seed(seed)
    model = Autoencoder(matrix.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    loader = DataLoader(
        TensorDataset(torch.as_tensor(matrix, dtype=torch.float32)),
        batch_size=64,
        shuffle=True,
    )
    model.train()
    for _ in range(20):
        for (batch,) in loader:
            optimizer.zero_grad()
            loss = ((model(batch) - batch) ** 2).mean()
            loss.backward()
            optimizer.step()
    return model


def reconstruction_errors(model, matrix):
    model.eval()
    with torch.no_grad():
        tensor = torch.as_tensor(matrix, dtype=torch.float32)
        return ((model(tensor) - tensor) ** 2).mean(dim=1).cpu().numpy()


def episodes(indices):
    result = []
    for index in sorted(set(int(value) for value in indices)):
        if not result or index - result[-1] > COOLDOWN:
            result.append(index)
    return result


def bootstrap_matrix(matrix, rng, size=STREAM_N):
    indices = rng.integers(0, len(matrix), size=size)
    return matrix[indices]


def adwin_alerts(matrix, family_delta):
    feature_count = matrix.shape[1]
    detectors = [
        drift.ADWIN(delta=family_delta / feature_count)
        for _ in range(feature_count)
    ]
    alerts = []
    for row_index, row in enumerate(matrix):
        signals = []
        for detector, value in zip(detectors, row):
            detector.update(float(value))
            signals.append(detector.drift_detected)
        if any(signals):
            alerts.append(row_index)
    return episodes(alerts)


def loss_adwin_alerts(loss_values, family_delta, label_delay=0):
    detector = drift.ADWIN(delta=family_delta)
    alerts = []
    for arrival_index in range(label_delay, len(loss_values)):
        source_index = arrival_index - label_delay
        detector.update(float(loss_values[source_index]))
        if detector.drift_detected:
            alerts.append(arrival_index)
    return episodes(alerts)


def false_stream_rate(alert_function, streams):
    return float(
        np.mean([bool(alert_function(stream)) for stream in streams])
    )


def select_setting(candidates, rate_by_candidate):
    eligible = [
        (setting, rate)
        for setting, rate in rate_by_candidate.items()
        if rate <= FALSE_STREAM_BUDGET
    ]
    if not eligible:
        return None
    return max(eligible, key=lambda item: item[0])[0]


def paired_summary(frame, group_columns, value_columns):
    rows = []
    for keys, group in frame.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        for column in value_columns:
            values = group[column].dropna().to_numpy(dtype=float)
            row[f"{column}_mean"] = float(np.mean(values)) if len(values) else np.nan
            row[f"{column}_sd"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
            )
            row[f"{column}_n"] = int(len(values))
        rows.append(row)
    return pd.DataFrame(rows)


def make_multi_event_stream(raw_frame, transformer):
    changed = raw_frame.copy().reset_index(drop=True)
    for event_number, (onset, duration) in enumerate(EVENTS):
        end = onset + duration
        if event_number == 0:
            changed.loc[onset:end - 1, "credit_amount"] *= 1.40
            changed.loc[onset:end - 1, "duration"] *= 1.40
        elif event_number == 1:
            changed.loc[onset:end - 1, "credit_amount"] *= 1.0734
        else:
            changed.loc[onset:end - 1, "credit_amount"] *= 1.042823
    return np.asarray(transformer.transform(changed), dtype=np.float32)


def score_multiple_events(alert_indices):
    alert_list = episodes(alert_indices)
    matched = {}
    false_alerts = []
    for alert in alert_list:
        eligible = [
            (event_index, onset)
            for event_index, (onset, duration) in enumerate(EVENTS)
            if onset <= alert < onset + duration and event_index not in matched
        ]
        if eligible:
            event_index, onset = min(eligible, key=lambda pair: abs(alert - pair[1]))
            matched[event_index] = alert - onset
        else:
            false_alerts.append(alert)
    event_count = len(EVENTS)
    alert_count = len(alert_list)
    stable_observations = STREAM_N - sum(duration for _, duration in EVENTS)
    return {
        "event_recall": len(matched) / event_count,
        "detected_events": len(matched),
        "total_events": event_count,
        "mean_delay": float(np.mean(list(matched.values()))) if matched else np.nan,
        "matched_event_delays": matched,
        "false_alert_episodes": len(false_alerts),
        "false_alert_rate": len(false_alerts) / stable_observations,
        "event_precision": len(matched) / alert_count if alert_count else np.nan,
        "event_fdr": len(false_alerts) / alert_count if alert_count else np.nan,
        "alert_episodes": alert_count,
    }


def bernoulli_logloss(probabilities, labels):
    clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
    return -(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped))


def auc_score(probabilities, labels):
    if len(np.unique(labels)) != 2:
        return np.nan
    positive = probabilities[labels == 1]
    negative = probabilities[labels == 0]
    comparisons = positive[:, None] - negative[None, :]
    return float(np.mean((comparisons > 0) + 0.5 * (comparisons == 0)))


def add_ci(values, rng, repetitions=5000):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return (np.nan, np.nan)
    samples = rng.choice(values, size=(repetitions, len(values)), replace=True)
    means = samples.mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]).tolist())


def run():
    global NUMERIC
    if not DATA.is_file():
        raise FileNotFoundError(DATA)
    OUT.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED + 9000)

    raw, metadata = arff.loadarff(DATA)
    credit = pd.DataFrame(raw)
    for column in credit.select_dtypes(include=["object"]).columns:
        credit[column] = credit[column].map(
            lambda value: value.decode("utf-8")
            if isinstance(value, bytes)
            else value
        )
    target = "class"
    credit[target] = credit[target].astype(str).str.strip()
    NUMERIC = credit.drop(columns=target).select_dtypes(
        include=[np.number]
    ).columns.tolist()
    categorical = [
        column
        for column in credit.columns
        if column not in NUMERIC + [target]
    ]
    if not {"credit_amount", "duration"}.issubset(NUMERIC):
        raise ValueError("Expected numeric credit_amount and duration columns.")

    reference = credit.iloc[:700]
    stream = credit.iloc[700:]
    train = reference.iloc[:560]
    calibration = reference.iloc[560:]
    transformer = ColumnTransformer(
        [
            ("numeric", RobustScaler(), NUMERIC),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ],
        remainder="drop",
    )
    x_train = np.asarray(
        transformer.fit_transform(train.drop(columns=target)), dtype=np.float32
    )
    x_calibration = np.asarray(
        transformer.transform(calibration.drop(columns=target)), dtype=np.float32
    )
    x_stream = np.asarray(
        transformer.transform(stream.drop(columns=target)), dtype=np.float32
    )
    y_train = (train[target].to_numpy() == "good").astype(int)
    y_calibration = (calibration[target].to_numpy() == "good").astype(int)
    y_stream = (stream[target].to_numpy() == "good").astype(int)

    classifier = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        random_state=SEED,
        eval_metric="logloss",
        n_jobs=1,
    )
    classifier.fit(x_train, y_train)
    model = fit_autoencoder(x_train, SEED)

    # The 140 calibration records are split before any alert-budget tuning.
    tune_x, validation_x = x_calibration[:70], x_calibration[70:]
    tune_y, validation_y = y_calibration[:70], y_calibration[70:]
    tune_errors = reconstruction_errors(model, tune_x)
    validation_errors = reconstruction_errors(model, validation_x)
    tune_loss = bernoulli_logloss(
        classifier.predict_proba(tune_x)[:, 1], tune_y
    )
    validation_loss = bernoulli_logloss(
        classifier.predict_proba(validation_x)[:, 1], validation_y
    )

    clean_x_streams = []
    clean_validation_labels = []
    for _ in range(CALIBRATION_STREAMS):
        indices = rng.integers(0, len(validation_x), size=STREAM_N)
        clean_x_streams.append(validation_x[indices])
        clean_validation_labels.append(validation_y[indices])
    clean_error_streams = [
        reconstruction_errors(model, values) for values in clean_x_streams
    ]
    clean_loss_streams = []
    for values, labels in zip(clean_x_streams, clean_validation_labels):
        probs = classifier.predict_proba(values)[:, 1]
        clean_loss_streams.append(bernoulli_logloss(probs, labels))

    ae_candidates = {
        float(q): float(np.quantile(tune_errors, q)) for q in AE_QUANTILES
    }
    ae_rates = {
        threshold: false_stream_rate(
            lambda values, t=threshold: episodes(
                np.flatnonzero(values > t)
            ),
            clean_error_streams,
        )
        for threshold in ae_candidates.values()
    }
    selected_ae = None
    for q in AE_QUANTILES:
        threshold = ae_candidates[float(q)]
        rate = ae_rates[threshold]
        if rate <= FALSE_STREAM_BUDGET:
            selected_ae = (float(q), threshold, rate)
            break

    delta_rates = {}
    for delta in DELTA_GRID:
        delta_rates[delta] = false_stream_rate(
            lambda values, d=delta: adwin_alerts(values, d),
            clean_x_streams,
        )
    selected_input_delta = select_setting(DELTA_GRID, delta_rates)
    selected_loss_delta = max(DELTA_GRID)

    calibration_rows = []
    for q, threshold in ae_candidates.items():
        calibration_rows.append(
            {
                "detector": "Autoencoder-input",
                "setting": f"error-quantile-{q:g}",
                "threshold_or_delta": threshold,
                "clean_stream_false_alert_rate": ae_rates[threshold],
                "budget": FALSE_STREAM_BUDGET,
                "selected": bool(
                    selected_ae is not None and q == selected_ae[0]
                ),
            }
        )
    for delta, rate in delta_rates.items():
        calibration_rows.append(
            {
                "detector": "ADWIN-input",
                "setting": f"family-delta-{delta:g}",
                "threshold_or_delta": delta,
                "clean_stream_false_alert_rate": rate,
                "budget": FALSE_STREAM_BUDGET,
                "selected": delta == selected_input_delta,
            }
        )
    loss_delta_rates = {}
    for delta in DELTA_GRID:
        loss_delta_rates[delta] = false_stream_rate(
            lambda values, d=delta: loss_adwin_alerts(values, d),
            clean_loss_streams,
        )
        calibration_rows.append(
            {
                "detector": "ADWIN-log-loss",
                "setting": f"delta-{delta:g}",
                "threshold_or_delta": delta,
                "clean_stream_false_alert_rate": loss_delta_rates[delta],
                "budget": FALSE_STREAM_BUDGET,
                "selected": False,
            }
        )
    selected_loss_delta = select_setting(DELTA_GRID, loss_delta_rates)
    for row in calibration_rows:
        if row["detector"] == "ADWIN-log-loss":
            row["selected"] = row["threshold_or_delta"] == selected_loss_delta
    pd.DataFrame(calibration_rows).to_csv(
        OUT / "extension_false_alert_calibration.csv", index=False
    )

    if selected_ae is None:
        raise RuntimeError(
            "No autoencoder threshold met the pre-registered clean-stream "
            f"false-alert budget of {FALSE_STREAM_BUDGET:.1%}."
        )
    if selected_input_delta is None:
        raise RuntimeError(
            "No input ADWIN delta met the pre-registered clean-stream "
            f"false-alert budget of {FALSE_STREAM_BUDGET:.1%}."
        )
    if selected_loss_delta is None:
        raise RuntimeError(
            "No loss ADWIN delta met the pre-registered clean-stream "
            f"false-alert budget of {FALSE_STREAM_BUDGET:.1%}."
        )

    ae_quantile, ae_threshold, ae_cal_rate = selected_ae
    ae_calibration_values = {
        "detector": "Autoencoder-input",
        "selected_setting": f"error-quantile-{ae_quantile:g}",
        "threshold_or_delta": ae_threshold,
        "clean_stream_false_alert_rate": ae_cal_rate,
    }
    input_adwin_calibration = {
        "detector": "ADWIN-input",
        "selected_setting": f"family-delta-{selected_input_delta:g}",
        "threshold_or_delta": selected_input_delta,
        "clean_stream_false_alert_rate": delta_rates[selected_input_delta],
    }
    loss_adwin_calibration = {
        "detector": "ADWIN-log-loss",
        "selected_setting": f"delta-{selected_loss_delta:g}",
        "threshold_or_delta": selected_loss_delta,
        "clean_stream_false_alert_rate": loss_delta_rates[selected_loss_delta],
    }
    pd.DataFrame(
        [
            ae_calibration_values,
            input_adwin_calibration,
            loss_adwin_calibration,
        ]
    ).to_csv(OUT / "extension_selected_settings.csv", index=False)

    multi_x_rows = []
    raw_stream = stream.drop(columns=target).reset_index(drop=True)
    for stream_id in range(N_STREAMS):
        indices = rng.integers(0, len(raw_stream), size=STREAM_N)
        sampled = raw_stream.iloc[indices].reset_index(drop=True)
        unchanged = np.asarray(
            transformer.transform(sampled), dtype=np.float32
        )
        changed = make_multi_event_stream(sampled, transformer)
        clean_ae_hits = np.flatnonzero(
            reconstruction_errors(model, unchanged) > ae_threshold
        )
        clean_adwin_hits = adwin_alerts(unchanged, selected_input_delta)
        ae_hits = np.flatnonzero(
            reconstruction_errors(model, changed) > ae_threshold
        )
        adwin_hits = adwin_alerts(changed, selected_input_delta)
        for detector_name, hits, clean_indices in [
            ("Autoencoder-input", ae_hits, clean_ae_hits),
            ("ADWIN-input", adwin_hits, clean_adwin_hits),
        ]:
            result = score_multiple_events(hits)
            delays = result.pop("matched_event_delays")
            clean_hits = episodes(clean_indices)
            clean_false_episodes = len(clean_hits)
            for event_index, (onset, duration) in enumerate(EVENTS):
                multi_x_rows.append(
                    {
                        "stream_id": stream_id,
                        "detector": detector_name,
                        "event_id": event_index + 1,
                        "event_onset": onset,
                        "event_duration": duration,
                        "detected": event_index in delays,
                        "delay": delays.get(event_index, np.nan),
                        "false_alert_episodes_stream": result[
                            "false_alert_episodes"
                        ],
                        "false_alert_rate_stream": result["false_alert_rate"],
                        "stream_event_recall": result["event_recall"],
                        "stream_event_precision": result["event_precision"],
                        "stream_event_fdr": result["event_fdr"],
                        "stream_alert_episodes": result["alert_episodes"],
                        "clean_operational_alert_episodes": len(clean_hits),
                        "clean_operational_false_alert_rate": (
                            clean_false_episodes / STREAM_N
                        ),
                        "clean_operational_stream_had_alert": bool(clean_hits),
                    }
                )
    multi_x = pd.DataFrame(multi_x_rows)
    multi_x.to_csv(OUT / "extension_multi_event_trials.csv", index=False)
    multi_summary = (
        multi_x.groupby("detector")
        .agg(
            streams=("stream_id", "nunique"),
            events=("detected", "size"),
            event_recall=("detected", "mean"),
            mean_delay=("delay", "mean"),
            delay_sd=("delay", "std"),
            event_precision=("stream_event_precision", "mean"),
            event_fdr=("stream_event_fdr", "mean"),
            false_alert_rate=("false_alert_rate_stream", "mean"),
            mean_alert_episodes=("stream_alert_episodes", "mean"),
            clean_operational_false_alert_rate=(
                "clean_operational_false_alert_rate", "mean"
            ),
            clean_operational_stream_alert_rate=(
                "clean_operational_stream_had_alert", "mean"
            ),
        )
        .reset_index()
    )
    multi_summary.to_csv(OUT / "extension_multi_event_summary.csv", index=False)
    recall_wide = (
        multi_x.groupby(["stream_id", "detector"])["detected"]
        .mean()
        .unstack("detector")
    )
    recall_difference = (
        recall_wide["Autoencoder-input"] - recall_wide["ADWIN-input"]
    )
    precision_wide = (
        multi_x.groupby(["stream_id", "detector"])["stream_event_precision"]
        .mean()
        .unstack("detector")
    )
    precision_pair = precision_wide.dropna()
    precision_difference = (
        precision_pair["Autoencoder-input"] - precision_pair["ADWIN-input"]
    )
    ci_rng = np.random.default_rng(SEED + 12000)
    recall_ci = add_ci(recall_difference.to_numpy(), ci_rng)
    precision_ci = add_ci(precision_difference.to_numpy(), ci_rng)
    multi_comparison = pd.DataFrame(
        [
            {
                "metric": "per-stream event recall difference (AE - ADWIN)",
                "paired_n_streams": len(recall_difference),
                "mean_difference": float(recall_difference.mean()),
                "bootstrap_ci_95_low": recall_ci[0],
                "bootstrap_ci_95_high": recall_ci[1],
            },
            {
                "metric": "per-stream event precision difference (AE - ADWIN)",
                "paired_n_streams": len(precision_difference),
                "mean_difference": (
                    float(precision_difference.mean())
                    if len(precision_difference)
                    else np.nan
                ),
                "bootstrap_ci_95_low": precision_ci[0],
                "bootstrap_ci_95_high": precision_ci[1],
            },
        ]
    )
    multi_comparison.to_csv(
        OUT / "extension_multi_event_comparison.csv", index=False
    )

    label_rows = []
    p_stream = classifier.predict_proba(x_stream)[:, 1]
    for stream_id in range(N_STREAMS):
        sampled_indices = rng.integers(0, len(x_stream), size=STREAM_N)
        sampled_x = x_stream[sampled_indices]
        original_y = y_stream[sampled_indices].copy()
        probabilities = p_stream[sampled_indices]
        flipped_y = original_y.copy()
        flip_mask = (
            (np.arange(STREAM_N) >= LABEL_ONSET)
            & (rng.random(STREAM_N) < LABEL_FLIP_PROBABILITY)
        )
        flipped_y[flip_mask] = 1 - flipped_y[flip_mask]
        losses = bernoulli_logloss(probabilities, flipped_y)
        label_shift_x = sampled_x.copy()

        for detector_name, hits in [
            ("Autoencoder-input", np.flatnonzero(
                reconstruction_errors(model, label_shift_x) > ae_threshold
            )),
            ("ADWIN-input", adwin_alerts(label_shift_x, selected_input_delta)),
        ]:
            alert_list = episodes(hits)
            after = [index for index in alert_list if index >= LABEL_ONSET]
            before = [index for index in alert_list if index < LABEL_ONSET]
            label_rows.append(
                {
                    "stream_id": stream_id,
                    "detector": detector_name,
                    "label_delay": 0,
                    "label_event_recall": np.nan,
                    "detection_delay_from_onset": np.nan,
                    "input_alert_after_label_onset": float(bool(after)),
                    "input_alert_episodes": len(alert_list),
                    "counterfactual_same_input_alerts": True,
                    "false_alert_episodes_before_onset": len(before),
                    "false_alert_rate_before_onset": len(before) / LABEL_ONSET,
                    "label_event_precision": np.nan,
                    "event_fdr": np.nan,
                    "auc_before": auc_score(
                        probabilities[:LABEL_ONSET], original_y[:LABEL_ONSET]
                    ),
                    "auc_after_shift": auc_score(
                        probabilities[LABEL_ONSET:], flipped_y[LABEL_ONSET:]
                    ),
                    "logloss_before": float(np.mean(losses[:LABEL_ONSET])),
                    "logloss_after_shift": float(np.mean(losses[LABEL_ONSET:])),
                    "flipped_outcomes": int(flip_mask.sum()),
                }
            )

        for label_delay in LABEL_DELAYS:
            hits = loss_adwin_alerts(
                losses, selected_loss_delta, label_delay=label_delay
            )
            after_available = [
                index
                for index in hits
                if index >= LABEL_ONSET + label_delay
            ]
            before_available = [
                index
                for index in hits
                if index < LABEL_ONSET + label_delay
            ]
            first = after_available[0] if after_available else np.nan
            matched_event_count = int(bool(after_available))
            unmatched_alerts = len(hits) - matched_event_count
            label_rows.append(
                {
                    "stream_id": stream_id,
                    "detector": "ADWIN-log-loss",
                    "label_delay": label_delay,
                    "label_event_recall": float(bool(after_available)),
                    "detection_delay_from_onset": (
                        first - LABEL_ONSET if after_available else np.nan
                    ),
                    "delay_after_label_available": (
                        first - (LABEL_ONSET + label_delay)
                        if after_available
                        else np.nan
                    ),
                    "false_alert_episodes_before_onset": len(before_available),
                    "false_alert_rate_before_onset": len(before_available)
                    / max(LABEL_ONSET + label_delay, 1),
                    "label_event_precision": (
                        matched_event_count / len(hits) if hits else np.nan
                    ),
                    "event_fdr": unmatched_alerts / len(hits) if hits else np.nan,
                    "auc_before": auc_score(
                        probabilities[:LABEL_ONSET], original_y[:LABEL_ONSET]
                    ),
                    "auc_after_shift": auc_score(
                        probabilities[LABEL_ONSET:], flipped_y[LABEL_ONSET:]
                    ),
                    "logloss_before": float(np.mean(losses[:LABEL_ONSET])),
                    "logloss_after_shift": float(np.mean(losses[LABEL_ONSET:])),
                    "flipped_outcomes": int(flip_mask.sum()),
                }
            )

    label_trials = pd.DataFrame(label_rows)
    label_trials.to_csv(OUT / "extension_outcome_shift_trials.csv", index=False)
    label_summary = (
        label_trials.groupby(["detector", "label_delay"], dropna=False)
        .agg(
            streams=("stream_id", "nunique"),
            event_recall=("label_event_recall", "mean"),
            mean_detection_delay=("detection_delay_from_onset", "mean"),
            delay_sd=("detection_delay_from_onset", "std"),
            mean_delay_after_label_available=("delay_after_label_available", "mean"),
            event_precision=("label_event_precision", "mean"),
            event_fdr=("event_fdr", "mean"),
            mean_pre_onset_false_alert_rate=("false_alert_rate_before_onset", "mean"),
            mean_auc_before=("auc_before", "mean"),
            mean_auc_after_shift=("auc_after_shift", "mean"),
            mean_logloss_before=("logloss_before", "mean"),
            mean_logloss_after_shift=("logloss_after_shift", "mean"),
            mean_flipped_outcomes=("flipped_outcomes", "mean"),
            input_alert_after_label_onset=("input_alert_after_label_onset", "mean"),
            input_alert_episodes=("input_alert_episodes", "mean"),
        )
        .reset_index()
    )
    label_summary.to_csv(OUT / "extension_outcome_shift_summary.csv", index=False)

    print("Shared stream-level false-alert budget:", FALSE_STREAM_BUDGET)
    print("Calibration set split: 70 tuning / 70 validation records")
    print("Bootstrap test streams:", N_STREAMS, "from the fixed 300-record stream")
    print("Selected settings:")
    print(pd.read_csv(OUT / "extension_selected_settings.csv").to_string(index=False))
    print("\nRepeated-event feature-shift results:")
    print(multi_summary.round(4).to_string(index=False))
    print("\nControlled outcome-shift results:")
    print(label_summary.round(4).to_string(index=False))
    print("\nFiles written to:", OUT)
    return {
        "calibration": pd.read_csv(OUT / "extension_selected_settings.csv"),
        "multi_event": multi_summary,
        "multi_event_comparison": multi_comparison,
        "outcome_shift": label_summary,
    }


if __name__ == "__main__":
    run()
