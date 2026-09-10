# GraphIDS Near-Real-Time Alert Replay

This folder turns the trained GraphIDS checkpoint into a replay-style detection service.
It does not capture live packets. It replays the NF-CICIDS2018-v3 CSV in timestamp order,
runs model inference in batches, and writes an alert immediately when a flow is predicted
as an attack.

## Default Demo Model

By default, `replay_alerts.py` uses the authors' pretrained checkpoint:

```bash
pretrained/GraphIDS_NF-CSE-CIC-IDS2018-v3.ckpt
```

This is the better default for the real-time alert demo because it has much stronger
detection performance than the 1-epoch local checkpoint.

To use the locally generated 1-epoch checkpoint instead:

```bash
checkpoints/GraphIDS_NF-CSE-CIC-IDS2018-v3_fraction0_2_epoch1.ckpt
```

## Run

From the repository root:

```bash
DGLBACKEND=pytorch WANDB_MODE=disabled ./.codex-dgl-env/bin/python "real-time model/replay_alerts.py"
```

To run a shorter preview:

```bash
DGLBACKEND=pytorch WANDB_MODE=disabled ./.codex-dgl-env/bin/python "real-time model/replay_alerts.py" --max_flows 50000
```

To use the locally generated 1-epoch model:

```bash
DGLBACKEND=pytorch WANDB_MODE=disabled ./.codex-dgl-env/bin/python "real-time model/replay_alerts.py" \
  --checkpoint checkpoints/GraphIDS_NF-CSE-CIC-IDS2018-v3_fraction0_2_epoch1.ckpt
```

## Outputs

The service writes these files under `real-time model/outputs/`:

```text
predictions.csv       every replayed flow with score, threshold, pred_label, true_label
alerts.csv            only flows where pred_label = 1
alerts.jsonl          one JSON alert per line, appended immediately
alerts_state.json     current summary and recent alerts for the dashboard
alert_metrics.json    final metrics after replay finishes
```

## Time Difference Metrics

The script reports two timing metrics using `FLOW_START_MILLISECONDS`:

```text
first_detection_lag_ms =
  first predicted attack flow time - first real attack flow time

first_true_positive_detection_lag_ms =
  first correctly predicted attack flow time - first real attack flow time

first_model_attack_to_first_true_attack_delta_ms =
  first real attack flow time - first predicted attack flow time

first_conflict_to_consistent_delay_ms =
  first later correct prediction time - first prediction-conflict time
```

In this replay setup, model decision time is also recorded as wall-clock UTC, but the
meaningful dataset timing comes from the flow timestamps.

`first_model_attack_to_first_true_attack_delta_ms` is written to every row in
`alerts.csv`, so it can be opened directly in a spreadsheet alongside the alert rows.

## Dashboard

The dashboard reads `outputs/alerts_state.json` every second:

```bash
cd "real-time model"
python3 -m http.server 8765
```

Then open:

```text
http://localhost:8765/dashboard.html
```
