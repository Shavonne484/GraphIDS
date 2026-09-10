import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("DGLBACKEND", "pytorch")

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.GraphIDS import GraphIDS
from utils.dataloaders import GraphDataLoader, NetFlowDataset, SequentialDataset, collate_fn
from utils.trainers import calculate_errors


METADATA_COLUMNS = [
    "FLOW_START_MILLISECONDS",
    "FLOW_END_MILLISECONDS",
    "IPV4_SRC_ADDR",
    "IPV4_DST_ADDR",
    "Label",
    "Attack",
]


def repo_path(path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def flatten_config(config_path):
    with open(config_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    config = {}
    for key, value in raw.items():
        if isinstance(value, dict) and "value" in value:
            config[key] = value["value"]
        else:
            config[key] = value
    return config


def fraction_key(fraction):
    return "none" if fraction is None else str(fraction).replace(".", "_")


def load_or_build_test_metadata(args):
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / f"test_metadata_{args.dataset}_{fraction_key(args.fraction)}_seed{args.seed}.csv"
    if cache_path.exists() and not args.rebuild_metadata:
        metadata = pd.read_csv(cache_path)
        print(f"Loaded replay metadata: {cache_path}", flush=True)
        return metadata

    csv_path = repo_path(args.data_dir) / args.dataset / f"{args.dataset}.csv"
    print(f"Building replay metadata from: {csv_path}", flush=True)
    metadata = pd.read_csv(csv_path, usecols=METADATA_COLUMNS)
    metadata.insert(0, "flow_id", metadata.index)

    if args.fraction is not None:
        metadata = metadata.groupby(by="Attack").sample(
            frac=args.fraction,
            random_state=args.seed,
        )

    train_df, val_test_df = train_test_split(
        metadata,
        test_size=0.2,
        random_state=args.seed,
        stratify=metadata["Attack"],
    )
    _, test_df = train_test_split(
        val_test_df,
        test_size=0.5,
        random_state=args.seed,
        stratify=val_test_df["Attack"],
    )

    if "v3" in args.dataset:
        test_df = test_df.sort_values(by="FLOW_START_MILLISECONDS")

    test_df = test_df.reset_index(drop=True)
    test_df.insert(0, "replay_index", np.arange(len(test_df), dtype=np.int64))
    test_df.to_csv(cache_path, index=False)
    print(f"Saved replay metadata: {cache_path}", flush=True)
    return test_df


def make_model(config, dataset, checkpoint_path, device):
    ndim_in = dataset.train_data.feature.read("node", None, "h").shape[1]
    edim_in = dataset.train_data.feature.read("edge", None, "h").shape[1]
    model = GraphIDS(
        ndim_in=ndim_in,
        edim_in=edim_in,
        ndim_hidden=config["ndim_hidden"],
        edim_out=config["edim_out"],
        embed_dim=config["ae_embedding_dim"],
        num_heads=4,
        num_layers=config["num_layers"],
        window_size=config["window_size"],
        dropout=config["dropout"],
        ae_dropout=config["ae_dropout"],
        positional_encoding=config["positional_encoding"],
        nhops=config["nhops"],
        mask_ratio=config["mask_ratio"],
    ).to(device)
    _, threshold = model.load_checkpoint(str(checkpoint_path))
    model.eval()
    return model, float(threshold), edim_in


def write_dashboard_state(path, summary, recent_alerts):
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "recent_alerts": recent_alerts[-100:],
    }
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp_path, path)


def empty_outputs(output_dir):
    predictions_path = output_dir / "predictions.csv"
    alerts_path = output_dir / "alerts.csv"
    alerts_jsonl_path = output_dir / "alerts.jsonl"
    state_path = output_dir / "alerts_state.json"
    metrics_path = output_dir / "alert_metrics.json"
    return predictions_path, alerts_path, alerts_jsonl_path, state_path, metrics_path


def row_payload(row, score, threshold, pred_label, decision_time):
    return {
        "replay_index": int(row.replay_index),
        "flow_id": int(row.flow_id),
        "flow_start_ms": int(row.FLOW_START_MILLISECONDS),
        "flow_end_ms": int(row.FLOW_END_MILLISECONDS),
        "src_ip": str(row.IPV4_SRC_ADDR),
        "dst_ip": str(row.IPV4_DST_ADDR),
        "score": float(score),
        "threshold": float(threshold),
        "pred_label": int(pred_label),
        "true_label": int(row.Label),
        "attack": str(row.Attack),
        "decision_time_utc": decision_time,
    }


def update_delay_metrics(metrics, row, pred_label):
    flow_time = int(row.FLOW_START_MILLISECONDS)
    true_label = int(row.Label)
    if true_label == 1 and metrics["first_true_attack_time_ms"] is None:
        metrics["first_true_attack_time_ms"] = flow_time
        metrics["first_true_attack_replay_index"] = int(row.replay_index)

    if pred_label == 1 and metrics["first_alert_time_ms"] is None:
        metrics["first_alert_time_ms"] = flow_time
        metrics["first_alert_replay_index"] = int(row.replay_index)

    if (
        pred_label == 1
        and true_label == 1
        and metrics["first_true_positive_alert_time_ms"] is None
    ):
        metrics["first_true_positive_alert_time_ms"] = flow_time
        metrics["first_true_positive_alert_replay_index"] = int(row.replay_index)

    conflict = int(pred_label) != true_label
    if conflict and metrics["first_conflict_time_ms"] is None:
        metrics["first_conflict_time_ms"] = flow_time
        metrics["first_conflict_replay_index"] = int(row.replay_index)
        metrics["first_conflict_pred_label"] = int(pred_label)
        metrics["first_conflict_true_label"] = true_label

    if (
        not conflict
        and metrics["first_conflict_time_ms"] is not None
        and metrics["first_consistent_after_conflict_time_ms"] is None
        and int(row.replay_index) > metrics["first_conflict_replay_index"]
    ):
        metrics["first_consistent_after_conflict_time_ms"] = flow_time
        metrics["first_consistent_after_conflict_replay_index"] = int(row.replay_index)


def finalize_metrics(metrics, scores, labels, preds, started_at, ended_at):
    if metrics["first_alert_time_ms"] is not None and metrics["first_true_attack_time_ms"] is not None:
        metrics["first_detection_lag_ms"] = (
            metrics["first_alert_time_ms"] - metrics["first_true_attack_time_ms"]
        )
    if (
        metrics["first_true_positive_alert_time_ms"] is not None
        and metrics["first_true_attack_time_ms"] is not None
    ):
        metrics["first_true_positive_detection_lag_ms"] = (
            metrics["first_true_positive_alert_time_ms"]
            - metrics["first_true_attack_time_ms"]
        )
    if (
        metrics["first_consistent_after_conflict_time_ms"] is not None
        and metrics["first_conflict_time_ms"] is not None
    ):
        metrics["first_conflict_to_consistent_delay_ms"] = (
            metrics["first_consistent_after_conflict_time_ms"]
            - metrics["first_conflict_time_ms"]
        )

    labels_arr = np.asarray(labels)
    preds_arr = np.asarray(preds)
    scores_arr = np.asarray(scores)
    metrics.update(
        {
            "started_at_utc": started_at,
            "ended_at_utc": ended_at,
            "processed_flows": int(len(labels_arr)),
            "alert_count": int(preds_arr.sum()),
            "true_attack_count": int(labels_arr.sum()),
            "false_positive_count": int(((preds_arr == 1) & (labels_arr == 0)).sum()),
            "false_negative_count": int(((preds_arr == 0) & (labels_arr == 1)).sum()),
            "macro_f1": float(f1_score(labels_arr, preds_arr, average="macro", zero_division=0)),
            "pr_auc": float(average_precision_score(labels_arr, scores_arr)),
        }
    )
    return metrics


def run_replay(args):
    config = flatten_config(repo_path(args.config))
    if args.fraction is None:
        args.fraction = config.get("fraction")

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset cache for {args.dataset}, fraction={args.fraction}", flush=True)
    dataset = NetFlowDataset(
        name=args.dataset,
        data_dir=str(repo_path(args.data_dir)),
        force_reload=False,
        fraction=args.fraction,
        data_type=config.get("data_type", "benign"),
        seed=args.seed,
    )
    metadata = load_or_build_test_metadata(args)
    graph_labels = dataset.test_data.feature.read("edge", None, "labels").numpy().reshape(-1)
    if len(metadata) != len(graph_labels):
        raise RuntimeError(
            f"Metadata/test graph length mismatch: metadata={len(metadata)}, graph={len(graph_labels)}"
        )
    if not np.array_equal(metadata["Label"].to_numpy(dtype=np.int32), graph_labels.astype(np.int32)):
        raise RuntimeError("Metadata labels do not align with the cached test graph.")

    checkpoint_path = repo_path(args.checkpoint)
    model, threshold, edim_in = make_model(config, dataset, checkpoint_path, device)
    print(
        f"Replay service ready: checkpoint={checkpoint_path}, features={edim_in}, "
        f"threshold={threshold:.8f}, device={device}",
        flush=True,
    )

    loader = GraphDataLoader(
        dataset.test_data,
        batch_size=args.batch_size,
        nhops=config["nhops"],
        seed=args.seed,
        shuffle=False,
        device=device,
    )

    predictions_path, alerts_path, alerts_jsonl_path, state_path, metrics_path = empty_outputs(output_dir)
    true_attack_rows = metadata[metadata["Label"].astype(np.int32) == 1]
    first_true_attack_time_ms = (
        int(true_attack_rows.iloc[0]["FLOW_START_MILLISECONDS"])
        if not true_attack_rows.empty
        else None
    )
    prediction_fields = [
        "replay_index",
        "flow_id",
        "flow_start_ms",
        "flow_end_ms",
        "src_ip",
        "dst_ip",
        "score",
        "threshold",
        "pred_label",
        "true_label",
        "attack",
        "decision_time_utc",
    ]
    alert_fields = prediction_fields + [
        "alert_type",
        "first_model_attack_time_ms",
        "first_true_attack_time_ms",
        "first_model_attack_to_first_true_attack_delta_ms",
    ]

    metrics = {
        "checkpoint": str(checkpoint_path),
        "dataset": args.dataset,
        "fraction": args.fraction,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "threshold": threshold,
        "first_true_attack_time_ms": None,
        "first_true_attack_replay_index": None,
        "first_alert_time_ms": None,
        "first_alert_replay_index": None,
        "first_detection_lag_ms": None,
        "first_true_positive_alert_time_ms": None,
        "first_true_positive_alert_replay_index": None,
        "first_true_positive_detection_lag_ms": None,
        "first_conflict_time_ms": None,
        "first_conflict_replay_index": None,
        "first_conflict_pred_label": None,
        "first_conflict_true_label": None,
        "first_consistent_after_conflict_time_ms": None,
        "first_consistent_after_conflict_replay_index": None,
        "first_conflict_to_consistent_delay_ms": None,
    }

    scores = []
    labels = []
    preds = []
    recent_alerts = []
    cursor = 0
    printed_alerts = 0
    started_at = datetime.now(timezone.utc).isoformat()
    start_clock = time.perf_counter()

    with open(predictions_path, "w", newline="", encoding="utf-8") as pred_file, open(
        alerts_path, "w", newline="", encoding="utf-8"
    ) as alert_file, open(alerts_jsonl_path, "w", encoding="utf-8") as alert_jsonl:
        pred_writer = csv.DictWriter(pred_file, fieldnames=prediction_fields)
        alert_writer = csv.DictWriter(alert_file, fieldnames=alert_fields)
        pred_writer.writeheader()
        alert_writer.writeheader()

        with torch.inference_mode():
            for batch_no, data in enumerate(loader, start=1):
                block, nfeats, efeats = (
                    data.blocks[0],
                    data.node_features["h"],
                    data.edge_features[0]["h"],
                )
                emb = model.encoder(block, nfeats, efeats, data.compacted_seeds.T)
                sequence_loader = DataLoader(
                    SequentialDataset(
                        emb,
                        window=config["window_size"],
                        step=config["window_size"],
                        device=device,
                    ),
                    batch_size=config["ae_batch_size"],
                    collate_fn=collate_fn,
                )
                batch_scores = []
                for sequence, mask in sequence_loader:
                    outputs = model.transformer(sequence, mask)
                    batch_scores.append(calculate_errors(outputs, sequence, mask).cpu())

                batch_scores = torch.cat(batch_scores).numpy()
                batch_preds = (batch_scores > threshold).astype(np.int32)
                batch_meta = metadata.iloc[cursor : cursor + len(batch_scores)]

                for row, score, pred_label in zip(
                    batch_meta.itertuples(index=False),
                    batch_scores,
                    batch_preds,
                ):
                    decision_time = datetime.now(timezone.utc).isoformat()
                    payload = row_payload(row, score, threshold, pred_label, decision_time)
                    pred_writer.writerow(payload)
                    scores.append(float(score))
                    labels.append(int(row.Label))
                    preds.append(int(pred_label))
                    update_delay_metrics(metrics, row, pred_label)

                    if int(pred_label) == 1:
                        alert_payload = dict(payload)
                        alert_payload["alert_type"] = "attack_flow_detected"
                        alert_payload["first_model_attack_time_ms"] = metrics["first_alert_time_ms"]
                        alert_payload["first_true_attack_time_ms"] = first_true_attack_time_ms
                        if (
                            metrics["first_alert_time_ms"] is not None
                            and first_true_attack_time_ms is not None
                        ):
                            alert_payload[
                                "first_model_attack_to_first_true_attack_delta_ms"
                            ] = first_true_attack_time_ms - metrics["first_alert_time_ms"]
                        else:
                            alert_payload[
                                "first_model_attack_to_first_true_attack_delta_ms"
                            ] = None
                        alert_writer.writerow(alert_payload)
                        alert_jsonl.write(json.dumps(alert_payload) + "\n")
                        alert_jsonl.flush()
                        alert_file.flush()
                        recent_alerts.append(alert_payload)
                        if printed_alerts < args.print_limit:
                            print(
                                "ALERT "
                                f"idx={payload['replay_index']} "
                                f"flow_id={payload['flow_id']} "
                                f"{payload['src_ip']}->{payload['dst_ip']} "
                                f"score={payload['score']:.8f} "
                                f"threshold={threshold:.8f} "
                                f"true={payload['true_label']} "
                                f"attack={payload['attack']}",
                                flush=True,
                            )
                            printed_alerts += 1

                cursor += len(batch_scores)
                pred_file.flush()
                summary = {
                    "processed_flows": cursor,
                    "alert_count": int(sum(preds)),
                    "true_attack_count_seen": int(sum(labels)),
                    "elapsed_seconds": round(time.perf_counter() - start_clock, 3),
                    "first_detection_lag_ms": metrics["first_detection_lag_ms"],
                    "first_true_positive_detection_lag_ms": metrics[
                        "first_true_positive_detection_lag_ms"
                    ],
                    "first_conflict_to_consistent_delay_ms": metrics[
                        "first_conflict_to_consistent_delay_ms"
                    ],
                }
                write_dashboard_state(state_path, summary, recent_alerts)
                if args.max_flows and cursor >= args.max_flows:
                    break
                if args.sleep_ms:
                    time.sleep(args.sleep_ms / 1000.0)

    ended_at = datetime.now(timezone.utc).isoformat()
    metrics = finalize_metrics(metrics, scores, labels, preds, started_at, ended_at)
    json_metrics = {
        key: value
        for key, value in metrics.items()
        if key != "first_true_positive_detection_lag_ms"
    }
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(json_metrics, handle, indent=2)
    write_dashboard_state(state_path, metrics, recent_alerts)

    print(f"Replay finished. Predictions: {predictions_path}", flush=True)
    print(f"Alerts CSV: {alerts_path}", flush=True)
    print(f"Alerts JSONL: {alerts_jsonl_path}", flush=True)
    print(f"Metrics: {metrics_path}", flush=True)
    print(
        f"Processed={metrics['processed_flows']} Alerts={metrics['alert_count']} "
        f"F1={metrics['macro_f1']:.4f} PR-AUC={metrics['pr_auc']:.4f}",
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Replay NetFlow rows as a near-real-time alert service.")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--dataset", default="NF-CSE-CIC-IDS2018-v3")
    parser.add_argument("--config", default="configs/NF-CSE-CIC-IDS2018-v3.yaml")
    parser.add_argument(
        "--checkpoint",
        default="pretrained/GraphIDS_NF-CSE-CIC-IDS2018-v3.ckpt",
    )
    parser.add_argument("--output_dir", default="real-time model/outputs")
    parser.add_argument("--fraction", type=float, default=None)
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--max_flows", type=int, default=None)
    parser.add_argument("--sleep_ms", type=int, default=0)
    parser.add_argument("--print_limit", type=int, default=20)
    parser.add_argument("--rebuild_metadata", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_replay(parse_args())
