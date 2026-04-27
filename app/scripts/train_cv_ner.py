!pip install evaluate
!pip install seqeval
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)
import evaluate

ROOT = Path.cwd().resolve().parents[1]
LOCAL_DATA = ROOT / "data" / "training" / "bio_dataset2.jsonl"
LOCAL_OUTPUT = ROOT / "data" / "models" / "cv_ner"

KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")
IS_KAGGLE = KAGGLE_INPUT.exists()

if IS_KAGGLE:
    DEFAULT_DATA = Path("/kaggle/input/datasets/marzeba/bio-dataset2/bio_dataset.jsonl")
    DEFAULT_OUTPUT = KAGGLE_WORKING / "cv_ner"
else:
    DEFAULT_DATA = LOCAL_DATA
    DEFAULT_OUTPUT = LOCAL_OUTPUT

SEED = 42


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA,
                   help="Path to BIO-tagged JSONL")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                   help="Output directory for model artifacts")
    p.add_argument("--model", type=str, default="xlm-roberta-base",
                   help="Pretrained backbone")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--mini", action="store_true",
                   help="Use only 50 records (sanity check)")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate config and exit without training")
   
    args, unknown = p.parse_known_args()
    return args


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_bio_dataset(path: Path) -> Dataset:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "tokens" in rec and "tags" in rec and len(rec["tokens"]) == len(rec["tags"]):
                records.append(rec)
    if not records:
        raise RuntimeError(f"No valid records in {path}")
    return Dataset.from_list(records)


def derive_label_list(ds: Dataset) -> list[str]:
    """Collect unique BIO labels from the dataset, with O first."""
    labels = set()
    for rec in ds:
        labels.update(rec["tags"])
    out = ["O"]
    out.extend(sorted(l for l in labels if l.startswith("B-")))
    out.extend(sorted(l for l in labels if l.startswith("I-")))
    return out


def build_align_fn(tokenizer, label2id: dict, max_length: int):
    """Tokenize words → subwords and align BIO labels."""
    def fn(examples):
        enc = tokenizer(
            examples["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        aligned_batch = []
        for batch_idx, raw_tags in enumerate(examples["tags"]):
            word_ids = enc.word_ids(batch_index=batch_idx)
            aligned = []
            prev_w = None
            for w_id in word_ids:
                if w_id is None:
                    aligned.append(-100)
                else:
                    tag = raw_tags[w_id]
                    if w_id == prev_w and tag.startswith("B-"):
                        # subword continuation of an entity → convert B- to I-
                        tag = "I-" + tag[2:]
                    aligned.append(label2id[tag])
                prev_w = w_id
            aligned_batch.append(aligned)
        enc["labels"] = aligned_batch
        return enc
    return fn


def make_compute_metrics(id2label: dict):
    metric = evaluate.load("seqeval")

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = np.argmax(preds, axis=2)

        true_p, true_l = [], []
        for pred_seq, lab_seq in zip(preds, labels):
            tp, tl = [], []
            for p_id, l_id in zip(pred_seq, lab_seq):
                if l_id == -100:
                    continue
                tp.append(id2label[int(p_id)])
                tl.append(id2label[int(l_id)])
            true_p.append(tp)
            true_l.append(tl)

        results = metric.compute(predictions=true_p, references=true_l)
        out = {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
        # Per-entity breakdown
        for key, val in results.items():
            if isinstance(val, dict) and "f1" in val:
                out[f"{key}_f1"] = val["f1"]
                out[f"{key}_precision"] = val["precision"]
                out[f"{key}_recall"] = val["recall"]
        return out
    return compute_metrics


def main():
    args = parse_args()
    set_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Environment: {'Kaggle' if IS_KAGGLE else 'local'}")
    print(f"Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"Data:   {args.data}")
    print(f"Output: {args.output}")
    print(f"Model:  {args.model}")
    print()

    if not args.data.exists():
        print(f"ERROR: data file not found: {args.data}", file=sys.stderr)
        sys.exit(1)

    args.output.mkdir(parents=True, exist_ok=True)

    print("Loading BIO dataset...")
    ds = load_bio_dataset(args.data)
    print(f"  total records: {len(ds)}")

    if args.mini:
        ds = ds.select(range(min(50, len(ds))))
        print(f"  --mini: limited to {len(ds)} records")

    label_list = derive_label_list(ds)
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for i, l in enumerate(label_list)}
    print(f"Labels ({len(label_list)}): {label_list}")

    splits = ds.train_test_split(test_size=0.2, seed=SEED)
    val_test = splits["test"].train_test_split(test_size=0.5, seed=SEED)
    train_ds = splits["train"]
    val_ds = val_test["train"]
    test_ds = val_test["test"]
    print(f"Splits: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    print(f"\nLoading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, add_prefix_space=True)
    align_fn = build_align_fn(tokenizer, label2id, args.max_length)

    print("Tokenizing + aligning labels...")
    train_ds = train_ds.map(align_fn, batched=True, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(align_fn, batched=True, remove_columns=val_ds.column_names)
    test_ds_proc = test_ds.map(align_fn, batched=True, remove_columns=test_ds.column_names)

    if args.dry_run:
        print("\n--dry-run: dataset valid, config OK. Exiting.")
        return

    print(f"\nLoading model: {args.model}")
    model = AutoModelForTokenClassification.from_pretrained(
        args.model,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir=str(args.output / "checkpoints"),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        seed=SEED,
        fp16=device == "cuda",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=make_compute_metrics(id2label),
    )

    print("\nStarting training...")
    trainer.train()

    print("\nEvaluating on held-out test set...")
    test_metrics = trainer.evaluate(test_ds_proc, metric_key_prefix="test")

    final_dir = args.output / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    metrics_payload = {
        "model": args.model,
        "label_list": label_list,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        "test_metrics": test_metrics,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n{'=' * 60}")
    print("TRAINING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Test set:  precision={test_metrics['test_precision']:.4f}  "
          f"recall={test_metrics['test_recall']:.4f}  "
          f"f1={test_metrics['test_f1']:.4f}")
    print(f"Model:     {final_dir}")
    print(f"Metrics:   {args.output / 'metrics.json'}")


if __name__ == "__main__":
    main()
