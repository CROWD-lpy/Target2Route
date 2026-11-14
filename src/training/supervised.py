import json
from itertools import islice

import torch
from torch.utils.data import DataLoader

from ..chemistry.building_blocks import BuildingBlockIndex
from ..data import RouteDataset, collate_routes, load_embedding_bank
from ..model import Target2RouteGen
from ..runtime import (
    create_run,
    load_pickle,
    model_from_checkpoint,
    model_from_state,
    read_checkpoint,
    save_torch,
    setup_logger,
)

LOSS_METRICS = (
    "total_loss",
    "action_loss",
    "template_loss",
    "fp_loss",
    "action_acc",
    "template_acc",
)
TOP_K = (1, 3, 5, 10, 20, 50)


def make_loader(config, split, embeddings, *, training=False):
    path = getattr(config, f"{split}_dataset_path")
    if not path:
        raise ValueError(f"Set {split}_dataset_path to evaluate the {split} split")
    dataset = RouteDataset(load_pickle(path), embeddings, config)
    if len(dataset) == 0 or (training and len(dataset) < 2):
        raise ValueError(f"Not enough samples in {path}")
    # The start-molecule BatchNorm layer needs at least two training samples.
    drop_last = training and len(dataset) % config.batch_size == 1
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=training,
        collate_fn=collate_routes,
        num_workers=config.num_workers,
        drop_last=drop_last,
    )


def _accumulate_losses(metrics, loss, details):
    metrics["total_loss"] += loss.item()
    for key in LOSS_METRICS[1:]:
        metrics[key] += details[key].item()


def train_epoch(model, loader, optimizer, config, logger):
    model.train()
    metrics = dict.fromkeys(LOSS_METRICS, 0.0)
    for count, batch in enumerate(islice(loader, config.max_batches), start=1):
        optimizer.zero_grad()
        loss, details = model.compute_loss(batch, mode="train")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        _accumulate_losses(metrics, loss, details)
        if count % config.log_interval == 0:
            logger.info("Batch %d/%d | loss %.4f", count, len(loader), loss.item())
    return {key: value / count for key, value in metrics.items()}


def _retrieval_hits(building_blocks, predictions, targets, first_try, all_tries):
    if predictions.ndim == 2:
        predictions = predictions.unsqueeze(1)
    n_targets, tries, dim = predictions.shape
    true_hits = building_blocks.search_continuous(targets, top_k=1, nprobe=64)
    candidates = building_blocks.search_continuous(
        predictions.reshape(n_targets * tries, dim), top_k=max(TOP_K), nprobe=64
    )
    for i, hit in enumerate(true_hits):
        if not hit:
            continue
        actual = hit[0][0]
        ranks = [
            next(
                (
                    rank
                    for rank, (smiles, _) in enumerate(row, start=1)
                    if smiles == actual
                ),
                max(TOP_K) + 1,
            )
            for row in candidates[i * tries : (i + 1) * tries]
        ]
        best = min(ranks)
        for k in TOP_K:
            first_try[k] += ranks[0] <= k
            all_tries[k] += best <= k


@torch.no_grad()
def evaluate_epoch(model, loader, building_blocks, config):
    model.eval()
    metrics = dict.fromkeys(LOSS_METRICS, 0.0)
    first_try = dict.fromkeys(TOP_K, 0)
    all_tries = dict.fromkeys(TOP_K, 0)
    n_queries = 0
    for count, batch in enumerate(islice(loader, config.max_batches), start=1):
        loss, details = model.compute_loss(batch, mode="test")
        _accumulate_losses(metrics, loss, details)
        targets = details["fp_targets"]
        if targets is not None:
            _retrieval_hits(
                building_blocks, details["fp_preds"], targets, first_try, all_tries
            )
            n_queries += len(targets)

    result = {key: value / count for key, value in metrics.items()}
    result.update({f"fp_top{k}_only": first_try[k] / max(1, n_queries) for k in TOP_K})
    result.update({f"fp_top{k}": all_tries[k] / max(1, n_queries) for k in TOP_K})
    result.update(num_search=n_queries, batches=count)
    return result


def _report(logger, name, results):
    logger.info("%s %s", name, json.dumps(results, sort_keys=True))


def train_supervised(config, output_dir=None, *, resume=None):
    device = torch.device(config.device)
    completed_epoch = 0
    rng_state = cuda_rng_state = None
    if resume:
        checkpoint, state = read_checkpoint(resume)
        completed_epoch = checkpoint.get("epoch")
        optimizer_state = checkpoint.get("optimizer_state_dict")
        if not isinstance(completed_epoch, int) or completed_epoch < 1 or optimizer_state is None:
            raise ValueError(
                f"{resume} must contain a completed supervised epoch and its optimizer state"
            )
        if config.num_epoch <= completed_epoch:
            raise ValueError(
                f"--epochs is the total epoch count and must exceed saved epoch {completed_epoch}"
            )
        model = model_from_state(config, state, resume)
        rng_state = checkpoint.get("torch_rng_state")
        cuda_rng_state = checkpoint.get("cuda_rng_state")
        del checkpoint, state
    else:
        model = Target2RouteGen(config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    if resume:
        optimizer.load_state_dict(optimizer_state)
        del optimizer_state
        config.lr = optimizer.param_groups[0]["lr"]
        config.weight_decay = optimizer.param_groups[0]["weight_decay"]

    config.resume = str(resume) if resume else None
    run_dir = create_run(config, "supervised", output_dir)
    logger = setup_logger(run_dir, "train.log")
    if resume:
        logger.info(
            "Resuming %s at epoch %d/%d | AdamW lr %.8g",
            resume,
            completed_epoch + 1,
            config.num_epoch,
            config.lr,
        )
        if rng_state is None:
            logger.info("This checkpoint has no saved PyTorch RNG state")

    embeddings = load_embedding_bank(config.target_embedding_path, required=True)
    training = make_loader(config, "train", embeddings, training=True)
    validation = make_loader(config, "valid", embeddings)
    building_blocks = BuildingBlockIndex(config)
    checkpoint_dir = run_dir / "ckpt"
    checkpoint_dir.mkdir()

    if rng_state is not None:
        torch.set_rng_state(rng_state)
    if device.type == "cuda" and cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state, device)

    checkpoints = []
    for epoch in range(completed_epoch + 1, config.num_epoch + 1):
        _report(
            logger,
            f"Train epoch {epoch}",
            train_epoch(model, training, optimizer, config, logger),
        )
        metrics = evaluate_epoch(model, validation, building_blocks, config)
        _report(logger, f"Validation epoch {epoch}", metrics)
        accuracy = metrics["fp_top1_only"]
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": accuracy,
            "torch_rng_state": torch.get_rng_state(),
        }
        if device.type == "cuda":
            checkpoint["cuda_rng_state"] = torch.cuda.get_rng_state(device)
        save_torch(checkpoint, checkpoint_dir / "last.pt")

        if len(checkpoints) == config.save_top_k and accuracy <= checkpoints[-1][0]:
            continue
        checkpoint_path = checkpoint_dir / f"model_epoch_{epoch:03d}_{accuracy:.4f}.pt"
        save_torch(checkpoint, checkpoint_path)
        logger.info(
            "Saved %s (validation first-try Top-1 %.4f)", checkpoint_path, accuracy
        )
        checkpoints.append((accuracy, checkpoint_path))
        checkpoints.sort(key=lambda entry: entry[0], reverse=True)
        if len(checkpoints) > config.save_top_k:
            checkpoints.pop()[1].unlink()

    if config.test_dataset_path:
        del training, validation, optimizer, checkpoint
        test = make_loader(config, "test", embeddings)
        best = checkpoints[0][1]
        _, state = read_checkpoint(best)
        model.load_state_dict(state, strict=True)
        logger.info("Testing %s", best)
        _report(logger, "Test", evaluate_epoch(model, test, building_blocks, config))


def evaluate_supervised(config, checkpoint, split):
    embeddings = load_embedding_bank(config.target_embedding_path, required=True)
    loader = make_loader(config, split, embeddings)
    building_blocks = BuildingBlockIndex(config)
    model, _ = model_from_checkpoint(config, checkpoint)
    return evaluate_epoch(model, loader, building_blocks, config)
