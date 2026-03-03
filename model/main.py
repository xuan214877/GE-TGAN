import dgl
import torch
import json
import torch.nn as nn
import random 
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, roc_curve, precision_score, recall_score
import os
import numpy as np
from config import *
from data_loader import load_pruned_data
from model import TGATModel, LinkPredictor
from negative_sampling import negative_sampling, get_dynamic_neg_ratio_by_graph_size
from embedding_generator import generate_embeddings


SEEDS = [42, 1024, 2025, 3, 7, 11, 19, 23, 31, 41]
all_seed_results = []


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


device = torch.device("cuda" if USE_CUDA and not DEBUG_MODE else "cpu")
if USE_CUDA:
    print(f"CUDA version: {torch.version.cuda}")
    torch.cuda.empty_cache()

TRAIN_WINDOWS = [(2015, 2018), (2019, 2019), (2020, 2020), (2021, 2021), (2022, 2022)]
VAL_WINDOW = [(2023, 2023)]
TEST_WINDOWS = [(2024, 2025)]

VAL_BEST_THRESHOLD = None
VAL_THRESHOLD_STATS = {} 
FREQUENCY_FIELD = 'frequency_per_year'

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "epoch_embeddings1"), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "epoch_embeddings_final"), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "seed_results"), exist_ok=True)
for i in range(1, NUM_EPOCHS // 5 + 1):
    os.makedirs(os.path.join(SAVE_DIR, f"checkpoint_{i}"), exist_ok=True)

g, edge_metadata, metadata = load_pruned_data(PRUNED_DIR, EMBEDDING_PATH)
num_nodes = g.num_nodes()
print(f"nodes: {num_nodes}, edges: {g.num_edges()}")


if len(metadata['pruned_nodes']) > 0:
    sample_node = metadata['pruned_nodes'][0]


def get_edges_by_time_window(g, windows):
    edge_mask = torch.zeros(g.num_edges(), dtype=torch.bool, device=device)
    total_edges = 0
    for start_year, end_year in windows:
        window_mask = (g.edata['edge_year'] >= start_year) & (g.edata['edge_year'] <= end_year)
        window_edge_count = torch.sum(window_mask).item()
        edge_mask |= window_mask
        total_edges += window_edge_count
    selected_edges = torch.where(edge_mask)[0]
    return selected_edges


train_edges_global = get_edges_by_time_window(g, TRAIN_WINDOWS)
val_edges_global = get_edges_by_time_window(g, VAL_WINDOW)
test_edges_global = get_edges_by_time_window(g, TEST_WINDOWS)


def build_set_graph(original_g, edge_indices, set_name):
    set_g = dgl.edge_subgraph(original_g, edge_indices).to(device)
    set_g.edata['original_year'] = set_g.edata['edge_year'].clone()
    set_g = dgl.add_reverse_edges(set_g, copy_edata=True)
    set_g.edata['is_reverse'] = torch.cat([
        torch.zeros(len(edge_indices), device=device),
        torch.ones(len(edge_indices), device=device)
    ])
    return set_g


train_g = build_set_graph(g, train_edges_global, "training")
val_g = build_set_graph(g, val_edges_global, "validation")
test_g = build_set_graph(g, test_edges_global, "testing")


def build_subgraphs_with_current_edges(g, windows, metadata, device, set_name):
    sub_graphs = []
    current_pos_edges = []
    time_diffs = []
    node_feats_list = []

    for start_year, end_year in windows:
        if 'is_reverse' in g.edata:
            original_edge_mask = (g.edata['is_reverse'] == 0)
        else:
            original_edge_mask = torch.ones(g.num_edges(), dtype=torch.bool, device=device)

        history_mask = original_edge_mask & (g.edata['edge_year'] < start_year)
        history_edge_idx = torch.where(history_mask)[0]
        print(f"\n{set_name}time window {start_year}-{end_year}:")

        if len(history_edge_idx) < 2:
            print(f"skip small window")
            continue

        history_sub_g = dgl.edge_subgraph(g, history_edge_idx)
        history_sub_g = dgl.add_reverse_edges(history_sub_g, copy_edata=True)
        history_years = history_sub_g.edata['edge_year']


        current_pos_mask = original_edge_mask & (g.edata['edge_year'] >= start_year) & (
                    g.edata['edge_year'] <= end_year)
        current_pos_edge_idx = torch.where(current_pos_mask)[0]
        current_pos_src, current_pos_dst = g.find_edges(current_pos_edge_idx)

        sub_node_ids = history_sub_g.ndata[dgl.NID]
        pos_src_mask = torch.isin(current_pos_src, sub_node_ids)
        pos_dst_mask = torch.isin(current_pos_dst, sub_node_ids)
        valid_pos_mask = pos_src_mask & pos_dst_mask
        valid_current_pos_src = current_pos_src[valid_pos_mask]
        valid_current_pos_dst = current_pos_dst[valid_pos_mask]

        if len(valid_current_pos_src) == 0:
            print(f"escape time window")
            continue

        node_degrees = (history_sub_g.out_degrees() + history_sub_g.in_degrees()).unsqueeze(1).float()

        window_frequencies = []
        sub_node_orig_ids = history_sub_g.ndata[dgl.NID].cpu().numpy()
        freq_zero_count = 0
        for nid in sub_node_orig_ids:
            total_freq = 0
            if nid >= len(metadata['pruned_nodes']):
                window_frequencies.append(total_freq)
                freq_zero_count += 1
                continue

            node = metadata['pruned_nodes'][nid]
            if FREQUENCY_FIELD in node:
                freq_dict = node[FREQUENCY_FIELD]
                for year_str, count in freq_dict.items():
                    year_int = int(year_str)
                    if year_int < start_year:
                        total_freq += count
            else:
                node_idx_in_subg = torch.where(history_sub_g.ndata[dgl.NID] == nid)[0].item()
                total_freq = history_sub_g.out_degrees(node_idx_in_subg) + history_sub_g.in_degrees(node_idx_in_subg)

            window_frequencies.append(total_freq)
            if total_freq == 0:
                freq_zero_count += 1

        degree_mean = torch.mean(node_degrees)
        degree_std = torch.std(node_degrees) if torch.std(node_degrees) > 1e-6 else 1.0
        node_degrees = (node_degrees - degree_mean) / degree_std

        freq_mean = torch.mean(node_frequencies)
        freq_std = torch.std(node_frequencies) if torch.std(node_frequencies) > 1e-6 else 1.0
        node_frequencies = (node_frequencies - freq_mean) / freq_std

        node_feats = torch.cat([history_sub_g.ndata['feat'], node_degrees, node_frequencies], dim=1)


        time_diff = end_year - history_sub_g.edata['edge_year'].float()
        time_diff = torch.clamp(time_diff, min=0)


        src_to_sub_id = {nid.item(): idx for idx, nid in enumerate(history_sub_g.ndata[dgl.NID])}
        dst_to_sub_id = {nid.item(): idx for idx, nid in enumerate(history_sub_g.ndata[dgl.NID])}

        def map_to_sub_id(ids, id_map):
            mapped = []
            for nid in ids:
                if nid.item() in id_map:
                    mapped.append(id_map[nid.item()])
            return torch.tensor(mapped, device=device)

        mapped_pos_src = map_to_sub_id(valid_current_pos_src, src_to_sub_id)
        mapped_pos_dst = map_to_sub_id(valid_current_pos_dst, dst_to_sub_id)

        sub_graphs.append(history_sub_g)
        current_pos_edges.append((mapped_pos_src, mapped_pos_dst))
        time_diffs.append(time_diff)
        node_feats_list.append(node_feats)

    return sub_graphs, current_pos_edges, time_diffs, node_feats_list


for seed_idx, seed in enumerate(SEEDS):
    VAL_BEST_THRESHOLD = None
    VAL_THRESHOLD_STATS = {}
    set_random_seed(seed)

    train_sub_graphs, train_current_pos_edges, train_time_diffs, train_node_feats = build_subgraphs_with_current_edges(
        train_g, TRAIN_WINDOWS, metadata, device, "training"
    )
    val_sub_graphs, val_current_pos_edges, val_time_diffs, val_node_feats = build_subgraphs_with_current_edges(
        g, VAL_WINDOW, metadata, device, "validation"
    )
    test_sub_graphs, test_current_pos_edges, test_time_diffs, test_node_feats = build_subgraphs_with_current_edges(
        g, TEST_WINDOWS, metadata, device, "testing"
    )

    best_val_auc = 0.0
    best_params = {}
    best_model_path = os.path.join(SAVE_DIR, f"best_tgat_model_seed_{seed}.pth")

    for out_dim in OUT_DIM_LIST:
        for lr in LR_LIST:
            for num_heads in NUM_HEADS_LIST:
                print(f"\nseed: {seed}: OUT_DIM={out_dim}, LR={lr}, NUM_HEADS={num_heads}")
                tgat_model = TGATModel(
                    in_dim=g.ndata['feat'].shape[1] + 2,
                    hidden_dim=2 * out_dim,
                    out_dim=out_dim,
                    num_heads=num_heads,
                    dropout=0.3
                ).to(device)
                predictor = LinkPredictor().to(device)
                criterion = nn.BCEWithLogitsLoss().to(device)
                optimizer = torch.optim.Adam(
                    list(tgat_model.parameters()) + list(predictor.parameters()),
                    lr=lr
                )

                current_best_val_auc = 0.0
                current_best_model_path = os.path.join(SAVE_DIR,
                                                       f"best_tgat_model_{out_dim}_{lr}_{num_heads}_seed_{seed}.pth")
                dropout_schedule = torch.linspace(0.1, 0.3, NUM_EPOCHS)

                for epoch in tqdm(range(NUM_EPOCHS)):
                    if hasattr(tgat_model.layer1, 'dropout'):
                        tgat_model.layer1.dropout = dropout_schedule[epoch].item()
                        tgat_model.layer2.dropout = dropout_schedule[epoch].item()

                    tgat_model.train()
                    predictor.train()
                    epoch_loss = 0.0
                    batch_count = 0

                    for sub_g, current_pos_edges, time_diff, node_feats in zip(
                            train_sub_graphs, train_current_pos_edges, train_time_diffs, train_node_feats
                    ):
                        try:
                            node_emb = tgat_model(sub_g, node_feats, time_diff)
                        except Exception as e:
                            device_cpu = torch.device("cpu")
                            tgat_model.to(device_cpu)
                            sub_g = sub_g.to(device_cpu)
                            node_feats = node_feats.to(device_cpu)
                            time_diff = time_diff.to(device_cpu)
                            node_emb = tgat_model(sub_g, node_feats, time_diff)
                            tgat_model.to(device)
                            sub_g = sub_g.to(device)
                            node_feats = node_feats.to(device)
                            time_diff = time_diff.to(device)
                            node_emb = node_emb.to(device)

                        try:
                            valid_pos_src, valid_pos_dst = current_pos_edges
                            if len(valid_pos_src) == 0:
                                continue

                            dynamic_neg_ratio = get_dynamic_neg_ratio_by_graph_size(sub_g)
                            valid_neg_src, valid_neg_dst = negative_sampling(
                                sub_g,
                                int(len(valid_pos_src) * NEG_SAMPLE_RATIO),
                                {'source_idx': valid_pos_src.tolist(), 'target_idx': valid_pos_dst.tolist()},
                                sub_g,
                                dynamic_ratio=dynamic_neg_ratio
                            )

                            max_idx = node_emb.size(0) - 1
                            valid_neg_src = valid_neg_src[valid_neg_src <= max_idx]
                            valid_neg_dst = valid_neg_dst[valid_neg_dst <= max_idx]
                            if len(valid_neg_src) == 0:
                                continue

                            pos_delta_t = torch.full((len(valid_pos_src),), time_diff.max().item(), device=device)
                            neg_delta_t = torch.full((len(valid_neg_src),), time_diff.max().item(), device=device)

                            pos_scores = torch.clamp(
                                predictor(node_emb[valid_pos_src], node_emb[valid_pos_dst]),
                                -100, 100
                            )
                            neg_scores = torch.clamp(
                                predictor(node_emb[valid_neg_src], node_emb[valid_neg_dst]),
                                -100, 100
                            )

                            loss = criterion(
                                torch.cat([pos_scores, neg_scores]),
                                torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
                            )
                            optimizer.zero_grad()
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(tgat_model.parameters(), 1.0)
                            optimizer.step()

                            epoch_loss += loss.item()
                            batch_count += 1
                        except Exception as e:
                            print(f"seed: {seed}: {str(e)}")

                    if batch_count > 0:
                        avg_loss = epoch_loss / batch_count
                        tqdm.write(f"seed {seed} Epoch {epoch + 1}/{NUM_EPOCHS}, avg_loss: {avg_loss:.4f}")

                    if (epoch + 1) % 5 == 0:
                        checkpoint_idx = (epoch + 1) // 5
                        tgat_model.eval()
                        predictor.eval()
                        val_metrics = {'auc': 0, 'f1': 0, 'ap': 0}
                        val_count = 0
                        val_all_scores = []
                        val_all_labels = []

                        for sub_g, current_pos_edges, time_diff, node_feats in zip(
                                val_sub_graphs, val_current_pos_edges, val_time_diffs, val_node_feats
                        ):
                            with torch.no_grad():
                                try:
                                    node_emb = tgat_model(sub_g, node_feats, time_diff)
                                except:
                                    continue

                                try:
                                    valid_pos_src, valid_pos_dst = current_pos_edges
                                    if len(valid_pos_src) == 0:
                                        continue

                                    valid_neg_src, valid_neg_dst = negative_sampling(
                                        sub_g,
                                        int(len(valid_pos_src) * NEG_SAMPLE_RATIO),
                                        {'source_idx': valid_pos_src.tolist(), 'target_idx': valid_pos_dst.tolist()},
                                        sub_g,
                                        dynamic_ratio=dynamic_neg_ratio
                                    )
                                    max_idx = node_emb.size(0) - 1
                                    valid_neg_src = valid_neg_src[valid_neg_src <= max_idx]
                                    valid_neg_dst = valid_neg_dst[valid_neg_dst <= max_idx]
                                    if len(valid_neg_src) == 0:
                                        continue

                                    pos_delta_t = torch.full((len(valid_pos_src),), time_diff.max().item(),
                                                             device=device)
                                    neg_delta_t = torch.full((len(valid_neg_src),), time_diff.max().item(),
                                                             device=device)

                                    pos_scores = predictor(node_emb[valid_pos_src], node_emb[valid_pos_dst])
                                    neg_scores = predictor(node_emb[valid_neg_src], node_emb[valid_neg_dst])

                                    batch_scores = torch.cat([pos_scores, neg_scores]).cpu().numpy()
                                    batch_labels = torch.cat([
                                        torch.ones_like(pos_scores),
                                        torch.zeros_like(neg_scores)
                                    ]).cpu().numpy()
                                    val_all_scores.extend(batch_scores)
                                    val_all_labels.extend(batch_labels)

                                    window_auc = roc_auc_score(batch_labels, batch_scores) if len(
                                        set(batch_labels)) > 1 else 0.5
                                    window_ap = average_precision_score(batch_labels, batch_scores)

                                    val_metrics['auc'] += window_auc
                                    val_metrics['ap'] += window_ap
                                    val_count += 1
                                except Exception as e:
                                    print(f"seed: {seed} {str(e)}")

                        if val_count > 0:
                            val_auc = val_metrics['auc'] / val_count
                            val_ap = val_metrics['ap'] / val_count

                            if len(set(val_all_labels)) > 1 and len(val_all_scores) > 0:
                                fpr, tpr, thresholds = roc_curve(val_all_labels, val_all_scores)
                                j_scores = tpr - fpr
                                best_idx = np.argmax(j_scores)
                                VAL_BEST_THRESHOLD = float(thresholds[best_idx])
                                VAL_THRESHOLD_STATS['best_threshold'] = VAL_BEST_THRESHOLD
                                VAL_THRESHOLD_STATS['j_score'] = float(j_scores[best_idx])
                                y_pred = (np.array(val_all_scores) > VAL_BEST_THRESHOLD).astype(int)
                                val_f1 = float(f1_score(val_all_labels, y_pred))
                                val_precision = float(precision_score(val_all_labels, y_pred))
                                val_recall = float(recall_score(val_all_labels, y_pred))
                            else:
                                val_f1 = 0.0
                                val_precision = 0.0
                                val_recall = 0.0

                            checkpoint_model_path = os.path.join(
                                SAVE_DIR, f"checkpoint_{checkpoint_idx}",
                                f"tgat_model_epoch_{epoch + 1}_outdim_{out_dim}_lr_{lr}_heads_{num_heads}_seed_{seed}.pth"
                            )
                            try:
                                torch.save({
                                    'tgat_state_dict': tgat_model.state_dict(),
                                    'predictor_state_dict': predictor.state_dict(),
                                    'val_auc': float(val_auc),
                                    'val_f1': float(val_f1),
                                    'val_ap': float(val_ap),
                                    'val_threshold': VAL_BEST_THRESHOLD,
                                    'epoch': epoch + 1,
                                    'seed': seed
                                }, checkpoint_model_path)
                                tqdm.write(
                                    f"seed: {seed} Epoch {epoch + 1}: {val_auc:.4f}, F1: {val_f1:.4f}, AP: {val_ap:.4f}"
                                )
                            except Exception as e:
                                print(f"seed {seed} save fail: {str(e)}")

                            if val_auc > current_best_val_auc:
                                current_best_val_auc = val_auc
                                current_best_val_f1 = val_f1
                                current_best_val_ap = val_ap
                                try:
                                    torch.save({
                                        'tgat_state_dict': tgat_model.state_dict(),
                                        'predictor_state_dict': predictor.state_dict(),
                                        'best_val_auc': float(current_best_val_auc),
                                        'best_val_f1': float(current_best_val_f1),
                                        'best_val_ap': float(current_best_val_ap),
                                        'val_best_threshold': VAL_BEST_THRESHOLD,
                                        'epoch': epoch + 1,
                                        'seed': seed
                                    }, current_best_model_path)
                                    tqdm.write(f"seed: {seed}, update model")
                                except Exception as e:
                                    print(f"seed: {seed}: model save failed {str(e)}")

                if current_best_val_auc > best_val_auc:
                    best_val_auc = current_best_val_auc
                    best_params = {'OUT_DIM': out_dim, 'LR': lr, 'NUM_HEADS': num_heads}
                    best_model_path = current_best_model_path

    seed_test_results = {
        'seed': seed,
        'best_params': best_params,
        'best_val_auc': float(best_val_auc),
        'val_best_threshold': VAL_BEST_THRESHOLD,
        'test_metrics': {}
    }
    if len(test_sub_graphs) > 0 and VAL_BEST_THRESHOLD is not None:
        print(f"\nseed: {seed} testing")
        checkpoint = torch.load(best_model_path, map_location=device)
        VAL_BEST_THRESHOLD = checkpoint.get('val_best_threshold', VAL_BEST_THRESHOLD)
        best_out_dim = best_params['OUT_DIM']

        tgat_model = TGATModel(
            in_dim=g.ndata['feat'].shape[1] + 2,
            hidden_dim=2 * best_out_dim,
            out_dim=best_out_dim,
            num_heads=best_params['NUM_HEADS']
        ).to(device)
        tgat_model.load_state_dict(checkpoint['tgat_state_dict'], strict=False)
        predictor = LinkPredictor().to(device)
        predictor.load_state_dict(checkpoint['predictor_state_dict'])

        tgat_model.eval()
        predictor.eval()
        test_metrics = {'auc': 0, 'f1': 0, 'ap': 0, 'precision': 0, 'recall': 0}
        test_count = 0
        test_all_scores = []
        test_all_labels = []

        with torch.no_grad():
            for sub_g, current_pos_edges, time_diff, node_feats in zip(
                    test_sub_graphs, test_current_pos_edges, test_time_diffs, test_node_feats
            ):
                try:
                    node_emb = tgat_model(sub_g, node_feats, time_diff)
                    valid_pos_src, valid_pos_dst = current_pos_edges
                    if len(valid_pos_src) == 0:
                        continue

                    valid_neg_src, valid_neg_dst = negative_sampling(
                        sub_g,
                        int(len(valid_pos_src) * NEG_SAMPLE_RATIO),
                        {'source_idx': valid_pos_src.tolist(), 'target_idx': valid_pos_dst.tolist()},
                        sub_g
                    )
                    max_idx = node_emb.size(0) - 1
                    valid_neg_src = valid_neg_src[valid_neg_src <= max_idx]
                    valid_neg_dst = valid_neg_dst[valid_neg_dst <= max_idx]
                    if len(valid_neg_src) == 0:
                        continue

                    pos_delta_t = torch.full((len(valid_pos_src),), time_diff.max().item(), device=device)
                    neg_delta_t = torch.full((len(valid_neg_src),), time_diff.max().item(), device=device)

                    pos_scores = predictor(node_emb[valid_pos_src], node_emb[valid_pos_dst])
                    neg_scores = predictor(node_emb[valid_neg_src], node_emb[valid_neg_dst])

                    batch_scores = torch.cat([pos_scores, neg_scores]).cpu().numpy()
                    batch_labels = torch.cat([
                        torch.ones_like(pos_scores),
                        torch.zeros_like(neg_scores)
                    ]).cpu().numpy()
                    test_all_scores.extend(batch_scores)
                    test_all_labels.extend(batch_labels)

                    window_auc = roc_auc_score(batch_labels, batch_scores) if len(set(batch_labels)) > 1 else 0.5
                    window_ap = average_precision_score(batch_labels, batch_scores)

                    y_pred = (batch_scores > VAL_BEST_THRESHOLD).astype(int)
                    window_precision = float(precision_score(batch_labels, y_pred)) if len(
                        set(batch_labels)) > 1 else 0.0
                    window_recall = float(recall_score(batch_labels, y_pred)) if len(set(batch_labels)) > 1 else 0.0
                    window_f1 = float(f1_score(batch_labels, y_pred)) if len(set(batch_labels)) > 1 else 0.0

                    test_metrics['auc'] += window_auc
                    test_metrics['f1'] += window_f1
                    test_metrics['ap'] += window_ap
                    test_metrics['precision'] += window_precision
                    test_metrics['recall'] += window_recall
                    test_count += 1
                except Exception as e:
                    print(f"seed: {seed} error: {str(e)}")

        if test_count > 0:
            test_auc = float(test_metrics['auc'] / test_count)
            test_f1 = float(test_metrics['f1'] / test_count)
            test_ap = float(test_metrics['ap'] / test_count)
            test_precision = float(test_metrics['precision'] / test_count)
            test_recall = float(test_metrics['recall'] / test_count)

            print(f"\nseed: {seed} AUC={test_auc:.4f}, F1={test_f1:.4f}, AP={test_ap:.4f}")

            seed_test_results['test_metrics'] = {
                'auc': test_auc,
                'f1': test_f1,
                'ap': test_ap,
                'precision': test_precision,
                'recall': test_recall
            }
            with open(os.path.join(SAVE_DIR, "seed_results", f"seed_{seed}_results.json"), 'w', encoding='utf-8') as f:
                json.dump(seed_test_results, f, indent=2)
        else:
            seed_test_results['test_metrics'] = {'auc': 0.0, 'f1': 0.0, 'ap': 0.0, 'precision': 0.0, 'recall': 0.0}
    elif VAL_BEST_THRESHOLD is None:
        seed_test_results['test_metrics'] = {'auc': 0.0, 'f1': 0.0, 'ap': 0.0, 'precision': 0.0, 'recall': 0.0}
    else:
        seed_test_results['test_metrics'] = {'auc': 0.0, 'f1': 0.0, 'ap': 0.0, 'precision': 0.0, 'recall': 0.0}

    all_seed_results.append(seed_test_results)


    def calculate_global_stats(metadata, all_sub_graphs, all_windows):
        all_degrees = []
        all_frequencies = []

        for windows in all_windows:
            for start_year, end_year in windows:
                for sub_g in all_sub_graphs:
                    if len(sub_g.edata['edge_year']) == 0:
                        continue
                    sub_min_year = sub_g.edata['edge_year'].min().item()
                    sub_max_year = sub_g.edata['edge_year'].max().item()
                    if sub_min_year == start_year and sub_max_year == end_year:
                        node_degrees = (sub_g.out_degrees() + sub_g.in_degrees()).tolist()
                        all_degrees.extend(node_degrees)
                        sub_node_orig_ids = sub_g.ndata[dgl.NID].cpu().numpy()
                        for nid in sub_node_orig_ids:
                            node = metadata['pruned_nodes'][nid]
                            total_freq = 0
                            for year in range(start_year, end_year + 1):
                                total_freq += node.get('frequency_per_year', {}).get(str(year), 0)
                            all_frequencies.append(total_freq)
                        break

        degree_mean = np.mean(all_degrees) if all_degrees else 0
        degree_std = np.std(all_degrees) if len(all_degrees) > 1 and np.std(all_degrees) > 0 else 1.0
        freq_mean = np.mean(all_frequencies) if all_frequencies else 0
        freq_std = np.std(all_frequencies) if len(all_frequencies) > 1 and np.std(all_frequencies) > 0 else 1.0

        return {
            'degree_mean': float(degree_mean),
            'degree_std': float(degree_std),
            'freq_mean': float(freq_mean),
            'freq_std': float(freq_std)
        }

    all_sub_graphs = train_sub_graphs + val_sub_graphs + test_sub_graphs
    all_windows = [TRAIN_WINDOWS, VAL_WINDOW, TEST_WINDOWS]
    global_stats = calculate_global_stats(metadata, all_sub_graphs, all_windows)

    embedding_path = os.path.join(SAVE_DIR, "epoch_embeddings_final", f"final_keyword_embeddings_seed_{seed}.json")
    generate_embeddings(
        g, tgat_model,
        embedding_path,
        metadata, num_nodes, global_stats
    )

val_aucs = [res['best_val_auc'] for res in all_seed_results if res['best_val_auc'] > 0]
test_aucs = [res['test_metrics']['auc'] for res in all_seed_results if res['test_metrics']['auc'] > 0]
test_f1s = [res['test_metrics']['f1'] for res in all_seed_results if res['test_metrics']['f1'] > 0]
test_aps = [res['test_metrics']['ap'] for res in all_seed_results if res['test_metrics']['ap'] > 0]
test_precisions = [res['test_metrics']['precision'] for res in all_seed_results if res['test_metrics']['precision'] > 0]
test_recalls = [res['test_metrics']['recall'] for res in all_seed_results if res['test_metrics']['recall'] > 0]

summary = {
    'total_seeds': len(SEEDS),
    'valid_seeds': len(val_aucs),
    'val_auc': {
        'mean': float(np.mean(val_aucs)),
        'std': float(np.std(val_aucs)),
        'all_values': val_aucs
    },
    'test_auc': {
        'mean': float(np.mean(test_aucs)),
        'std': float(np.std(test_aucs)),
        'all_values': test_aucs
    },
    'test_f1': {
        'mean': float(np.mean(test_f1s)),
        'std': float(np.std(test_f1s)),
        'all_values': test_f1s
    },
    'test_ap': {
        'mean': float(np.mean(test_aps)),
        'std': float(np.std(test_aps)),
        'all_values': test_aps
    },
    'test_precision': {
        'mean': float(np.mean(test_precisions)),
        'std': float(np.std(test_precisions)),
        'all_values': test_precisions
    },
    'test_recall': {
        'mean': float(np.mean(test_recalls)),
        'std': float(np.std(test_recalls)),
        'all_values': test_recalls
    },
    'all_seed_results': all_seed_results
}

with open(os.path.join(SAVE_DIR, "10_seeds_summary_results.json"), 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2)
