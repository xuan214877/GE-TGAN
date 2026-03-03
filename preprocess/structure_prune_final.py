import dgl
import torch
import json
import os
from tqdm import tqdm
from collections import defaultdict


def load_text_embeddings(embedding_path):
    try:
        with open(embedding_path, 'r', encoding='utf-8') as f:
            embedding_data = json.load(f)

        if 'nodes' not in embedding_data:
            raise ValueError("missing nodes")

        keyword_embeddings = {}
        for node in embedding_data['nodes']:
            keyword = node.get('keyword')
            vector = node.get('initial_feature')
            if keyword and vector:
                keyword_embeddings[keyword] = vector

        print(f"{len(keyword_embeddings)} keyword embedding")
        return keyword_embeddings

    except Exception as e:
        print(f"fail: {str(e)}")
        return {}


def build_cooccurrence_graph(embedding_path, min_co_occurrence=2):
    try:
        with open(embedding_path, 'r', encoding='utf-8') as f:
            embedding_data = json.load(f)

        if 'nodes' not in embedding_data or 'edges' not in embedding_data:
            raise ValueError("missing required field")

        keyword_papers = defaultdict(set)
        keyword_frequency_per_year = defaultdict(lambda: defaultdict(int))
        for node in embedding_data['nodes']:
            keyword = node.get('keyword')
            occurrences = node.get('occurrences', {})
            original_rows = occurrences.get('original_rows', [])
            years = occurrences.get('years', [])
            if keyword and original_rows:
                keyword_papers[keyword] = set(original_rows)
                for year in years:
                    keyword_frequency_per_year[keyword][year] += 1

        edges = []
        for edge in embedding_data['edges']:
            source = edge.get('source')
            target = edge.get('target')
            years = edge.get('co_occurrence_years', [])
            paper_ids = edge.get('paper_ids', [])

            if source and target and source in keyword_papers and target in keyword_papers:
                common_papers = keyword_papers[source].intersection(keyword_papers[target])
                if len(common_papers) >= min_co_occurrence:
                    if not years and paper_ids:
                        try:
                            years = sorted(set(int(pid.split('_')[0]) for pid in paper_ids))
                        except:
                            years = []

                    edges.append({
                        'source': source,
                        'target': target,
                        'co_occurrence_years': years,
                        'paper_ids': list(common_papers),
                        'co_occurrence_count': len(common_papers)
                    })

        print(f"construct {len(edges)} original edges")

        nodes = []
        for node in embedding_data['nodes']:
            keyword = node.get('keyword')
            if keyword and keyword in keyword_papers:
                nodes.append({
                    'keyword': keyword,
                    'initial_feature': node.get('initial_feature'),
                    'degree': node.get('degree', len(keyword_papers[keyword])),
                    'frequency_per_year': dict(keyword_frequency_per_year[keyword])
                })

        return nodes, edges

    except Exception as e:
        print(f"fail: {str(e)}")
        import traceback
        traceback.print_exc()
        return [], []


def graph_pruning(nodes, edges, min_degree=3, min_co=3, device='cpu'):
    node_indices = {n['keyword']: idx for idx, n in enumerate(nodes)}

    filtered_edges = []
    edge_info = [] 

    for e in tqdm(edges):
        src, dst = e['source'], e['target']
        if src in node_indices and dst in node_indices:
            co_count = e.get('co_occurrence_count', len(e.get('co_occurrence_years', [])))
            if co_count >= min_co:
                years = sorted(list(set(e.get('co_occurrence_years', []))))

                for edge_year in years:
                    filtered_edges.append((
                        node_indices[src],
                        node_indices[dst],
                        edge_year 
                    ))

                    edge_info.append({
                        'source': src,
                        'target': dst,
                        'edge_year': edge_year,
                        'co_occurrence_years': e['co_occurrence_years'], 
                        'co_occurrence_count': co_count,
                        'paper_ids': e['paper_ids']
                    })

    print(f"{len(filtered_edges)} time edges")

    if not filtered_edges:
        return [], dgl.graph(([], []), num_nodes=0), []

    src_list, dst_list, edge_years = zip(*filtered_edges)
    g = dgl.graph((src_list, dst_list), num_nodes=len(nodes)).to(device)
    g.edata['edge_year'] = torch.tensor(edge_years, dtype=torch.long, device=device) 

    node_degrees = g.out_degrees(g.nodes()) + g.in_degrees(g.nodes())
    valid_mask = node_degrees >= min_degree
    valid_indices = torch.where(valid_mask)[0].tolist()

    orig2new = {orig: new for new, orig in enumerate(valid_indices)}

    pruned_edges = []
    pruned_edge_info = []
    node_degree_per_year = defaultdict(lambda: defaultdict(int))
    node_co_occurrence_years = defaultdict(set)

    for edge in edge_info:
        src_key = edge['source']
        dst_key = edge['target']

        if src_key in node_indices and dst_key in node_indices:
            src_idx = node_indices[src_key]
            dst_idx = node_indices[dst_key]

            if src_idx in orig2new and dst_idx in orig2new:
                edge_year = edge['edge_year'] 
                node_degree_per_year[src_key][edge_year] += 1
                node_degree_per_year[dst_key][edge_year] += 1

                node_co_occurrence_years[src_key].add(edge_year)
                node_co_occurrence_years[dst_key].add(edge_year)

                pruned_edges.append((orig2new[src_idx], orig2new[dst_idx]))
                pruned_edge_info.append(edge)

    if not pruned_edges:
        return [], dgl.graph(([], []), num_nodes=0), []

    final_g = dgl.graph(pruned_edges, num_nodes=len(valid_indices)).to(device)

    final_g.edata['edge_year'] = torch.tensor(
        [e['edge_year'] for e in pruned_edge_info],
        dtype=torch.long,
        device=device
    )

    final_nodes = [nodes[i] for i in valid_indices]
    for node in final_nodes:
        keyword = node['keyword']
        node['degree_per_year'] = dict(node_degree_per_year[keyword])
        node['co_occurrence_years'] = sorted(node_co_occurrence_years[keyword])

    return final_nodes, final_g, pruned_edge_info


def save_pruned_network(nodes, graph, edge_info, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    node_data = []
    for n in nodes:
        node_entry = {
            'keyword': n['keyword'],
            'initial_feature': n['initial_feature'],
            'original_degree': n.get('degree', 0),
            'degree_per_year': n.get('degree_per_year', {}),
            'frequency_per_year': n.get('frequency_per_year', {}),
            'co_occurrence_years': n.get('co_occurrence_years', [])
        }
        node_data.append(node_entry)

    with open(os.path.join(save_dir, 'pruned_nodes1.json'), 'w', encoding='utf-8') as f:
        json.dump(node_data, f, indent=2, ensure_ascii=False)

    with open(os.path.join(save_dir, 'pruned_edges1.json'), 'w', encoding='utf-8') as f:
        json.dump(edge_info, f, indent=2, ensure_ascii=False)

    dgl.save_graphs(os.path.join(save_dir, 'pruned_graph1.bin'), [graph])

    print(f"{graph.num_nodes()} nodes, {graph.num_edges()} edges")


if __name__ == "__main__":
    EMBEDDING_PATH = "E:/preprocess/data/keywords_textembeddings.json"
    SAVE_DIR = "E:/preprocess/pruned_network"
    MIN_DEGREE = 1
    MIN_CO = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nodes, edges = build_cooccurrence_graph(EMBEDDING_PATH, min_co_occurrence=MIN_CO)

    if not nodes or not edges:
        print("please check file format")
    else:
        pruned_nodes, pruned_graph, edge_info = graph_pruning(
            nodes,
            edges,
            min_degree=MIN_DEGREE,
            min_co=MIN_CO,
            device=device
        )

        save_pruned_network(pruned_nodes, pruned_graph, edge_info, SAVE_DIR)