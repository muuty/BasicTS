#!/usr/bin/env python
"""
Experiment Results Aggregator

Usage:
    # Collect all SAN_BERNARDINO results
    python experiments/get_results.py

    # Filter by type
    python experiments/get_results.py --type freeze

    # Custom output
    python experiments/get_results.py --output my_results.csv
"""
import os
import json
import argparse
import yaml
import re
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Set


INCIDENT_KEYS = [
    'non_incident_overall',
    'all_incident_overall'
]

INCIDENT_PREFIX_MAP = {
    'non_incident_overall': 'non_inc',
    'all_incident_overall': 'inc'
}


# =============================================================================
# NEW: Simple results aggregation
# =============================================================================

def load_metrics(checkpoint_dir: str) -> dict:
    """Load test_metrics.json and test_incident_metrics.json."""
    result = {}

    # Load main metrics
    metrics_path = os.path.join(checkpoint_dir, 'test_metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            data = json.load(f)

            # Overall
            if 'overall' in data:
                result['MAE'] = data['overall'].get('MAE')
                result['RMSE'] = data['overall'].get('RMSE')

            # Robustness - per_sample
            if 'robustness' in data and 'per_sample' in data['robustness']:
                ps = data['robustness']['per_sample']
                result['worst_1pct'] = ps.get('worst_1pct_MAE')
                result['worst_5pct'] = ps.get('worst_5pct_MAE')
                result['worst_10pct'] = ps.get('worst_10pct_MAE')

            # Robustness - per_node
            if 'robustness' in data and 'per_node' in data['robustness']:
                pn = data['robustness']['per_node']
                result['node_std'] = pn.get('std_MAE')

    # Load incident metrics
    incident_path = os.path.join(checkpoint_dir, 'test_incident_metrics.json')
    if os.path.exists(incident_path):
        with open(incident_path, 'r') as f:
            data = json.load(f)
            # New format: nested objects
            if 'incident_overall' in data:
                result['inc_MAE'] = data['incident_overall'].get('MAE')
            if 'normal_overall' in data:
                result['non_inc_MAE'] = data['normal_overall'].get('MAE')
            # Old format: direct values
            if 'overall_incident_MAE' in data:
                result['inc_MAE'] = data['overall_incident_MAE']
            if 'overall_normal_MAE' in data:
                result['non_inc_MAE'] = data['overall_normal_MAE']

    return result


def parse_experiment_name(path: str) -> dict:
    """Extract experiment info from checkpoint path."""
    info = {'experiment': None, 'dataset': None, 'method': None, 'type': None}

    for part in path.split('/'):
        if 'ContextContrastive' in part:
            exp_name = part.replace('ContextContrastive_', '')
            info['experiment'] = exp_name

            if 'freeze_' in part:
                info['type'] = 'freeze'
                info['method'] = exp_name.replace('freeze_', '').replace('_3mo', '')
            elif 'finetune_' in part:
                info['type'] = 'finetune'
                info['method'] = exp_name.replace('finetune_', '').replace('_3mo', '')
            elif 'pretrain' in part.lower():
                info['type'] = 'pretrain'
                info['method'] = exp_name.replace('_3mo', '')
            elif 'baseline' in part.lower():
                info['type'] = 'baseline'
                info['method'] = 'STAEformer'
            elif 'scratch' in part.lower():
                info['type'] = 'scratch'
                info['method'] = exp_name.replace('_3mo', '')
            else:
                info['type'] = 'other'
                info['method'] = exp_name.replace('_3mo', '')

        # Also check for non-ContextContrastive baselines
        if 'GPTSTForForecasting' in part:
            info['type'] = 'baseline'
            info['method'] = 'GPT-ST'
            info['experiment'] = 'GPT-ST'
        if 'STAEformer_with_TemporalEncoder' in part:
            info['type'] = 'baseline'
            info['method'] = 'STAEformer+TE'
            info['experiment'] = 'STAEformer+TE'
        if 'STMAE_finetune' in part:
            info['type'] = 'finetune'
            info['method'] = 'STMAE'
            info['experiment'] = 'STMAE_finetune'
        if 'STMAE_pretrain' in part:
            info['type'] = 'pretrain'
            info['method'] = 'STMAE'
            info['experiment'] = 'STMAE_pretrain'

        if 'SAN_BERNARDINO' in part:
            info['dataset'] = 'SAN_BERNARDINO'
        elif 'PEMS' in part:
            info['dataset'] = part.split('_')[0]

    return info


def find_all_results(base_dir: str, dataset_filter: str = None) -> list:
    """Find all experiment results."""
    results = []

    for root, dirs, files in os.walk(base_dir):
        if 'test_metrics.json' not in files:
            continue
        if '_old_' in root:
            continue

        info = parse_experiment_name(root)

        if dataset_filter and info['dataset'] != dataset_filter:
            continue

        metrics = load_metrics(root)
        if not metrics or metrics.get('MAE', 0) > 50:
            continue

        results.append({'path': root, **info, **metrics})

    return results


def format_val(val, prec=2):
    """Format value for display."""
    if val is None:
        return '-'
    return f'{val:.{prec}f}' if isinstance(val, float) else str(val)


def print_table(df: pd.DataFrame, title: str):
    """Print formatted table."""
    print("\n" + "=" * 110)
    print(f" {title}")
    print("=" * 110)

    if df.empty:
        print("No results.")
        return

    df_sorted = df.sort_values('MAE')

    # Header
    print(f"{'Method':<25} {'Type':<10} {'MAE':>7} {'Inc':>7} {'NonInc':>7} "
          f"{'W1%':>7} {'W5%':>7} {'W10%':>7} {'NodeStd':>7}")
    print("-" * 110)

    for _, r in df_sorted.iterrows():
        print(f"{str(r.get('method', '-')):<25} {str(r.get('type', '-')):<10} "
              f"{format_val(r.get('MAE')):>7} {format_val(r.get('inc_MAE')):>7} "
              f"{format_val(r.get('non_inc_MAE')):>7} {format_val(r.get('worst_1pct')):>7} "
              f"{format_val(r.get('worst_5pct')):>7} {format_val(r.get('worst_10pct')):>7} "
              f"{format_val(r.get('node_std')):>7}")

    print("=" * 110)
    print(f"Total: {len(df)} experiments\n")


def aggregate_results(base_dir='checkpoints', dataset='SAN_BERNARDINO',
                      output=None, type_filter=None):
    """Main aggregation function."""
    results = find_all_results(base_dir, dataset)

    if not results:
        print("No results found!")
        return None

    df = pd.DataFrame(results)

    if type_filter:
        df = df[df['type'] == type_filter]

    # Print to terminal
    print_table(df, f"Results ({dataset})")

    # Save to CSV
    if output is None:
        output = f'experiments/results_{dataset}.csv'

    os.makedirs(os.path.dirname(output), exist_ok=True)

    cols = ['method', 'type', 'MAE', 'RMSE', 'inc_MAE', 'non_inc_MAE',
            'worst_1pct', 'worst_5pct', 'worst_10pct', 'node_std',
            'dataset', 'experiment', 'path']
    cols = [c for c in cols if c in df.columns]

    df.sort_values('MAE')[cols].to_csv(output, index=False)
    print(f"Saved: {output}")

    return df


def load_experiment_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def extract_target_keys(exp_config: Dict) -> List[str]:
    target_keys: Set[str] = set()
    
    for run in exp_config.get('runs', []):
        params = run.get('params', {})
        target_keys.update(params.keys())
    
    return sorted(list(target_keys))


def parse_cfg_txt(cfg_path: str) -> Dict[str, Any]:
    with open(cfg_path, 'r') as f:
        content = f.read()
        content = re.sub(r'tensor\((.*?)\)', r'"\1"', content, flags=re.DOTALL)
        return yaml.safe_load(content) or {}


def get_nested_value(d: Dict, key_path: str) -> Any:
    keys = key_path.split('.')
    value = d
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return None
    return value


def make_column_name(key_path: str) -> str:
    return key_path.lower().replace('.', '_')


def extract_metadata(config: Dict) -> Dict[str, Any]:
    metadata = {}
    
    model_name = get_nested_value(config, 'MODEL.NAME')
    if model_name:
        metadata['model'] = model_name
    
    dataset_name = get_nested_value(config, 'DATASET.NAME')
    if dataset_name:
        metadata['dataset'] = dataset_name.split('/')[-1] if '/' in dataset_name else dataset_name
    
    return metadata


def extract_params(config: Dict, target_keys: List[str]) -> Dict[str, Any]:
    result = {}
    for key in target_keys:
        value = get_nested_value(config, key)
        col_name = make_column_name(key)
        result[col_name] = value
    return result


def find_experiment_results(base_dir: str, exp_name: str) -> List[Dict[str, Any]]:
    results = []
    exp_path = os.path.join(base_dir, exp_name)
    
    for root, dirs, files in os.walk(exp_path):
        if 'cfg.txt' not in files or 'test_metrics.json' not in files:
            continue
        
        result_entry = {
            'path': root,
            'cfg': parse_cfg_txt(os.path.join(root, 'cfg.txt')),
            'metrics': {},
            'incident_metrics': {}
        }
        
        with open(os.path.join(root, 'test_metrics.json'), 'r') as f:
            result_entry['metrics'] = json.load(f)
        
        incident_path = os.path.join(root, 'test_incident_metrics.json')
        if os.path.exists(incident_path):
            with open(incident_path, 'r') as f:
                result_entry['incident_metrics'] = json.load(f)
        
        results.append(result_entry)
    
    return results


def extract_run_from_path(path: str, base_dir: str, exp_name: str) -> str:
    prefix = os.path.join(base_dir, exp_name)
    if path.startswith(prefix):
        relative = path[len(prefix):].lstrip(os.sep)
        return relative
    return path


def build_records(experiment_results: List[Dict], target_keys: List[str], 
                  base_dir: str, exp_name: str, metrics: List[str]) -> List[Dict]:
    records = []

    for result in experiment_results:
        cfg = result['cfg']
        metrics_data = result['metrics']
        incident_metrics = result['incident_metrics']
        path = result['path']
        
        metadata = extract_metadata(cfg)
        
        record = {
            'model': metadata.get('model', 'unknown'),
            'dataset': metadata.get('dataset', 'unknown'),
            'run': extract_run_from_path(path, base_dir, exp_name),
        }
        
        params = extract_params(cfg, target_keys)
        record.update(params)
        
        if 'overall' in metrics_data:
            for m in metrics:
                record[m] = metrics_data['overall'].get(m)
        
        for inc_key in INCIDENT_KEYS:
            if inc_key in incident_metrics:
                prefix = INCIDENT_PREFIX_MAP[inc_key]
                for m in metrics:
                    record[f"{prefix}_{m}"] = incident_metrics[inc_key].get(m)
        
        records.append(record)
    
    return records


def compute_statistics(df: pd.DataFrame, target_keys: List[str]) -> pd.DataFrame:
    if df.empty:
        return df
    
    meta_cols = ['model', 'dataset']
    param_cols = [make_column_name(k) for k in target_keys if make_column_name(k) in df.columns]
    group_cols = meta_cols + param_cols
    
    metric_cols = [c for c in df.columns if c not in group_cols and c != 'run']
    
    if not metric_cols:
        return df
    
    agg_dict = {col: ['mean', 'std'] for col in metric_cols}
    agg_dict[metric_cols[0]].append('count')
    
    stats = df.groupby(group_cols, dropna=False).agg(agg_dict).reset_index()
    
    new_columns = []
    for col in stats.columns:
        if isinstance(col, tuple):
            if col[1] == '':
                new_columns.append(col[0])
            elif col[1] == 'count':
                new_columns.append('n_runs')
            else:
                new_columns.append(f"{col[0]}_{col[1]}")
        else:
            new_columns.append(col)
    
    stats.columns = new_columns
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--base_dir', type=str, default='checkpoints')
    parser.add_argument('--output_dir', type=str, default='experiments/result')
    parser.add_argument('--metrics', type=str, nargs='+', default=['MAE'], 
                        choices=['MAE', 'MAPE', 'RMSE'])
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading config: {args.config}")
    exp_config = load_experiment_config(args.config)
    target_keys = extract_target_keys(exp_config)
    exp = Path(args.config).stem
    
    if target_keys:
        print(f"Target parameters:")
        for key in target_keys:
            print(f"  {key} -> {make_column_name(key)}")
    
    print(f"Metrics: {args.metrics}")
    
    print(f"\nSearching: {args.base_dir}/{exp}")
    experiment_results = find_experiment_results(args.base_dir, exp)
    
    if not experiment_results:
        print("No results found!")
        return
    
    print(f"Found {len(experiment_results)} result(s)")
    
    records = build_records(experiment_results, target_keys, args.base_dir, exp, args.metrics)
    df = pd.DataFrame(records)
    
    if df.empty:
        print("No valid results")
        return
    
    stats_df = compute_statistics(df, target_keys)
    
    output_path = os.path.join(args.output_dir, f'{exp}.csv')
    stats_df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")
    
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Models: {df['model'].unique().tolist()}")
    print(f"Datasets: {df['dataset'].unique().tolist()}")
    print(f"Total runs: {len(df)}")
    
    param_cols = [make_column_name(k) for k in target_keys if make_column_name(k) in df.columns]
    if param_cols:
        print(f"\nParameters:")
        for col in param_cols:
            print(f"  {col}: {df[col].dropna().unique().tolist()}")


def main_simple():
    """Simple CLI for results aggregation."""
    parser = argparse.ArgumentParser(description='Aggregate experiment results')
    parser.add_argument('--base_dir', default='checkpoints')
    parser.add_argument('--dataset', default='SAN_BERNARDINO')
    parser.add_argument('--output', default=None)
    parser.add_argument('--type', choices=['freeze', 'finetune', 'pretrain', 'baseline', 'scratch'])
    args = parser.parse_args()

    aggregate_results(args.base_dir, args.dataset, args.output, args.type)


if __name__ == '__main__':
    import sys
    # If --config is provided, use old behavior; otherwise use simple aggregation
    if '--config' in sys.argv:
        main()
    else:
        main_simple()