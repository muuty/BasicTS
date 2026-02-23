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


if __name__ == '__main__':
    main()