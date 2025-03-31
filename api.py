from typing import List, Literal

from fastapi import FastAPI, Query

from easytorch.device import set_device_type
set_device_type('cpu')
from easytorch.config import init_cfg
from baselines.STGCN.PEMS03 import CFG as PEMS03CFG
from baselines.STGCN.PEMS04 import CFG as PEMS04CFG
from baselines.STGCN.PEMS07 import CFG as PEMS07CFG
from baselines.STGCN.PEMS08 import CFG as PEMS08CFG
from baselines.STGCN.PEMS_BAY import CFG as PEMSBAYCFG
from baselines.STGCN.METR_LA import CFG as METRLACFG

from fastapi.middleware.cors import CORSMiddleware



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 또는 ["http://localhost:63342"]처럼 정확히 지정
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
datasets = {}

representatives = {
    "PEMS03" : [240,325,242,36,273,241,113,57,174,221],
    "PEMS04" : [183,196,236,142,63,254,86,19,177,34],
    "PEMS07" : [638,91,259,396,843,200,349,300,624,833],
    "PEMS08" : [169,73,85,142,72,161,149,51,100,137],
    "PEMS_BAY" : [177,246,241,247,233,156,82,73,144,223],
    "METR_LA" : [7,193,159,56,19,169,177,204,18,199],
}

# 예시: 데이터 로딩
def load_dataset():
    names = ['PEMS03', 'PEMS04', 'PEMS07', 'PEMS08', 'PEMS_BAY', 'METR_LA']
    CFGs = [PEMS03CFG, PEMS04CFG, PEMS07CFG, PEMS08CFG, PEMSBAYCFG, METRLACFG]

    for name, CFG in zip(names, CFGs):
        cfg = init_cfg(CFG, True)
        runner = CFG.RUNNER(cfg)
        train_dataset = runner.build_train_dataset(cfg)
        test_dataset = runner.build_test_dataset(cfg)
        datasets[name] = {"train": train_dataset, "test": test_dataset}

print("Loading datasets...")
load_dataset()
print("Datasets loaded.")

@app.get("/data/batch")
def get_batch_data(
    dataset: str,
    query_index: int,
    topk_indices: List[int] = Query(...)
):
    print(f"Getting batch data for dataset {dataset}...")
    """
    Returns 1 query (test) sample + k support (train) samples.
    """
    ds_test = datasets[dataset]["test"]
    ds_train = datasets[dataset]["train"]

    representative_sensors = representatives[dataset]
    query_inputs = ds_test[query_index]["inputs"][:,representative_sensors,:].tolist()
    query_target = ds_test[query_index]["target"][:,representative_sensors,:].tolist()
    topk_inputs = [ds_train[i]["inputs"][:,representative_sensors,:].tolist() for i in topk_indices]
    topk_target = [ds_train[i]["target"][:,representative_sensors,:].tolist() for i in topk_indices]

    return {
        "dataset": dataset,
        "query_index": query_index,
        "query_inputs": query_inputs,
        "query_target": query_target,
        "topk_indices": topk_indices,
        "topk_inputs": topk_inputs,
        "topk_target": topk_target
    }