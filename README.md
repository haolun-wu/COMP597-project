# Reinforcement Knowledge Graph Reasoning for Explainable Recommendation
This repository contains the source code of the SIGIR 2019 paper "[Reinforcement Knowledge Graph Reasoning for Explainable Recommendation](https://arxiv.org/abs/1906.05237)" [2].

## Datasets
Two Amazon datasets (Amazon_Beauty, Amazon_Cellphones) are available in the "data/" directory and the split is consistent with [1].
All four datasets used in this paper can be downloaded [here](https://drive.google.com/uc?export=download&confirm=Tiux&id=1CL4Pjumj9d7fUDQb1_leIMOot73kVxKB).

## Requirements
- Python >= 3.6
- PyTorch = 1.0

## Skip the training phase
We've included all the required files for the Cellphone dataset so you can simply evaluate performance and play around with the recommendation parameters
1. Copy the contents of this folder ( https://drive.google.com/open?id=1ZbR7nWmpmDjMYmv56ChxonB5q3gKEsAI ) to the root directory of your project

2. Run the multi-agent evaluation
```bash
python test_multi_agent.py --dataset cell --run_path False --run_eval True --re_rank True
```

## How to run the code from scratch
1. Proprocess the data first:
```bash
python preprocess.py --dataset <dataset_name>
```
"<dataset_name>" should be one of "cd", "beauty", "cloth", "cell" (refer to utils.py).

2. Train knowledge graph embeddings (TransE in this case):
```bash
python train_transe_model.py --dataset <dataset_name>
```

3. Train User RL agent:
```bash
python train_agent.py --dataset <dataset_name>
```

4. Evaluate the User agent
```bash
python test_agent.py --dataset <dataset_name> --run_path True --run_eval True
```
If "run_path" is True, the program will generate paths for recommendation according to the trained policy. **Note that for subsequent runs you should set this to 'False'**

If "run_eval" is True, the program will evaluate the recommendation performance based on the resulting paths.

5. Train Product RL agent:
```bash
python train_product_agent.py --dataset <dataset_name>
```

6. Evaluate the Product agent
```bash
python test_product_agent.py --dataset <dataset_name> --run_path True --run_eval True
```

7. Evaluate the final multi-agent setup
```bash
python test_multi_agent.py --dataset <dataset_name> --run_path False --run_eval True --re_rank True
```

## References
[1] Yongfeng Zhang, Qingyao Ai, Xu Chen, W. Bruce Croft. "Joint Representation Learning for Top-N Recommendation with Heterogeneous Information Sources". In Proceedings of CIKM. 2017.

[2] Yikun Xian, Zuohui Fu, S. Muthukrishnan, Gerard de Melo, Yongfeng Zhang. "Reinforcement Knowledge Graph Reasoning for Explainable Recommendation." In Proceedings of SIGIR. 2019.
