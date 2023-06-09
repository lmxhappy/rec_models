# rec_models

* dssm
* eges
  * eges_1：
    * 用的是二分类交叉熵loss
    * torch
  * eges_2: 
    * 路径：eges/EGES_2/run_EGES.py
    * 用的是sampled softmax loss。
    * tf
* comirec
  * 路径：ComiRec/src/train.py
    * 参数： --dataset book --model_type ComiRec-SA
    * ComiRec-SA、ComiRec-DR
  * 用的是sampled softmax loss。
  * tf 1.13
  * python 3.6

* item2vec：
  * 路径：item2vec/calculate_similar_items.py
    * 参数：--data data/data.csv --epochs 30
  * 用的是nce loss。
  * tf 1.13
  * python 3.6
