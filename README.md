# Knowledge_graph_completion_with_message_passing

## Contributors
* Ridha Alkhabaz (ridhama2@illinois.edu/ ridha.alkhabaz@gmail.com)



## Project overview
Here, we study the **Knowledge Graph Completion** (KGC). We challenge the recent findings made by [Li et al](https://arxiv.org/pdf/2205.10652.pdf). We find two message passing graph neural networks outperforming MLP methods mentioned in his method.  
### Relevant Publications
* https://paperswithcode.com/paper/gpatcher-a-simple-and-adaptive-mlp-model-for
* https://aclanthology.org/2023.acl-long.597.pdf
* https://arxiv.org/pdf/1802.09691.pdf

### Relevant git repository:
* https://github.com/facebookresearch/SEAL_OGB/tree/main
* https://github.com/Juanhui28/Are_MPNNs_helpful



## File structure:

* **Data**: you can download the data using `from torch_geometric.datasets import FB15K-237`. Then, run the code `dataset = FB15k_237('data')` in our `experimentations.ipynb`. Or find the data in data folder. 

* **Running**: to reproduce our seal resutls, run the following notebook:
* SEAL replication: `seal_exp.ipynb`
* Embed replication : `node2vec.ipynb`
* RCGNN: `gcn.ipynb`
* results: `visualizations.ipynb`


please note that you need to download the data first. 


* **Analysis**
As seen in the paper, we see a significant difference in performance between Li et al.'s MLP-based methods and SEAL-DCGNN. We may attribute this difference to the fact that FB15K-237 is a featureless graph. Thus, Li et al.'s work tried to mitigate this issue by generating random features for nodes. The randomly generated features might have contributed to CompGCN's sub-bar performance compared to SEAL-DCGNN. This also might explain SEAL-DCGNN's exceptional accuracy. Since FB15K-237 is featureless, SEAL-DCGNN utilizes SEAL's DRNL labeling to add positional features to nodes and produce more informative embedding. 

More importantly, we notice something interesting in our results. All embedding-based methods outperform rGCN and MLP-based methods. However, they still significantly underperform compared to SEAL-DCGNN. We believe this due to two reasons. First, the proximity measure used to determine entities' relation is similar to the $\gamma-$decaying heuristic that Zhang and Chen mentioned \cite{zhang2018link}. The main similarity stems from knowledge graph entities' exponentially decaying relevance with number of hops. Thus, SEAL's 1-hop approximation is very suited for knowledge graph completion. Second reason, DCGNN 

## Results
* We achieve around 97% mean reciporical rank. 


