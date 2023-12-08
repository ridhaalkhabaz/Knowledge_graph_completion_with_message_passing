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

* **Data**: you can download the data using `from torch_geometric.datasets import FB15K-237`. Then, run the code `dataset = FB15k_237('data')` in our `experimentations.ipynb`

* **Running**: to reproduce our seal resutls, run the following notebook:
* SEAL replication: `seal_exp.ipynb`
* Node2Vec replication : `node2vec.ipynb`
* RCGNN: `gcn.ipynb`


please note that you need to download the data first. 

In `li_work_rep` folder, we include the necessary scripts to reproduce [Li et al work](https://aclanthology.org/2023.acl-long.597.pdf). Please note, you need to download the data in similar fashion to reproduce inside the folder. Also, it takes more than 5 hours to produce a single experiment. 

* **Analysis**
Since knwoledge graphs at hands have little to no node features, it might be implied that message passing techniques on generated noise might result in poor perform. Thus, the neural node labeling applied in SEAL is a good first step. 


## Results
* We achieve around 97% mean reciporical rank. 
* more results to added later on. 


