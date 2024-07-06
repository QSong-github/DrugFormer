# DrugFormer


This repository is prepared for ''DrugFormer: graph-enhanced language model to predict drug sensitivity.''

## Overview

Drug resistance poses a crucial challenge in healthcare, with response rates to chemotherapy and targeted therapy remaining low. Individual patient’s resistance is exacerbated by the intricate heterogeneity exhibited by tumor cells, a primary hurdle limiting the effectiveness of targeted therapies. Such intra-tumoral cellular heterogeneity presents significant obstacles to effective treatment. However, existing studies investigating drug resistance often rely on cell line-based knowledge, lacking in vivo relevance and generalization capability. Recently, the emerging single-cell RNA sequencing (scRNA-seq) technologies provide opportunities to unravel the mechanisms of drug resistance at the cellular level, offering insights into effective therapeutic targets. To facilitate such insights, we propose DrugFormer, a novel graph augmented large language model designed to predict drug resistance at single cell level. DrugFormer integrates both serialized gene tokens and gene-based knowledge graph for the accurate predictions of drug response. After training on comprehensive single-cell data with drug response information, DrugFormer model presents outperformance, with higher F1, precision, and recall in predicting drug response. Based on the scRNA-seq data from refractory multiple myeloma (MM) and acute myeloid leukemia (AML) patients, DrugFormer demonstrates high efficacy in identifying resistant cells and uncovering underlying molecular mechanisms. Through pseudotime trajectory analysis, we reveal unique drug-resistant cellular states associated with poor patient outcomes. Furthermore, DrugFormer identifies potential therapeutic targets, such as COX8A, for overcoming drug resistance across different cancer types. In conclusion, DrugFormer represents a significant advancement in the field of drug resistance prediction, offering a powerful tool for unraveling the heterogeneity of cellular response to drugs and guiding personalized treatment strategies.

## Installation
Download DrugFormer:
```git clone https://github.com/QSong-github/DrugFormer```


Install Environment:
```pip install -r requirements.txt``` or ```conda env create -f environment.yml```



## Running

   Here we provide a small datset sved in 'subdt' for testing. Please use the 'cell_dt' for training. And before running the code, please download the original data from [google drive](https://drive.google.com/file/d/16Tf6opBb8NJ8comha2kV9ohwXmoD7sFC/view?usp=drive_link) and generate the dataset. 
   
   (1) Run ```python gene_graph.py``` to get the knowledge graph.
   
   (2) Run ```python dataset.py``` to get the datasets.
   
   (3) Run ```python main.py``` to get the prediction results.
   


   
