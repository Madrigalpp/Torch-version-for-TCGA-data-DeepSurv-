# Deep-Learning-for-survival-analysis  **Pytorch**
realization DeepSurv and Coxnnet in TCGA mi-RNA analysis   
This project's paper includes a section dedicated to its deep learning analysis, specifically focusing on the implementation of the XGBENC deep learning analysis： [XGBENC](google.com)  
##  Overview
This project uses deep learning methods to implement prognostic analysis of TCGA-miRNA. We propose a scalable code framework covering grid search, 5-fold cross-validation methods. This project analyzes two veteran survival analysis algorithms. DeepSurv and Coxnnet and give model evaluation.
##  Data-available
you should download from TCGA and convert to s-g_data100_data.csv format
##  How to use ？
**network.py** contains all the Network settings and Partial Likelihood loss function  
**ini_file.py** contain all the hyper parameters  
**utils.py** contain the c-index calculation and other settings  
you can run **DL_survival_main** to run the model.

##  Reference
[czifan/DeepSurv.pytorch](https://github.com/czifan/DeepSurv.pytorch) (czifan@pku.edu.cn)  
if you have any problems, please contact Wankang Zhai (wzhai2@uh.edu) 

