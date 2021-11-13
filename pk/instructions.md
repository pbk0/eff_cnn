# Multi-Experiments

We do know that the models obtained with different hyper-parameters perform differently.
But, here we use same models and same well tuned hyper-parameters and show that they perform 
differently.

Here we will do 100 experiments (using same model and dataset):

+ In each experiment:
  + We perform training with
    + given model architecture
    + given hyper-parameters and
    + given epoch
  + Then we compute average rank for trained model
    + average rank is computed using attack dataset which is randomly shuffled for each attack
    
    
Also, we will implement early stopping based on validation accuracy and rerun experiments again.


## Install

python version used: `3.7.10`

```cmd
pip install numpy==1.19.5 
pip install tqdm==4.41.1 
pip install h5py==2.10.0 
pip install matplotlib==3.2.2 
pip install tensorflow==1.15.2 
pip install keras==2.3.1 
pip install scikit-learn==0.24.2 
pip install kaleido==0.2.1 
pip install psutil==5.8.0 
pip install pandas==1.2.4 
pip install plotly==5.0.0 
```

## Check reports

You can view experiments precomputed results across different datasets with below links 

+ [ascad_0](reports/report_ascad_0.md)
+ [ascad_50](reports/report_ascad_50.md)
+ [ascad_100](reports/report_ascad_100.md)
+ [aes_hd](reports/report_aes_hd.md)
+ [aes_rd](reports/report_aes_rd.md)
+ [dpav4](reports/report_dpav4.md)

## Some numbers

### ASCAD-fk-desync-0

|Model|Reported|Observed Range|Failed cases|
|---|---|---|---|
|eff-cnn|191|280-inf|4%|
|simplified-eff-cnn|NA|178-391|0%|
|aisy-hw-mlp|
|aisy-hw-cnn|
|aisy-id-mlp|129|
|aisy-id-cnn|158|

## Links

+ [eff-cnn](https://tches.iacr.org/index.php/TCHES/article/view/8391/7787)
+ [simplified-eff-cnn](https://tches.iacr.org/index.php/TCHES/article/view/8586/8153)
+ [aisy1](https://eprint.iacr.org/2020/1293.pdf)
+ [aisy2](https://eprint.iacr.org/2021/071.pdf)


[forked repo](https://github.com/SpikingNeuron/TCHES20V3_CNN_SCA)
