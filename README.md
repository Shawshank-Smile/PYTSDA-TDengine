# PyODDS
[![Build Status](https://travis-ci.com/datamllab/PyODDS.svg?branch=master)](https://travis-ci.com/datamllab/PyODDS)
[![Coverage Status](https://coveralls.io/repos/github/datamllab/PyODDS/badge.svg?branch=master)](https://coveralls.io/github/datamllab/PyODDS?branch=master)
[![Documentation Status](https://readthedocs.org/projects/pyodds-handbook/badge/?version=latest)](https://pyodds.github.io/)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/3456033f37744ae2a5a69da448ee430d)](https://www.codacy.com/manual/pyodds/PyODDS?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=pyodds/PyODDS&amp;utm_campaign=Badge_Grade)
[![PyPI version](https://badge.fury.io/py/pyodds.svg)](https://badge.fury.io/py/pyodds)

Official Website: [https://www.taosdata.com/cn/](https://www.taosdata.com/cn/)

TDengine Version：1.6.5.9 ,	Any question about TDengine, wechat : 13720014098

##

**PyODDS** is an end-to end time series data analysis **Python** system with **TDengine**. PyODDS provides algorithms which meet the demands for users in time series data analysis fields, w/wo data science or machine learning background. PyODDS gives the ability to execute machine learning algorithms in-database without moving data out of the database server or over the network. It also provides access to a wide range of time series data analysis algorithms, including statistical analysis and more recent deep learning based approaches. 

PyODDS is featured for:

  - **Full Stack Service** which supports operations and maintenances from light-weight SQL based database to back-end machine learning algorithms and makes the throughput speed faster;

  - **State-of-the-art Time Series Data Analysis Approaches** including **Statistical/Machine Learning/Deep Learning** models with unified APIs and detailed documentation;

  - **Powerful Data Analysis Mechanism** which supports **time-series data** analysis with flexible time-slice(sliding-window) segmentation.  
  
  - **Automated Machine Learning** PyODDS describes the first attempt to incorporate automated machine learning with time series data, and belongs to one of the first attempts to extend automated machine learning concepts into real-world data mining tasks.

The Full API Reference can be found in [`handbook`](https://pyodds.github.io/).

## API Demo:


```sh
from utils.import_algorithm import algorithm_selection
from utils.utilities import output_performance,connect_server,query_data

# connect to the database
conn,cursor=connect_server(host, user, password)

# query data from specific time range
data = query_data(database_name,table_name,start_time,end_time)

# train the anomaly detection algorithm
clf = algorithm_selection(algorithm_name)
clf.fit(X_train)

# get outlier result and scores
prediction_result = clf.predict(X_test)
outlierness_score = clf.decision_function(test)

#visualize the prediction_result
visualize_distribution(X_test,prediction_result,outlierness_score)

```



## Quick Start
```sh
python demo.py --ground_truth --visualize_distribution
```

### Results are shown as
```sh
connect to TDengine success
Load dataset and table
Loading cost: 0.151061 seconds
Load data successful
Start processing:
100%|████████████████████████████████████| 10/10 [00:00<00:00, 14.02it/s]
==============================
Results in Algorithm dagmm are:
accuracy_score: 0.98
precision_score: 0.99
recall_score: 0.99
f1_score: 0.99
roc_auc_score: 0.99
processing time: 15.330137 seconds
==============================
connection is closed

```
<img src="https://github.com/datamllab/PyODDS/blob/master/output/img/Result.png" width="50%" height="45%">

## Installation

To install the package, please use the [`pip`](https://pip.pypa.io/en/stable/installing/) installation as follows:

```sh
pip install pyodds
pip install git+git@github.com:datamllab/PyODDS.git
```
**Note:** PyODDS is only compatible with **Python 3.6** and above.

### Required Dependencies

```sh
- pandas>=0.25.0
- taos==1.4.15
- tensorflow==2.0.0b1
- numpy>=1.16.4
- seaborn>=0.9.0
- torch>=1.1.0
- luminol==0.4
- tqdm>=4.35.0
- matplotlib>=3.1.1
- scikit_learn>=0.21.3
```
目前测试的环境：
```sh
pandas =1.0.3
tensorflow =2.1.0
numpy=1.18.1
seaborn=0.10.1
torch=1.4.0+cpu
luminol=0.4
hyperopt=0.2.4
tqdm=4.46.0
matplotlib=3.1.3
scikit_learn=0.23.1
```


To compile and package the JDBC driver source code, you should have a Java jdk-8 or higher and Apache Maven 2.7 or higher installed. To install openjdk-8 on Ubuntu:

```sh
sudo apt-get install openjdk-8-jdk
```

To install Apache Maven on Ubuntu:

```sh
sudo apt-get install maven
```
To install the TDengine as the back-end database service, please refer to [this instruction](https://www.taosdata.com/en/getting-started/#Install-from-Package).

To enable the Python client APIs for TDengine, please follow [this handbook](https://www.taosdata.com/en/documentation/connector/#Python-Connector). 

To insure the locale in config file is valid:

```sh
sudo locale-gen "en_US.UTF-8"
export LC_ALL="en_US.UTF-8"
locale

```
To start the service after installation, in a terminal, use:
```sh
taosd
```

## Implemented Algorithms
### Statistical Based Methods
Methods | Algorithm | Class API
------------ | -------------|-------------
CBLOF | Clustering-Based Local Outlier Factor | :class:`algo.cblof.CBLOF`
HBOS | Histogram-based Outlier Score | :class:`algo.hbos.HBOS`
IFOREST | Isolation Forest | :class:`algo.iforest.IFOREST`
KNN | k-Nearest Neighbors  | :class:`algo.knn.KNN`
LOF | Local Outlier Factor | :class:`algo.cblof.CBLOF`
OCSVM | One-Class Support Vector Machines | :class:`algo.ocsvm.OCSVM`
PCA | Principal Component Analysis | :class:`algo.pca.PCA`
RobustCovariance | Robust Covariance| :class:`algo.robustcovariance.RCOV`
SOD | Subspace Outlier Detection| :class:`algo.sod.SOD`

### Deep Learning Based Methods
Methods | Algorithm | Class API
------------ | -------------|-------------
autoencoder | Outlier detection using replicator neural networks | :class:`algo.autoencoder.AUTOENCODER`
dagmm | Deep autoencoding gaussian mixture model for unsupervised anomaly detection | :class:`algo.dagmm.DAGMM`

### Time Serie Methods
Methods | Algorithm | Class API
------------ | -------------|-------------
lstmad | Long short term memory networks for anomaly detection in time series | :class:`algo.lstm_ad.LSTMAD`
lstmencdec | LSTM-based encoder-decoder for multi-sensor anomaly detection | :class:`algo.lstm_enc_dec_axl.LSTMED`
luminol | Linkedin's luminol	 | :class:`algo.luminol.LUMINOL`

## APIs Cheatsheet

The Full API Reference can be found in [`handbook`](https://pyodds.github.io/).

  - **connect_server(hostname,username,password)**: Connect to Apache backend TDengine Service.

  - **query_data(connection,cursor,database_name,table_name,start_time,end_time)**: Query data from table *table_name* in database *database_name* within a given time range.

  - **algorithm_selection(algorithm_name,contamination)**: Select an algorithm as detector.

  - **fit(X)**: Fit *X* to detector.

  - **predict(X)**: Predict if instance in *X* is outlier or not.

  - **decision_function(X)**: Output the anomaly score of instances in *X*.

  - **output_performance(algorithm_name,ground_truth,prediction_result,outlierness_score)**: Output the prediction result as evaluation matrix in *Accuracy*, *Precision*, *Recall*, *F1 Score*, *ROC-AUC Score*, *Cost time*.

  - **visualize_distribution(X,prediction_result,outlierness_score)**: Visualize the detection result with the the data distribution.

  - **visualize_outlierscore(outlierness_score,prediction_result,contamination)** Visualize the detection result with the outlier score.


## License
<!-- Biblatex entry: -->

You may use this software under the MIT License.

## Environment configuration
<!-- Biblatex entry: -->
Linux：TDengine Server  1.6.5.9 

Windows：TDengine 1.5.5.9 client & Python Connector  https://www.taosdata.com/cn/documentation/connector/#Python-Connector and follows：

1 anaconda 安装
清华Anaconda镜像地址：https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/

2 全部装上吧
https://github.com/Shawshank-Smile/pyodds
pandas =1.0.3
tensorflow =2.1.0
numpy=1.18.1
seaborn=0.10.1
torch=1.4.0+cpu
luminol=0.4
hyperopt=0.2.4
tqdm=4.46.0
matplotlib=3.1.3
scikit_learn=0.23.1
	
3  https://blog.csdn.net/ANNILingMo/article/details/88032599
https://www.cnblogs.com/andrew-address/p/12733669.html （这个是可以的）
	Torch 没有安装成功，先不搞了。如果需要安装，参考以上的链接吧。

4 安装vscode

5 vscode 使用anaconda的环境
https://blog.csdn.net/m0_45161766/article/details/105729025
https://www.cnblogs.com/lataku/p/10743257.html

6 vscode 配置 git
https://www.cnblogs.com/ostrich-sunshine/p/11329444.html

7 Git安装教程（windows）
https://www.cnblogs.com/wj-1314/p/7993819.html

8 git配置 
https://www.cnblogs.com/lucy-xyy/p/11733317.html
https://www.cnblogs.com/ashidamana/p/6122619.html
https://www.cnblogs.com/wj-1314/p/7992543.html

