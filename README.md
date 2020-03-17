# Predictively Encoded Graph Convolutional Network
Official project for 'Predictively Encoded Graph Convolutional Network for Noise-Robust Skeleton-based Action Recognition'
(Implemented with Py36 and Pytorch)
![](./123.png)


* The paper is under the review.

If you use this source code, please cite the paper as follows. 
~~~
Will be updated
~~~


## Abstract.
In skeleton-based action recognition, graph convolutional networks (GCNs), which model human body
skeletons using graphical components such as nodes and connections, have achieved remarkable performance recently. However, current state-of-the-art methods for skeleton-based action recognition usually work on the assumption that the completely observed skeletons will be provided. This may be problematic to apply this assumption in real scenarios since there is always a possibility that captured skeletons are incomplete or noisy. In this work, we propose a skeleton-based action recognition method which is robust to noise information of given skeleton features. The key insight of our approach is to train a model by maximizing the mutual information between normal and noisy skeletons using a predictive coding manner. We have conducted comprehensive experiments about skeleton-based action recognition with defected skeletons using NTU-RGB+D and Kinetics-Skeleton datasets. The experimental results demonstrate that our approach achieves outstanding performance when skeleton samples are noised compared with existing state-of-the-art methods.

## File configuration
.<br>
├── frequency_discriminator.pkl<br>
├── image_discriminator.pkl<br>
├── inception_score_graph.txt<br>
├── logs<br>
│   └── events.out.tfevents.1578130324.neumann-System-Product-Name<br>
├── main.py<br>
├── model<br>
│   ├── aiftn.py<br>
│   └── __pycache__<br>
│       └── aiftn.cpython-36.pyc<br>
├── negative_generator.pkl<br>
├── positive_generator.pkl<br>
├── README.md<br>
├── src<br>
│   ├── config.py<br>
│   ├── dataset.py<br>
│   ├── __init__.py<br>
│   ├── __pycache__<br>
│   │   ├── config.cpython-36.pyc<br>
│   │   ├── dataset.cpython-36.pyc<br>
│   │   ├── __init__.cpython-36.pyc<br>
│   │   ├── tensorboard_logger.cpython-36.pyc<br>
│   │   └── utils.cpython-36.pyc<br>
│   ├── tensorboard_logger.py<br>
│   └── utils.py<br>
├── tb_log.txt<br>
├── test.png<br>
├── _t_main.py<br>
└── training_result_vis<br>


## How to train
~~~
python main.py
~~~
