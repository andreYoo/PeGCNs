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
├── abla.py<br>
├── arun.sh<br>
├── config<br>
│   ├── config.txt<br>
│   ├── kinetics-skeleton<br>
│   │   ├── test_bone.yaml<br>
│   │   ├── test_joint_mutual_none.yaml<br>
│   │   ├── test_joint_mutual_tmp.yaml<br>
│   │   ├── test_joint_mutual.yaml<br>
│   │   ├── test_joint.yaml<br>
│   │   ├── train_bone.yaml<br>
│   │   ├── train_joint_mutual_v2.yaml<br>
│   │   ├── train_joint_mutual.yaml<br>
│   │   └── train_joint.yaml<br>
│   ├── nturgbd-cross-subject<br>
│   │   ├── test_bone.yaml<br>
│   │   ├── test_joint_mutual_tmp.yaml<br>
│   │   ├── test_joint_mutual.yaml<br>
│   │   ├── test_joint.yaml<br>
│   │   ├── train_bone.yaml<br>
│   │   ├── train_joint_aagcn.yaml<br>
│   │   ├── train_joint_mutual.yaml<br>
│   │   ├── train_joint_tmp.yaml<br>
│   │   └── train_joint.yaml<br>
│   └── nturgbd-cross-view<br>
│       ├── test_bone.yaml<br>
│       ├── test_joint_mutual.yaml<br>
│       ├── test_joint_tmp.yaml<br>
│       ├── test_joint.yaml<br>
│       ├── tmp_train_joint_mutual.yaml<br>
│       ├── train_bone.yaml<br>
│       ├── train_joint_mutual.yaml<br>
│       └── train_joint.yaml<br>
├── data<br>
│   ├── kinetics -> /home/peter/workspace/dataset/Kinetics/kinetics-skeleton/<br>
│   └── ntu -> /home/peter/workspace/dataset/NTURGB+D/NTURGBD-2s-AGCN/ntu/<br>
├── data_gen<br>
│   ├── gen_bone_data.py<br>
│   ├── gen_motion_data.py<br>
│   ├── __init__.py<br>
│   ├── kinetics_gendata.py<br>
│   ├── merge_joint_bone_data.py<br>
│   ├── ntu_gendata.py<br>
│   ├── preprocess.py<br>
│   ├── __pycache__<br>
│   │   ├── __init__.cpython-36.pyc<br>
│   │   ├── preprocess.cpython-36.pyc<br>
│   │   └── rotation.cpython-36.pyc<br>
│   └── rotation.py<br>
├── ensemble.py<br>
├── feeders<br>
│   ├── feeder.py<br>
│   ├── __init__.py<br>
│   ├── __pycache__
│   │   ├── feeder.cpython-36.pyc<br>
│   │   ├── __init__.cpython-36.pyc<br>
│   │   └── tools.cpython-36.pyc<br>
│   └── tools.py<br>
├── graph<br>
│   ├── __init__.py<br>
│   ├── kinetics.py<br>
│   ├── ntu_rgb_d.py<br>
│   ├── __pycache__
│   │   ├── __init__.cpython-36.pyc<br>
│   │   ├── kinetics.cpython-36.pyc<br>
│   │   ├── ntu_rgb_d.cpython-36.pyc<br>
│   │   └── tools.cpython-36.pyc<br>
│   └── tools.py<br>
├── inference_ab.py<br>
├── inference.py<br>
├── main2.py<br>
├── main3.py<br>
├── main_ce.py<br>
├── main.py<br>
├── main_total.py<br>
├── model<br>
│   ├── aagcn.py<br>
│   ├── agcn.py<br>
│   ├── Autoregression.py<br>
│   ├── __init__.py<br>
│   ├── magcn.py<br>
│   └── __pycache__<br>
│       ├── aagcn.cpython-36.pyc<br>
│       ├── agcn.cpython-36.pyc<br>
│       ├── Autoregression.cpython-36.pyc<br>
│       ├── __init__.cpython-36.pyc<br>
│       └── magcn.cpython-36.pyc<br>
├── nturgbd_raw<br>
│   └── samples_with_missing_skeletons.txt<br>
├── README.md<br>
├── run.py<br>
├── runs<br>
├── run.sh<br>
├── test.py<br>
├── tmp2.sh<br>
└── work_dirs<br>




## How to train
~~~
python main.py
~~~



## How to monitoring
~~~
python main.py
~~~
