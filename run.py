import random
import yaml
import os
import json
import platform

if platform.node() == 'obama':
    EXE = '/home/peter/anaconda3/envs/2sagcn/bin/python'  # OBAMA
    device = [0, 1, 2, 3]
    test_batch_size = 512
else:
    EXE = '/home/peter/anaconda3/envs/agcn/bin/python'  # PUMA
    device = [0]
    test_batch_size = 128


def run_train():
    """
    - Noise, Zero
    - Spatial
    """
    PY = 'main.py'
    default_yaml = 'config/nturgbd-cross-view/train_joint.yaml'
    tmp_yaml = 'config/nturgbd-cross-view/train_joint_tmpas.yaml'
    arg = yaml.safe_load(open(default_yaml, 'r'))

    originaml_model_name = arg['model_saved_name']

    drop_num_list = [1, 3, 5, 9]
    modf_list = ['drop', 'noise', 'zero']
    ntype_list = ['Dev', 'Chaos']
    ttype_list = ['All', 'Fix']

    nmod = 'speical'
    ntype = 'Chaos'
    ttype = 'Fix'

    for drop_num in [1]:

        for i in range(1):
            # arg['train_feeder_args']['modf'] = nmod
            # arg['test_feeder_args']['modf'] = nmod
            arg['train_feeder_args']['modf'] = 'shld'
            arg['test_feeder_args']['modf'] = 'knee'
            arg['modf'] = 'shld'

            arg['train_feeder_args']['drop_num'] = drop_num
            arg['test_feeder_args']['drop_num'] = drop_num

            arg['train_feeder_args']['ntype'] = ntype
            arg['test_feeder_args']['ntype'] = ntype

            arg['train_feeder_args']['ttype'] = ttype
            arg['test_feeder_args']['ttype'] = ttype

            arg['weights'] = f'./runs/ntu_cv_agcn_joint_speical1ChaosFix-13-06580.pt'
            arg['model_saved_name'] = originaml_model_name + f'_{nmod}{str(drop_num)}{ntype}{ttype}'

            # PUMA
            # arg['device']=[0]
            # arg['train_feeder_args']['debug'] = True
            # arg['test_feeder_args']['debug'] = True

            yaml.dump(arg, open(tmp_yaml, 'w'))
            command = f'{EXE} {PY} --config {tmp_yaml} '
            os.system(command)


def run_test2():
    PY = 'main.py'
    default_yaml = 'config/nturgbd-cross-view/test_joint.yaml'
    tmp_yaml = 'config/nturgbd-cross-view/test_joint_tmp.yaml'
    arg = yaml.safe_load(open(default_yaml, 'r'))

    weightfile = 'ntu_cv_agcn_joint_noise2DevFix-49-23970.pt'

    drop_num_list = [1, 3, 5, 9]
    modf_list = ['drop', 'noise', 'zero']
    ntype_list = ['Dev', 'Chaos']
    ttype_list = ['All', 'Fix']

    drop_num = 2
    nmod = 'normal'
    ntype = 'Dev'
    ttype = 'Fix'

    paramas = [
        # ('ntu_cv_agcn_joint_noise2DevAll-39-18800.pt', 'normal',0, 'Dev', 'Fix'),
        # ('ntu_cv_agcn_joint_noise2DevAll-39-18800.pt', 'normal', 0, 'Dev', 'All'),
        # ('ntu_cv_agcn_joint_noise2DevAll-39-18800.pt', 'noise', 2, 'Dev', 'All'),
        # ('ntu_cv_agcn_joint_noise2DevAll-39-18800.pt', 'noise', 2, 'Dev', 'All'),
        # ('ntu_cv_agcn_joint_noise2DevAll-39-18800.pt', 'noise', 2, 'Dev', 'All'),


        ('ntu_cv_agcn_joint_speical1ChaosFix-49-23970.pt', 'shld', 1, 'Chaos', 'Fix'),
        ('ntu_cv_agcn_joint_speical1ChaosFix-49-23970.pt', 'knee', 1, 'Chaos', 'Fix'),
        ('ntu_cv_agcn_joint_speical1ChaosFix-49-23970.pt', 'noise', 1, 'Chaos', 'Fix'),
        ('ntu_cv_agcn_joint_speical1ChaosFix-49-23970.pt', 'noise', 1, 'Chaos', 'Fix'),
        ('ntu_cv_agcn_joint_speical1ChaosFix-49-23970.pt', 'noise', 1, 'Chaos', 'Fix')
        # ('ntu_cv_agcn_joint_speical1ChaosFix-49-23970.pt', 'normal', 1, 'Chaos', 'Fix')

    ]

    # for i in range(3):
    for param in paramas:
        weightfile, nmod, drop_num, ntype, ttype = param

        arg['test_feeder_args']['modf'] = nmod
        arg['modf'] = nmod
        arg['test_feeder_args']['drop_num'] = drop_num
        arg['test_feeder_args']['ntype'] = ntype
        arg['test_feeder_args']['ttype'] = ttype

        arg['test_batch_size'] = test_batch_size
        arg['device'] = device
        arg['weights'] = f'./runs/{weightfile}'

        yaml.dump(arg, open(tmp_yaml, 'w'))
        command = f'{EXE} {PY} --config {tmp_yaml} '
        os.system(command)


def run_test():
    """
    - Noise, Zero
    - Spatial
    """
    # EXE = '/home/peter/anaconda3/envs/2sagcn/bin/python'  # OBAMA
    EXE = '/home/peter/anaconda3/envs/agcn/bin/python'  # PUMA
    PY = 'main.py'
    default_yaml = 'config/nturgbd-cross-view/test_joint.yaml'
    tmp_yaml = 'config/nturgbd-cross-view/test_joint_tmp.yaml'
    arg = yaml.safe_load(open(default_yaml, 'r'))
    # nmod = 'noise'
    # nmode ='zero'

    aa = [(1, 'ntu_cv_agcn_joint_tmp1-33-4116.pt'),
          (5, 'ntu_cv_agcn_joint_tmp5-33-24770.pt'),
          (2, 'ntu_cv_agcn_joint_tmp2-31-25088.pt'),
          (4, 'ntu_cv_agcn_joint_tmp4-49-23500.pt'),
          (3, 'ntu_cv_agcn_joint_tmp3-49-29400.pt'),
          (0, 'ntu_cv_agcn_joint-49-47050.pt')
          ]

    drop_list = [1, 2, 3, 444]
    # for drop_num in [1, 5, 3, 10][:2]:

    for a in aa:
        with open('{}/logtrain.txt'.format(arg['work_dir']), 'a') as f:
            print('-------------------------------', file=f)

        for i in range(2):
            print(a)
            drop_num = a[0]
            wegihtfile = a[1]

            # if drop_num > 0:
            #     nmod ='noise'
            # else:
            #     nmod ='normal'

            nmod = 'normal'
            drop_num = 0

            arg['test_feeder_args']['noise_mode'] = nmod
            arg['model_args']['graph_args']['noise_mode'] = nmod

            arg['test_feeder_args']['drop_num'] = drop_num

            arg['model_saved_name'] = arg['model_saved_name'] + f'_tmp{str(drop_num)}'
            arg['weights'] = f'./runs/{wegihtfile}'

            yaml.dump(arg, open(tmp_yaml, 'w'))
            command = f'{EXE} {PY} --config {tmp_yaml} '
            os.system(command)
            # quit()


# make_config2()
# run()
# run_train()
run_test2()
