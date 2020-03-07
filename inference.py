import random
import yaml
import os
import json
import platform

# testing
# why? wowowow
if platform.node() == 'obama':
    EXE = '/home/peter/anaconda3/envs/2sagcn/bin/python'  # OBAMA
    device = [0, 1, 2, 3]
    test_batch_size = 512
else:
    EXE = '/home/peter/anaconda3/envs/agcn/bin/python'  # PUMA
    device = [0]
    test_batch_size = 10


def run_bone():
    PY = 'main.py'
    ki, cs, cv = 'kinetics-skeleton', 'nturgbd-cross-subject', 'nturgbd-cross-view'
    agcn, aagcn = 'model.agcn.Model', 'model.aagcn.Model'

    default_joint_yaml = f'config/{cv}/test_joint.yaml'
    default_bone_yaml = f'config/{cv}/test_bone.yaml'

    tmp_joint_yaml = f'config/{cv}/test_joint_tmp.yaml'
    tmp_bone_yaml = f'config/{cv}/test_bone_tmp.yaml'

    arg_joint = yaml.safe_load(open(default_joint_yaml, 'r'))
    arg_bone = yaml.safe_load(open(default_bone_yaml, 'r'))

    # -----------------------------------------------> Common

    device = [0]
    batch1 = 4
    batch2 = 4
    # noise_mode = False
    noise_mode = True
    if noise_mode:
        arg_joint['test_feeder_args']['ntype'] = 'Choas'
        arg_joint['test_feeder_args']['ttype'] = 'Fix'
        arg_joint['test_feeder_args']['modf'] = 'noise'
        # arg_joint['test_batch_size'] = batch1
        # arg_joint['device'] = device

        arg_bone['test_feeder_args']['ntype'] = 'Choas'
        arg_bone['test_feeder_args']['ttype'] = 'Fix'
        arg_bone['test_feeder_args']['modf'] = 'noise'
        # arg_bone['test_batch_size'] = batch2
        # arg_bone['device'] = device

        # drop_num_list = [1, 3, 5, 10]
        repeat = 10
        drop_num_list = [10]

    else:
        arg_joint['test_feeder_args']['modf'] = 'normal'
        arg_bone['test_feeder_args']['modf'] = 'normal'
        drop_num_list = [0]
        repeat = 1

    for drop_num in drop_num_list:

        arg_joint['test_feeder_args']['drop_num'] = drop_num
        yaml.dump(arg_joint, open(tmp_joint_yaml, 'w'))

        arg_bone['test_feeder_args']['drop_num'] = drop_num
        yaml.dump(arg_bone, open(tmp_bone_yaml, 'w'))

        # for r in range(repeat):
        for r in range(5, 10):

            # 1. test and save noisy joint data
            command = f'{EXE} {PY} --config {tmp_joint_yaml} '
            print(tmp_joint_yaml)
            os.system(command)

            # 2. genrate bone data
            command = f'{EXE} gen_bone_data.py'
            os.chdir('./data_gen')
            print(command)
            os.system(command)
            os.chdir('..')

            # 3. test bone data
            command = f'{EXE} {PY} --config {tmp_bone_yaml} '
            print(command)
            os.system(command)

            # 4. ensemble
            command = f'{EXE} ensemble.py'
            print(command)
            os.system(command)

            # 5. save file
            score_file = 'epoch1_test_score.pkl'
            for ttype in ['joint', 'bone']:
                path = f'./work_dir/ntu/xview/agcn_test_{ttype}'
                new_score_file = f'epoch1_test_score_{ttype}_{(str(drop_num).zfill(2))}_{str(r).zfill(2)}.pkl'

                frompath = os.path.join(path, score_file)
                topath = os.path.join('./work_dir/ntu/xview/test_normal', new_score_file)
                c1 = f'mv {frompath} {topath}'
                print(c1)
                os.system(c1)

            # 6. remove
            os.system('rm ./data/ntu/xview/noisy_data_joint.npy')
            os.system('rm ./data/ntu/xview/noisy_data_bone.npy')

            # quit()


def run_test_knee():
    PY = 'main.py'
    ki, cs, cv = 'kinetics-skeleton', 'nturgbd-cross-subject', 'nturgbd-cross-view'
    agcn, aagcn = 'model.agcn.Model', 'model.aagcn.Model'

    param = (cv, 'ntu_cv_agcn_joint-49-47050.pt', agcn, 'ntu/xview/agcn_test_joint_tmp')

    dataset, weightfile, model, work_dir = param

    default_yaml = f'config/{dataset}/test_joint.yaml'
    tmp_yaml = f'config/{dataset}/test_joint_tmp.yaml'
    arg = yaml.safe_load(open(default_yaml, 'r'))

    # -----------------------------------------------> Common

    arg['test_feeder_args']['ntype'] = 'Choas'
    arg['test_feeder_args']['ttype'] = 'Fix'
    arg['test_batch_size'] = 10
    arg['device'] = device
    arg['weights'] = f'./runs/weights/{weightfile}'
    arg['model'] = model
    arg['work_dir'] = f'./work_dir/{work_dir}'
    # ----------------------------------------------> normal only
    arg['test_feeder_args']['modf'] = 'normal'
    yaml.dump(arg, open(tmp_yaml, 'w'))

    for drop_num in range(25):
        arg['test_feeder_args']['drop_num'] = 99
        yaml.dump(arg, open(tmp_yaml, 'w'))
        command = f'{EXE} {PY} --config {tmp_yaml} '
        os.system(command)
        quit()


def run_test2():
    PY = 'main2.py'

    drop_num_list = [1, 3, 5, 10]
    modf_list = ['drop', 'noise', 'zero']
    ntype_list = ['Dev', 'Chaos']
    ttype_list = ['All', 'Fix']

    ki, cs, cv = 'kinetics-skeleton', 'nturgbd-cross-subject', 'nturgbd-cross-view'
    agcn, aagcn, magcn = 'model.agcn.Model', 'model.aagcn.Model', 'model.magcn.Model'
    paramas = [
        # ('ntu_cv_agcn_joint_noise2DevAll-39-18800.pt', 'noise', 2, 'Dev', 'All'),



        # ('ntu_cv_agcn_joint_speical1ChaosFix-49-23970.pt', 'normal', 1, 'Chaos', 'Fix')
        (ki, 512),
        (cs, 512),
        (cv, 512),
    ]


    # for param in paramas:
    for param in paramas:
        dataset, batch_size = param
        default_yaml = f'config/{dataset}/test_joint_mutual.yaml'
        tmp_yaml = f'config/{dataset}/test_joint_mutual_tmp.yaml'
        arg = yaml.safe_load(open(default_yaml, 'r'))

        # -----------------------------------------------> Common

        arg['test_batch_size'] = batch_size
        # ----------------------------------------------> normal only
        arg['test_feeder_args']['modf'] = 'normal'
        arg['test_feeder_args']['drop_num'] = 0
        yaml.dump(arg, open(tmp_yaml, 'w'))
        command = f'{EXE} {PY} --config {tmp_yaml} '
        os.system(command)

        # ----------------------------------------------> noise only
        arg['test_feeder_args']['modf'] = 'noise'

        # for drop_num in [1]:
        for drop_num in drop_num_list:
            arg['test_feeder_args']['drop_num'] = drop_num
            for i in range(10):
            # for i in range(1):
                print('drop num:' , drop_num, 'iteration: ', i )
                yaml.dump(arg, open(tmp_yaml, 'w'))
                command = f'{EXE} {PY} --config {tmp_yaml} '
                os.system(command)
                # quit()


run_test2()
# run_test_knee()
# run_bone()
