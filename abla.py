import random
import yaml
import os
import json
import platform

# testing
# why? wowowow
EXE = '/home/peter/anaconda3/envs/2sagcn/bin/python'  # OBAMA
device = [0, 1, 2, 3]
test_batch_size = 512


def run_train():
    cs, cv = 'nturgbd-cross-subject', 'nturgbd-cross-view'

    xsub, xview = 'ntu/xsub', 'ntu/xview'
    txsub, txview = 'cs', 'cv'
    ce, total = 'ce', 'total'

    pp_list = [
        (cv, 'train_joint_mutual.yaml', xview, txview),
        (cs, 'train_joint_mutual.yaml', xsub, txsub)
    ]


    main_list = [ ce, total]
    drop_num_list = [1, 3, 5]


    for pp in pp_list:
        protocol, yaml_name, dataset,tag = pp
        path_yaml_default = f'config/{protocol}/{yaml_name}'
        arg_default= yaml.safe_load(open(path_yaml_default))

        # -----------------------------------------------> Common


        for main_tag in main_list:
            PY = f'main_{main_tag}.py'


            for drop_num in drop_num_list:
                arg_default['train_feeder_args']['drop_num'] = drop_num
                arg_default['num_epoch']= 40
                arg_default['work_dir'] = f'./work_dir/{dataset}/ablation'
                arg_default['model_saved_name'] = f'./runs/ntu_{tag}_joint_mutual{drop_num}dynamic_{main_tag}'
                path_yaml_tmp = f'config/{protocol}/tmp/dynamic_mutual_{main_tag}_{str(drop_num).zfill(2)}_{yaml_name}'
                yaml.dump(arg_default, open(path_yaml_tmp, 'w'))

                #command = f'{EXE} {PY} --config {path_yaml_tmp} '
                #print('Running', PY,  path_yaml_tmp)
                #os.system(command)


run_train()
