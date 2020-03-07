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
    test_batch_size = 16




def run_test2():
    PY = 'main2.py'

    drop_num_list = [1, 3, 5,7, 10]
    
    ki,cs,cv = 'kinetics-skeleton', 'nturgbd-cross-subject', 'nturgbd-cross-view'
    agcn, aagcn,magcn = 'model.agcn.Model', 'model.aagcn.Model', 'model.magcn.Model'

    paramas = [
        #( ki,'ki_agcn_joint_normal-1ChaosFix-64-61974.pt', agcn, 'kinetics/agcn_test_joint')
        # ( ki,'ki_aagcn_joint_warm_normal0ChaosFix-64-123948.pt', aagcn, 'kinetics/aagcn_test_joint')
        ( cv,'ntu_cv_joint_mutual5dynamic_ce-main-39-49392.pt',          'ntu_cv_joint_mutual5dynamic_ce-regression-39-49392.pt',           'ntu_cv_joint_mutual5dynamic_ce',          magcn)
        #( cv,'ntu_cv_agcn_joint_mutual5dynamic-main-49-61152.pt', 'ntu_cv_agcn_joint_mutual5dynamic-regression-49-61152.pt',  'ntu_cv_agcn_joint_mutual5dynamic',   magcn)



        # ('ntu_cv_agcn_joint_speical1ChaosFix-49-23970.pt', 'normal', 1, 'Chaos', 'Fix')
    ]
    
    for param in paramas:
        dataset, weight_main,weight_ar, work_dir,model = param
       
        default_yaml = f'config/{dataset}/test_joint.yaml'
        tmp_yaml = f'config/{dataset}/test_joint_tmp.yaml'
        arg = yaml.safe_load(open(default_yaml, 'r'))    
        
        #-----------------------------------------------> Common
        
        arg['modf'] = 'mutual'
        arg['test_feeder_args']['ntype'] = 'Choas'
        arg['test_feeder_args']['ttype'] = 'Fix'
        arg['test_batch_size'] = test_batch_size
        arg['device'] = device
        arg['weights'] = f'./runs/weights/{weight_main}'
        arg['ar_weights'] = f'./runs/weights/{weight_ar}'

        arg['model'] = model
        arg['work_dir'] = f'./work_dir/{work_dir}'
        #----------------------------------------------> normal only
        arg['test_feeder_args']['modf'] = 'normal'
        arg['test_feeder_args']['drop_num'] = 0
        yaml.dump(arg, open(tmp_yaml, 'w'))
        command = f'{EXE} {PY} --config {tmp_yaml} '
        os.system(command)
        #quit()
        
        #----------------------------------------------> noise only
        arg['test_feeder_args']['modf'] ='noise'
        
        for drop_num in drop_num_list:
            arg['test_feeder_args']['drop_num'] = drop_num
            for i in range(10):
                yaml.dump(arg, open(tmp_yaml, 'w'))
                command = f'{EXE} {PY} --config {tmp_yaml} '
                os.system(command)
                quit()



run_test2()
# run_test_knee()
#run_bone()
