import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
import sys
import time
import random
import platform

sys.path.extend(['../'])
from feeders import tools


class Feeder(Dataset):
    def __init__(self, data_path, label_path, modf, drop_num, ntype, ttype,
                 random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=True):
        """
        
        :param data_path: 
        :param label_path: 
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move: 
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        """

        """
        ttype determines whether each frame has differnt noised joint or notZipFile The class for reading and writing ZIP files.  See section 
        
        CEGEN-JM, however, determines how many random joints are genereated. Stiatic|Dynmaic
        
        """

        self.debug = debug
        self.drop_num = drop_num
        self.modf = modf
        self.ntype = ntype # noise type, choas|deviation
        self.ttype = ttype # different noise on each frame
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization  # False
        self.use_mmap = use_mmap  # Ture
        self.load_data()

        print('Feeder, nomralization' , normalization, ', use_mmap ', use_mmap)
        print('normalization', self.normalization, 'random_shift', self.random_shift, 'random_choose', self.random_choose, 'window_size' , self.window_size, 'random_move', self.random_move)

        if normalization:
            self.get_mean_map()

    def drop_joint(self, data):  # NEW
        """

        :param data: shape = B, C, T, J ,M
        :return:
        """

        _data = np.transpose(data, (3, 0, 1, 2, 4))  # J B C T M
        _data = np.delete(_data, self.drop_list, axis=0)
        _data = np.transpose(_data, (1, 2, 3, 0, 4))

        return _data

    def noise_joint(self, data, idx=0, log=False):
        # np.random.seed()
        work_dir = '/home/peter/workspace/projects/xiah/2sagcn/work_dir/ntu/xview/agcn_joint'
        # drop_list = self.get_drop_joint_list(self.drop_num)
        # drop_list = np.random.randint(0, high=25, size=self.drop_num)
        drop_list = np.random.choice(25, self.drop_num, replace=False)
        # drop_list = [14]

        if log:
            if self.data.shape[0] > 20000:
                msg = f'index {idx},  trainset drop num  {self.drop_num} drop joints: {drop_list}'
            else:
                msg = f'index {idx},  valset drop num  {self.drop_num} drop joints: {drop_list}'
            print(msg)
            # print(f'before: \n{data[:, 0, drop_list, 0]}')

            with open(f'{work_dir}/{platform.node()}_log.txt', 'a') as f:
                print(msg, file=f)
                # print(f'before: \n{data[:,0,drop_list,0]}', file=f)

        for jidx in drop_list:
            for t in range(300):
                for m in range(2):
                    person = data[:, t, :, m]
                    if person.sum(-1).sum(-1) == 0:
                        continue
                    newval = self.get_random(person)
                    data[:, t, jidx, m] = newval

        # if log:
        #     print(f'after:\n{data[:, 0, drop_list, 0]}')
        # with open(f'{work_dir}/{platform.node()}_log.txt', 'a') as f:
        # print(f'after: \n{data[:,0,drop_list,0]}', file=f)

        return data

    def tmp_noise_knee_test(self, data):

        drop_list = [17]
        for t in range(300):
            for m in range(2):
                person = data[:, t, :, m]
                if person.sum(-1).sum(-1) == 0:
                    continue
                newvals = self.get_random_vals(person, self.drop_num)

                if self.ntype == 'Dev':  # deviation
                    data[:, t, drop_list, m] = data[:, t, drop_list, m] + newvals
                else:  # chaos
                    data[:, t, drop_list, m] = newvals

        return data

    def tmp_noise_shld_train(self, data):

        drop_list = [8]
        for t in range(300):
            for m in range(2):
                person = data[:, t, :, m]
                if person.sum(-1).sum(-1) == 0:
                    continue
                newvals = self.get_random_vals(person, self.drop_num)

                if self.ntype == 'Dev':  # deviation
                    data[:, t, drop_list, m] = data[:, t, drop_list, m] + newvals
                else:  # chaos
                    data[:, t, drop_list, m] = newvals

        return data

    def noise_joint_all(self, data):
        # np.random.seed()

        for t in range(300):
            for m in range(2):
                person = data[:, t, :, m]
                if person.sum(-1).sum(-1) == 0:
                    continue
                newvals = self.get_random_vals(person, self.drop_num)

                drop_list = np.random.choice(25, self.drop_num, replace=False)

                if self.ntype == 'Dev':  # deviation
                    data[:, t, drop_list, m] = data[:, t, drop_list, m] + newvals
                else:  # chaos
                    data[:, t, drop_list, m] = newvals

        return data

    def noise_joint_static(self, data, noise_num):
        # np.random.seed()
        C,T,J,M = data.shape

        drop_list = np.random.choice(J, noise_num, replace=False)

        for t in range(300):
            for m in range(2):
                person = data[:, t, :, m]
                if person.sum(-1).sum(-1) == 0:
                    continue
                newvals = self.get_random_vals(person, noise_num)
                data[:, t, drop_list, m] = newvals

        return data
    
    


    def noise_joint_fix(self, data, idx=0, log=False):
        # np.random.seed()
        C,T,J,M = data.shape

        drop_list = np.random.choice(J, self.drop_num, replace=False)

        if log:
            msg = f'index {idx}, drop num  {self.drop_num} drop joints: {drop_list}'
            print(msg)
            print('before', data[:, 0, drop_list[0], 0])

        for t in range(300):
            for m in range(2):
                person = data[:, t, :, m]
                if person.sum(-1).sum(-1) == 0:
                    continue
                newvals = self.get_random_vals(person, self.drop_num)

                if self.ntype == 'Dev':  # deviation
                    data[:, t, drop_list, m] = data[:, t, drop_list, m] + newvals
                else:  # chaos
                    data[:, t, drop_list, m] = newvals

        if log:
            print('after', data[:, 0, drop_list[0], 0])

        return data

    def fill_zero(self, data):
        print('fill zero')
        _data = np.array(data)
        _data[:, :, :, self.drop_list, :] = 0
        return _data

    def load_data(self):
        # data: N C V T M
        try:
            with open(self.label_path) as f:
                self.sample_name, self.label = pickle.load(f)
        except:
            # for pickle file from python2
            with open(self.label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f, encoding='latin1')

        # load data
        if self.use_mmap:
            self.data = np.load(self.data_path, mmap_mode='r')  # B, C, T, J ,M
        else:
            self.data = np.load(self.data_path)

        if self.debug:
            self.label = self.label[0:5]
            self.data = self.data[0:5]
            self.sample_name = self.sample_name[0:5]

        print('load_data, data len ', len(self.data), self.data.shape)

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def get_random_vals(self, person, drop_num):
        nx = np.random.uniform(np.min(person[0, :]), np.max(person[0, :]), size=(1, drop_num))
        ny = np.random.uniform(np.min(person[1, :]), np.max(person[1, :]), size=(1, drop_num))
        nz = np.random.uniform(np.min(person[2, :]), np.max(person[2, :]), size=(1, drop_num))

        return np.vstack([nx, ny, nz])
        # return [nx, ny, nz]

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        """
        self.data is fixed data.
        each epoch, the sequence of index shuffled.
        ex) epoch0 --> [ 1 3 2 4 0]
        ex) epoch1 --> [ 0 3 4 2 1]


        :param index:
        :return:
        """
        _data_numpy = self.data[index]  # sample index.
        label = self.label[index]
        data_numpy = np.array(_data_numpy)  # 3, 300, 25, 2

        if self.modf == 'knee':
            data_numpy = self.tmp_noise_knee_test(data_numpy)
        if self.modf == 'shld':
            data_numpy = self.tmp_noise_shld_train(data_numpy)

        if self.modf == 'noise':
            if self.ttype == 'All':  # all
                data_numpy = self.noise_joint_all(data_numpy)
            else:  # Fixed
                data_numpy = self.noise_joint_fix(data_numpy)

        if self.modf=='mutual':
            if self.ttype =='static':
                __data_numpy = self.noise_joint_static(data_numpy, self.drop_num)
            else:
                dynamic_drop_num = np.random.randint(1, self.drop_num+1)
                __data_numpy = self.noise_joint_static(data_numpy, dynamic_drop_num)
                
            data_numpy= np.array([_data_numpy, __data_numpy])

        if self.normalization:
            data_numpy = (data_numpy - self.mean_map) / self.std_map
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)
        # print(f'getitem {index} after', data_numpy[:,0,14,0 ])

        return data_numpy, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def test(data_path, label_path, vid=None, graph=None, is_3d=False):
    '''
    vis the samples using matplotlib
    :param data_path: 
    :param label_path: 
    :param vid: the id of sample
    :param graph: 
    :param is_3d: when vis NTU, set it True
    :return: 
    '''
    import matplotlib.pyplot as plt
    loader = torch.utils.data.DataLoader(
        dataset=Feeder(data_path, label_path),
        batch_size=64,
        shuffle=False,
        num_workers=2)

    if vid is not None:
        sample_name = loader.dataset.sample_name
        sample_id = [name.split('.')[0] for name in sample_name]
        index = sample_id.index(vid)
        data, label, index = loader.dataset[index]
        data = data.reshape((1,) + data.shape)

        # for batch_idx, (data, label) in enumerate(loader):
        N, C, T, V, M = data.shape

        plt.ion()
        fig = plt.figure()
        if is_3d:
            from mpl_toolkits.mplot3d import Axes3D
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)

        if graph is None:
            p_type = ['b.', 'g.', 'r.', 'c.', 'm.', 'y.', 'k.', 'k.', 'k.', 'k.']
            pose = [
                ax.plot(np.zeros(V), np.zeros(V), p_type[m])[0] for m in range(M)
            ]
            ax.axis([-1, 1, -1, 1])
            for t in range(T):
                for m in range(M):
                    pose[m].set_xdata(data[0, 0, t, :, m])
                    pose[m].set_ydata(data[0, 1, t, :, m])
                fig.canvas.draw()
                plt.pause(0.001)
        else:
            p_type = ['b-', 'g-', 'r-', 'c-', 'm-', 'y-', 'k-', 'k-', 'k-', 'k-']
            import sys
            from os import path
            sys.path.append(
                path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
            G = import_class(graph)()
            edge = G.inward
            pose = []
            for m in range(M):
                a = []
                for i in range(len(edge)):
                    if is_3d:
                        a.append(ax.plot(np.zeros(3), np.zeros(3), p_type[m])[0])
                    else:
                        a.append(ax.plot(np.zeros(2), np.zeros(2), p_type[m])[0])
                pose.append(a)
            ax.axis([-1, 1, -1, 1])
            if is_3d:
                ax.set_zlim3d(-1, 1)
            for t in range(T):
                for m in range(M):
                    for i, (v1, v2) in enumerate(edge):
                        x1 = data[0, :2, t, v1, m]
                        x2 = data[0, :2, t, v2, m]
                        if (x1.sum() != 0 and x2.sum() != 0) or v1 == 1 or v2 == 1:
                            pose[m][i].set_xdata(data[0, 0, t, [v1, v2], m])
                            pose[m][i].set_ydata(data[0, 1, t, [v1, v2], m])
                            if is_3d:
                                pose[m][i].set_3d_properties(data[0, 2, t, [v1, v2], m])
                fig.canvas.draw()
                # plt.savefig('/home/lshi/Desktop/skeleton_sequence/' + str(t) + '.jpg')
                plt.pause(0.01)


if __name__ == '__main__':
    import os

    os.environ['DISPLAY'] = 'localhost:10.0'
    data_path = "../data/ntu/xview/val_data_joint.npy"
    label_path = "../data/ntu/xview/val_label.pkl"
    graph = 'graph.ntu_rgb_d.Graph'
    test(data_path, label_path, vid='S004C001P003R001A032', graph=graph, is_3d=True)
    # data_path = "../data/kinetics/val_data.npy"
    # label_path = "../data/kinetics/val_label.pkl"
    # graph = 'graph.Kinetics'
    # test(data_path, label_path, vid='UOD7oll3Kqo', graph=graph)
