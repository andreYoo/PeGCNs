import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import yaml
import random


# print(type(aa))


def test_permuate():
    b1 = np.array([x for x in range(1, 25)])
    c = b1.reshape((2, 3, 4))
    print('original', c.shape)
    print(c)

    cc = c.transpose((1, 0, 2))  # 2,3,4
    print('transposed', cc.shape)
    print(cc)

    dd = np.delete(cc, [0, 1], 0)
    dd = dd.transpose((1, 0, 2))
    print('removed', dd.shape)
    print(dd)


def test_graph():
    num_node = 25

    self_link = [(i, i) for i in range(num_node)]

    A = np.ones((6, 6))
    for i in range(6):
        A[i, i] = (i + 1) * 3

    exclude = [2, 4]

    def drop_joint(AA, exclude):
        A = np.array(AA)
        A = np.delete(A, exclude, 0)
        A = np.delete(A, exclude, 1)
        return A

    def normalize_digraph(A):  # 除以每列的和
        Dl = np.sum(A, 0)
        h, w = A.shape
        Dn = np.zeros((w, w))
        for i in range(w):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i] ** (-1)
        AD = np.dot(A, Dn)
        return AD

    print('original')
    print(A)
    print('drop')
    print(drop_joint(A, exclude))


def fill_test():
    a = np.array([x for x in range(0, 16)]).reshape(4, 4)

    ex = [0, 3]

    print(a)
    a[:, ex] = -1
    print(a)
    a[ex, :] = -2

    print(a)


def get_random(person):
    nx = np.random.uniform(np.min(person[0, :]), np.max(person[0, :]))
    ny = np.random.uniform(np.min(person[1, :]), np.max(person[1, :]))
    nz = np.random.uniform(np.min(person[2, :]), np.max(person[2, :]))
    return [nx,ny,nz]


def random_test():
    import time
    a = './data/ntu/xview/val_data_joint.npy'
    x = np.load(a)  # b, c, t, j, m

    from tqdm import tqdm
    jidx = 14
    # name_desc = tqdm(range(x.shape[0]))
    ckpt = time.time()
    count =0
    for clip in x:
        # name_desc.update(1)
        for t in range(300):
            for m in range(2):
                person = clip[:, t, :, m] # 3, 300, 25, 2

                if person.sum(-1).sum(-1) == 0 :
                    pass
                    # print(t, m , 'is zero')
                    # print(person)



                # print(person[0,:])
                # print(person[1,:])
                # print(person[2,:])
                print('min:', np.min(person[0, :]), 'max', np.max(person[0, :]),'mean', np.mean(person[0, :]),'std', np.std(person[0, :]))
                print('min:', np.min(person[1, :]), 'max', np.max(person[1, :]), 'mean', np.mean(person[1, :]), 'std',
                      np.std(person[1, :]))
                print('min:', np.min(person[2, :]), 'max', np.max(person[2, :]), 'mean', np.mean(person[2, :]), 'std',
                      np.std(person[2, :]))


                # print(np.min(person[1, :]), np.max(person[1, :]))
                # print(np.min(person[2, :]), np.max(person[2, :]))


                # print('original', clip[:,t,14,m])
                # a = get_random(person)
                # person[:,14] = a
                # print('new val' , a)
                # print('clip', clip[:,t,14,m])
                #
                # print(get_random(person))
                # print(get_random(person))
                # print(get_random(person))
                # print(get_random(person))
        quit()


            

    print(f'{count} items, took {time.time()-ckpt} s')




def dodo(x, jidx, output):
    print('starts from :', jidx, ' size of ', x.shape)
    for clip in x:
        for t in range(300):
            for m in range(2):
                person1 = clip[:, t, :, m]
                person1[:,jidx] =  get_random(person1)

    output.put(x)



def test_mp():
    x = np.ones((10, 3, 2))
    x = np.random.rand(190,3,300,25,2)
    # import multiprocessing
    # from functools import partial
    from multiprocessing import Process, Queue
    # a = './data/ntu/xview/val_data_joint.npy'
    # x = np.load(a)  # b, c, t, j, m

    batch= 15
    num_q = x.shape[0] // batch
    num_q +=1
    for xi in range(num_q):
        print( xi * batch, ' ~ ' ,  (xi + 1) * batch)

    qlist =[ Queue() for x in range(num_q)]
    proc = [Process(target=dodo, args=(  x[xi*batch: (xi+1)*batch],xi,qlist[xi]) ) for xi in range(num_q)]
    for p in proc:
        p.start()
    for p in proc:
        p.join()

    olist =[ q.get() for q in qlist]
    result = np.vstack(olist)
    print(result.shape)






def test_cuda():
    # a = './data/ntu/xview/val_data_joint.npy'
    # x = np.load(a)  # b, c, t, j, m

    x = np.random.rand(4,3,25)
    _data = torch.from_numpy(x.astype(np.float32))
    data = _data
    data.to('cuda')



    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.conv1 = nn.Conv2d(1, 20, 5)
            self.conv2 = nn.Conv2d(20, 20, 5)
            self.idx = 14

        def get_rand(self, person):

            x = torch.FloatTensor(1).uniform_(torch.min(person[0, :]), torch.max(person[0, :]))
            y = torch.FloatTensor(1).uniform_(torch.min(person[1, :]), torch.max(person[1, :]))
            z = torch.FloatTensor(1).uniform_(torch.min(person[2, :]), torch.max(person[2, :]))

            return torch.from_numpy(np.array([x,y,z]).astype(np.float32))


        def forward(self, x):
            x[:,self.idx] = self.get_rand(x)

            return x

    model = Model()
    data = Variable(data.float().to('cuda'), requires_grad=False)
    out = model(data)


    # data = Variable(
    #     # tmp_data.float().cuda(self.output_device),
    #     data.float().cuda([0]),
    #     requires_grad=False,
    #     volatile=True)

    # print(data.cuda.is_available())


def uniform_test():


    def get_rand():
        np.random.seed()

        a = np.array([ -0.22960308194160461 ,  0.22960308194160461, -0.156345634563456])
        # mn= -0.22960308194160461
        # mx= 0.22140608727931976
        mn = np.min(a)
        mx = np.max(a)
        return [np.random.uniform(mn,mx), np.random.uniform(mn,mx)]

    for i in range(5):
        print(get_rand())
        
        

def random_test2():
    b = np.ones((3,300,25,2))

    drop_list = np.random.choice(25, 2, replace=False)


    news = []
    for i in range(3):
        drop_list = np.random.choice(25, 2, replace=False)


        a= np.random.uniform(-0.9, 1.3, size=(1,drop_num))
        print(a)
        news.append(a)
    news= np.vstack(news)
    print(news.shape)






def yaml_test():
    default_yaml = 'config/nturgbd-cross-view/train_joint_tmp.yaml'

    arg = yaml.safe_load(open(default_yaml, 'r'))

    for k in arg.keys():
        print(k, arg[k])
    # print(arg)

# yaml_test()
# test_permuate()
# test_graph()
# fill_test()
# random_test()
# uniform_test()
random_test2()
# test_cuda()
# test_mp()