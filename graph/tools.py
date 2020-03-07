import numpy as np


def edge2mat(link, num_node, exclude, is_drop=False):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1

    if len(exclude):
        if is_drop:
            A = drop_jonit(A, exclude)

    return A


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def get_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node, [])
    In = normalize_digraph(edge2mat(inward, num_node, []))
    Out = normalize_digraph(edge2mat(outward, num_node, []))
    A = np.stack((I, In, Out))
    return A


def get_spatial_graph_pre(num_node, self_link, inward, outward, exclude):
    I = edge2mat(self_link, num_node, exclude, True)
    In = normalize_digraph(edge2mat(inward, num_node, exclude, True))
    Out = normalize_digraph(edge2mat(outward, num_node, exclude, True))
    A = np.stack((I, In, Out))
    return A


def get_spatial_graph_post(num_node, self_link, inward, outward, exclude):
    I = edge2mat(self_link, num_node, [], True)
    In = normalize_digraph(edge2mat(inward, num_node, [], True))
    Out = normalize_digraph(edge2mat(outward, num_node, [], True))

    A = np.stack((drop_jonit(I, exclude), drop_jonit(In, exclude), drop_jonit(Out, exclude)))
    return A


def drop_jonit(B, exclude):
    AA = np.delete(B, exclude, 1)
    AA = np.delete(AA, exclude, 0)
    return AA
