import json
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.distributions as distributions
from daphne import daphne
# funcprimitives
from evaluation_based_sampling import evaluate_program
from tests import is_tol, run_prob_test,load_truth
# Useful functions
from utils import _hashmap, _vector, _totensor
from utils import _put, _remove, _append, _get
from utils import _squareroot, _mat_repmat, _mat_transpose


basic_ops = {'+':torch.add,
             '-':torch.sub,
             '*':torch.mul,
             '/':torch.div
}

one_ops = {'sqrt': lambda x: _squareroot(x),
           'vector': lambda x: _vector(x),
           'hash-map': lambda x: _hashmap(x),
           'first': lambda x: x[0],
           'second': lambda x: x[1],
           'last': lambda x: x[-1],
           'rest': lambda x: x[1:],
           "mat-tanh": lambda a: torch.tanh(a),
           "mat-transpose": lambda a: _mat_transpose(a)
}

two_ops={'get': lambda x, idx: _get(x, idx),
        'append': lambda x, y: _append(x, y),
        'remove': lambda x, idx: _remove(x, idx),
        ">": lambda a, b: a > b,
        "=": lambda a, b: a == b,
        ">=": lambda a, b: a >= b,
        "<=": lambda a, b: a <= b,
        "or": lambda a, b: a or b,
        "and": lambda a, b: a and b,
}

dist_ops = {"normal":lambda mu, sig: distributions.normal.Normal(loc=mu, scale=sig),
            "beta":lambda a, b: distributions.beta.Beta(concentration1=a, concentration0=b),
            "gamma": lambda concentration, rate: distributions.gamma.Gamma(concentration=concentration, rate=rate),
            "uniform": lambda low, high: distributions.uniform.Uniform(low=low, high=high),
            "exponential":lambda rate: distributions.exponential.Exponential(rate=rate),
            "discrete": lambda probs: distributions.categorical.Categorical(probs=probs),
            "dirichlet": lambda concentration: distributions.dirichlet.Dirichlet(concentration=concentration),
            "bernoulli": lambda probs: distributions.bernoulli.Bernoulli(probs=probs),
            "flip": lambda probs: distributions.bernoulli.Bernoulli(probs=probs)
}

## GRAPH Utils
def make_link(G, node1, node2):
    """
    Create a DAG
    """
    if node1 not in G:
        G[node1] = {}
    (G[node1])[node2] = 1
    if node2 not in G:
        G[node2] = {}
    (G[node2])[node1] = -1

    return G


def eval_path(path, l={}, Y={}, P={}):
    """
    Evaluate the trace
    """
    outputs = []
    sigma = {}

    # Add Y to local vars
    for y in Y.keys():
        l[y] = Y[y]

    for n in path:
        # Evaluate node
        if DEBUG:
            print('Node: ', n)

        # If observation node, return value
        if n in l.keys():
            output_ = l[n]
            outputs.append([output_])

        else:
            p = P[n] # [sample* [n, 5, [sqrt, 5]]]
            root = p[0]
            tail = p[1]
            if DEBUG:
                print('PMF for Node: ', p)
                print('Empty sample Root: ', root)
                print('Empty sample Root: ', tail)

            if root == "sample*":
                sample_eval = ["sample", tail]
                if DEBUG:
                    print('Sample AST: ', sample_eval)
                output_, sigma = evaluate_program(ast=[sample_eval], sig=sigma, l=l)
                if DEBUG:
                    print('Evaluated sample: ', output_)

            elif root == "observe*":
                try:
                    sample_eval = ["observe", tail, p[2]]
                except:
                    sample_eval = ["observe", tail]
                if DEBUG:
                    print('Sample AST: ', sample_eval)
                output_, sigma = evaluate_program(ast=[sample_eval], sig=sigma, l=l)
                if DEBUG:
                    print('Evaluated sample: ', output_)

            else:
                raise AssertionError('Unsupported operation!')

            if DEBUG:
                print('Node eval sample output: ', output_)

            # Check if not torch tensor
            if not torch.is_tensor(output_):
                if isinstance(output_, list):
                    output_ = torch.tensor(output_, dtype=torch.float32)
                else:
                    output_ = torch.tensor([output_], dtype=torch.float32)

            # Add to local var
            l[n] = output_

            # Collect
            outputs.append([output_])

    return outputs, sigma


def eval_node(node, path, G, Y, P):
    """
    Evaluate a node
    """
    path = traverse(G=G, node=node, visit={}, path=path)
    if DEBUG:
        print('Evaluated graph path: ', path)
    # List Reverse
    path.reverse()
    # Evaluate path
    output = eval_path(path, l={}, Y=Y, P=P)
    if DEBUG:
        print('Evaluated reverse graph path: ', path)
        print('Evaluated graph output: ', output)
    return output


def traverse(G, node, visit={}, path=[], include_v=True):
    """
    Traverse the DAG graph
    """
    visit[node] = True
    neighbors = G[node]
    if DEBUG:
        print('Node: ', node)
        print('visit: ', visit)
        print('Neighbours: ', neighbors)
        print('\n')

    # Path should be empty only at the first traversal node
    if path == [] and include_v:
        path.append(node)

    for n in neighbors:
        if DEBUG:
            print('Neighbor: ', n)

        if (neighbors[n] == -1):
            if n not in path:
                path.append(n)
            traverse(G, node=n, visit=visit, path=path)

        elif (neighbors[n] == 1):
            continue

        else:
            raise AssertionError('n is unreachable, something went wrong!')

    return path

#-------------------------------------HMC---------------------------------------
# Global vars
DEBUG = False # Set to true to see intermediate outputs for debugging purposes
def grad_UX(X, Y, P):
    X_, Y_ = redo_maps(X=X, Y=Y, P=P)
    Ex = 0.0
    Ey = 0.0
    for x in X_.keys():
        Ex = Ex + X_[x].log_prob(X[x])
    for y in Y_.keys():
        Ey = Ey + Y_[y].log_prob(Y[y])
    Eu = -1.0*(Ex + Ey)

    # Compute grads
    Eu.backward()

    # Collect grads
    dU = {}
    for x in X.keys():
        dU[x] = X[x].grad
    for y in Y_.keys():
        dU[y] = Y[y].grad

    return dU

def leapFrog(X, Y, P, R_0, T, eps):
    gradU = grad_UX(X=X, Y=Y, P=P)
    if DEBUG:
        print("Gradients: ", gradU)

    R_12 = {}
    for lvar in X.keys():
        R_12[lvar] = R_0[lvar] - (0.5) * eps * gradU[lvar]
    if DEBUG:
        print("R12: ", R_12)

    for t in range(T):
        for x in X.keys():
            with torch.no_grad():
                X[x] = X[x] + (eps * R_12[x])
        # Add gradient support back
        for x in X.keys():
            X[x].requires_grad=True
        gradU_t = grad_UX(X=X, Y=Y, P=P)
        for x in R_12.keys():
            R_12[x] = R_12[x] - (eps * gradU_t[x])

    X_T = {}
    for x in X.keys():
        with torch.no_grad():
            X_T[x] = X[x] + (eps * R_12[x])
    # Add gradient support back
    for x in X.keys():
        X_T[x].requires_grad=True

    R_T = {}
    dU_t = grad_UX(X=X, Y=Y, P=P)
    for x in R_12.keys():
        R_T[x] = R_12[x] - (0.5 * eps * dU_t[x])

    return X_T, R_T

def computeH(X, Y, P, R, M):
    X_, Y_ = redo_maps(X=X, Y=Y, P=P)
    with torch.no_grad():
        # Compute U
        Ex = 0.0
        Ey = 0.0
        for x in X_.keys():
            Ex = Ex + X_[x].log_prob(X[x])
        for y in Y_.keys():
            Ey = Ey + Y_[y].log_prob(Y[y])
            # Ey = Ey + torch.log(Y_[y])
        U  = -1.0*(Ex + Ey)

    # Compute K
    Rm = torch.zeros(0, dtype=torch.float32)
    for lvar in R.keys():
        xm = R[lvar]
        # Check if not torch tensor
        if not torch.is_tensor(xm):
            if isinstance(xm, list):
                xm = torch.tensor(xm, dtype=torch.float32)
            else:
                xm = torch.tensor([xm], dtype=torch.float32)
        # Check for 0 dimensional tensor
        elif xm.shape == torch.Size([]):
            xm = torch.tensor([xm.item()], dtype=torch.float32)
        try:
            Rm = torch.cat((Rm, xm))
        except:
            raise AssertionError('Cannot append the torch tensors')
    Rm = torch.unsqueeze(Rm, axis=1)
    if DEBUG:
        print('Rm: ', Rm)
        print('Rm: ', Rm.shape)
        print('RM.T: ', (torch.transpose(Rm, 0, 1)).shape)

    if len(X.keys()) == 1:
        K = Rm * Rm
        K = K/2.0
        K = -1.0 * K[0]
    else:
        #M1 = torch.inverse(M)
        #K1 = torch.matmul(torch.transpose(Rm, 0, 1), M1.type(torch.float32))
        K = torch.matmul(torch.transpose(Rm, 0, 1), Rm)
        K = K/2.0
        K = -1.0 * K[0]

    H_ = U + K

    return H_

def redo_maps(X, Y, P):
    X_map = {}
    # Setup local vars
    l = {**X}
    for lv in X.keys():
        p = P[lv]
        root = p[0]
        tail = p[1]
        if root == "sample*":
            output_, sigma = evaluate_program(ast=[tail], sig={}, l=l)
            X_map[lv] = output_
        else:
            continue
    if DEBUG:
        print('\n')
        print('Evaluted Latent dist: ', X_map)
        print('\n')
    del l

    Y_map = {}
    # Setup local vars
    with torch.no_grad():
        l = {}
        for x0 in X.keys():
            l[x0] = copy.deepcopy(X[x0])
            l[x0].requires_grad = False
        for yvar in Y.keys():
            p = P[yvar]
            root = p[0]
            tail = p[1]
            if root == "observe*":
                output_, sigma = evaluate_program(ast=[tail], sig={}, l=l)
                Y_map[yvar] = output_
            else:
                continue
    # import pdb; pdb.set_trace()
    if DEBUG:
        print('\n')
        print('Evaluted Observed Vars dist: ', Y_map)
        print('\n')

    return X_map, Y_map


def initial_setup_evals_and_maps(X, Y, P, G):
    X_0 = {}
    for lvar in X:
        path = []
        path = traverse(G=G, node=lvar, visit={}, path=path)
        if DEBUG:
            print('Evaluated graph path: ', path)
        # List Reverse
        path.reverse()
        # Evaluate path
        output, sigma = eval_path(path, l={}, Y=Y, P=P)
        if DEBUG:
            print('Evaluated reverse graph path: ', path)
            print('Evaluated graph output: ', output)
        output_ = output[-1]
        try:
            output_ = torch.tensor(output_, requires_grad=True, dtype=torch.float32)
        except:
            output_ = torch.tensor(output_[0], requires_grad=True, dtype=torch.float32)
        X_0[lvar] = output_
    # import pdb; pdb.set_trace()
    if DEBUG:
        print('\n')
        print('Evaluted Latent Vars: ', X_0)
        print('\n')

    Y_0 = {}
    for yvar in Y.keys():
        output_ = Y[yvar]
        if isinstance(output_, int):
            output_ = [float(output_)]
        output_ = torch.tensor(output_, requires_grad=True)
        Y_0[yvar] = output_
    # import pdb; pdb.set_trace()
    if DEBUG:
        print('\n')
        print('Evaluted Observed Vars: ', Y_0)
        print('\n')

    X_map = {}
    # Setup local vars
    l = {**X_0}
    for lv in X:
        p = P[lv]
        root = p[0]
        tail = p[1]
        if root == "sample*":
            output_, sigma = evaluate_program(ast=[tail], sig={}, l=l)
            X_map[lv] = output_
        else:
            continue
    if DEBUG:
        print('\n')
        print('Evaluted Latent dist: ', X_map)
        print('\n')
    del l

    Y_map = {}
    # Setup local vars
    with torch.no_grad():
        l = {}
        for x0 in X_0.keys():
            l[x0] = copy.deepcopy(X_0[x0])
            l[x0].requires_grad = False
        for yvar in Y.keys():
            p = P[yvar]
            root = p[0]
            tail = p[1]
            if root == "observe*":
                output_, sigma = evaluate_program(ast=[tail], sig={}, l=l)
                Y_map[yvar] = output_
            else:
                continue
    # import pdb; pdb.set_trace()
    if DEBUG:
        print('\n')
        print('Evaluted Observed Vars dist: ', Y_map)
        print('\n')

    return X_0, Y_0, X_map, Y_map


def HMC(graph, S, T, eps):
    D, G, E = graph[0], graph[1], graph[2]

    # Compiled graph
    V = G['V']
    A = G['A']
    P = G['P']
    Y = G['Y']
    # Find the link nodes aka nodes not in V
    adj_list = []
    for a in A.keys():
        links = A[a]
        for link in links:
            adj_list.append((a, link))
    # Create Graph
    G_ = {}
    for (n1, n2) in adj_list:
        G_ = make_link(G=G_, node1=n1, node2=n2)

    # Collect Latent Vars
    all_nodes = set(V)
    O = set(Y.keys())
    X = list(all_nodes - O)
    O = list(O)
    if DEBUG:
        print('Observed Vars: ', O)
        print('Latent Vars: ', X)
        print('\n')

    X_0, Y_0, X_map, Y_map = initial_setup_evals_and_maps(X=X, Y=Y, P=P, G=G_)

    # Compute covariance
    Xm = torch.zeros(0, dtype=torch.float32)
    for lvar in X_0.keys():
        xm = X_0[lvar]
        # Check if not torch tensor
        if not torch.is_tensor(xm):
            if isinstance(xm, list):
                xm = torch.tensor(xm, dtype=torch.float32)
            else:
                xm = torch.tensor([xm], dtype=torch.float32)
        # Check for 0 dimensional tensor
        elif xm.shape == torch.Size([]):
            xm = torch.tensor([xm.item()], dtype=torch.float32)
        try:
            Xm = torch.cat((Xm, xm))
        except:
            raise AssertionError('Cannot append the torch tensors')
    Xm = Xm.cpu().detach().numpy()
    if DEBUG:
        print('\n')
        print('X matrix: \n', Xm)
        print('X matrix Shape: \n', Xm.shape)

    if Xm.size == 1:
        mean = torch.tensor([0.0], dtype=torch.float32)
        M = torch.tensor([1.0], dtype=torch.float32)
    else:
        mean = torch.zeros(Xm.shape[0], dtype=torch.float32)
        M = torch.eye(Xm.shape[0], dtype=torch.float32)

    if DEBUG:
        print('\n')
        print('Mean matrix: \n', mean)
        print('Covariance matrix: \n', M)
        print('Covariance matrix Shape: \n', M.shape)
        print('\n')

    R_0 = {}
    uniform_dist = distributions.uniform.Uniform(low=0.0, high=1.0)
    if Xm.size == 1:
        normal_dist = distributions.normal.Normal(loc=mean, scale=M)
    else:
        normal_dist = distributions.multivariate_normal.MultivariateNormal(loc=mean, covariance_matrix=M)
    all_outputs  = []
    for i in range(S):
        R_0_ = normal_dist.sample() # X_ x 1
        # Convert to dict
        R_0 = {}
        i = 0
        for lvar in X_0.keys():
            R_0[lvar] = R_0_[i]
            i += 1
        XT, RT = leapFrog(X=X_0, Y=Y_0, P=P, R_0=R_0, T=T, eps=eps)
        u = (uniform_dist.sample()).item()
        accept1 = computeH(X=XT,  Y=Y_0, P=P, R=RT,  M=M)
        accept2 = computeH(X=X_0, Y=Y_0, P=P, R=R_0, M=M)
        accept = torch.exp((-1.0 * accept1) + accept2)
        if DEBUG:
            print('\n')
            print('Accept: ', accept)
        if u < accept:
            X_0 = {**XT}
        # Compute joint
        joint_log_prob = 0.0
        for lvar in X_0.keys():
            p = P[lvar]
            x1 = X_0[lvar]
            root = p[0]
            tail = p[1]
            if root == "sample*":
                dist_, sigma = evaluate_program(ast=[tail], sig={}, l={})
                if isinstance(x1, list):
                    x1 = x1[0]
                x1_value = dist_.log_prob(x1)
                try:
                    joint_log_prob += x1_value
                except:
                    joint_log_prob += x1_value[0]
            else:
                continue
        all_outputs.append([X_0, joint_log_prob])

    return all_outputs


#------------------------------MAIN--------------------------------------------
if __name__ == '__main__':
    daphne_path = '/Users/tony/Documents/prog-prob/CS539-HW-2'

    # run_deterministic_tests()

    # run_probabilistic_tests()

    for i in range(2,3):
        ## Note: this path should be with respect to the daphne path!
        # ast = daphne(['graph', '-i', f'{daphne_path}/src/programs/{i}.daphne'])
        # ast_path = f'./jsons/graphs/final/{i}.json'
        # with open(ast_path, 'w') as fout:
        #     json.dump(ast, fout, indent=2)
        # print('\n\n\nSample of prior of program {}:'.format(i))

        if i == 1:
            print('Running Graph-Based-sampling for Task number {}:'.format(str(i)))
            ast_path = f'./jsons/HW3/graph/{i}.json'
            with open(ast_path) as json_file:
                ast = json.load(json_file)
            # print(ast)
            # print("Single Run Evaluation: ")
            # output = sample_from_joint(ast)
            # print("Evaluation Output: \n", output)
            # print("\n")
            S = 10000
            T = 10
            eps = 0.01
            all_outputs = HMC(graph=ast, S=S, T=T, eps=eps)
            # print("Evaluation Output: \n", all_outputs)
            # print("\n")
            print("--------------------------------")
            print("HMC Evaluation: ")
            EX = 0.0
            joint_log_prob = []
            ex1 = []
            for i in range(S):
                Xs, joint_prob = all_outputs[i]
                try:
                    EX += Xs['sample2']
                    ex1.append(Xs['sample2'])
                except:
                    EX += Xs['sample2'][0]
                    ex1.append(Xs['sample2'][0])
                try:
                    joint_prob = joint_prob[0].tolist()
                except:
                    pass
                joint_log_prob.append(-joint_prob)

            print("Posterior Mean: ", EX/S)
            print("--------------------------------")
            print("\n")

            plt.plot(joint_log_prob)
            plt.xlabel("Iterations")
            plt.ylabel("Joint-Log-Probability")
            plt.savefig(f'plots/1_HMC_1.png')
            plt.clf()

            plt.plot(ex1)
            plt.xlabel("Iterations")
            plt.ylabel("Slope-Trace")
            plt.savefig(f'plots/1_HMC_2.png')
            plt.clf()

        if i == 2:
            print('Running Graph-Based-sampling for Task number {}:'.format(str(i)))
            ast_path = f'./jsons/HW3/graph/{i}.json'
            with open(ast_path) as json_file:
                ast = json.load(json_file)
            # print(ast)
            # print("Single Run Evaluation: ")
            # output = sample_from_joint(ast)
            # print("Evaluation Output: \n", output)
            # print("\n")
            S = 10000
            T = 10
            eps = 0.01
            print("--------------------------------")
            print("HMC Evaluation: ")
            all_outputs = HMC(graph=ast, S=S, T=T, eps=eps)
            #print("Evaluation Output: \n", output)
            print("\n")
            EX1 = 0.0
            EX2 = 0.0
            ex1 = []
            ex2 = []
            joint_log_prob = []
            for i in range(S):
                Xs, joint_prob = all_outputs[i]
                try:
                    EX1 += Xs['sample1']
                    ex1.extend(Xs['sample1'])
                except:
                    EX1 += Xs['sample1'][0]
                    ex1.extend(Xs['sample1'])
                try:
                    EX2 += Xs['sample2']
                    ex2.extend(Xs['sample2'])
                except:
                    EX2 += Xs['sample2'][0]
                    ex2.extend(Xs['sample2'])
                try:
                    joint_prob = joint_prob[0].tolist()
                except:
                    pass
                joint_log_prob.append(joint_prob)

            print("Posterior Bias Mean: ",  EX2/S)
            print("Posterior Slope Mean: ", EX1/S)
            print("--------------------------------")
            print("\n")

            plt.plot(joint_log_prob)
            plt.xlabel("Iterations")
            plt.ylabel("Joint-Log-Probability")
            plt.savefig(f'plots/2_HMC_1.png')
            plt.clf()

            plt.plot(ex1)
            plt.xlabel("Iterations")
            plt.ylabel("Slope-Trace")
            plt.savefig(f'plots/2_HMC_2.png')
            plt.clf()

            plt.plot(ex2)
            plt.xlabel("Iterations")
            plt.ylabel("Bias-Trace")
            plt.savefig(f'plots/2_HMC_3.png')
#-------------------------------------------------------------------------------
