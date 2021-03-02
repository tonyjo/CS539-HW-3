import json
import math
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
DEBUG = True # Set to true to see intermediate outputs for debugging purposes
def grad_UX(X_, Y_, X, Y):
    Ex = 0.0
    Ey = 0.0
    for x in X_.keys():
        Ex = Ex + torch.log(X_[x])
    for y in Y_.keys():
        Ey = Ey + torch.log(Y_[y])
    Eu = -1.0*(Ex + Ey)

    # Compute grads
    Eu.backward()

    # Collect grads
    dU = {}
    for x in X_.keys():
        dU[x] = X_[x].grad
    for y in Y_.keys():
        dU[y] = Y_[y].grad

    return dU

def leapFrog(X, Y, X_, Y_, R_0, T, eps):
    gradU = grad_UX(X_=X_, Y_=Y_, X=X, Y=Y)
    if DEBUG:
        print("Gradients: ", gradU)

    R_12 = {}
    for du in gradU.keys():
        R_12[du] = R_0[du] - (0.5) * eps * gradU[du]

    for t in range(T):
        for x in X_.keys():
            X_[x] = X_[x] + (eps * R_12[x])
        dU_t = grad_UX(X_=X_, Y_=Y_, X=X, Y=Y)
        for x in X_.keys():
            R_12[x] = R_12[x] - (eps * dU_t[x])

    X_T = {}
    for x in X_.keys():
        X_T[x] = X_[x] + (eps * R_12[x])
    R_T = {}
    dU_t = grad_UX(X_=X_, Y_=Y_, X=X, Y=Y)
    for x in X_.keys():
        R_T[x] = R_12[x] - (eps * dU_t[x])

    return X_T, R_T

def H(X_, Y_, X, Y, R, M):
    # Compute U
    Ex = 0.0
    Ey = 0.0
    for x in X_.keys():
        Ex = Ex + torch.log(X_[x])
    for y in Y_.keys():
        Ey = Ey + torch.log(Y_[y])
    U = -1.0*(Ex + Ey)
    # Compute K
    K = 0.0
    for r in R.keys():
        K += R[r]*torch.inverse(M[r])*R[r]
    K = k/2.0

    H = U + K

    return H


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

    X_0 = {}
    for lvar in X:
        path = []
        path = traverse(G=G_, node=lvar, visit={}, path=path)
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
        output_ = torch.tensor(output_, requires_grad=True)
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

    # Compute variance
    M = {}
    for lvar in X:

    R_0 = {}
    uniform_dist = distributions.uniform.Uniform(low=0.0, high=1.0)
    all_outputs = []
    for i in range(S):
        X_T, R_T = leapFrog(X=X, Y=Y, X_=X_0, Y_=Y_0, R_0=R, T=T, eps=eps)
        u = (uniform_dist.sample()).item()
        accept = torch.exp((-1.0 * H(X_, Y_, X, Y, R, M)) + H(X_, Y_, X, Y, R, M))
        if DEBUG:
            print('\n')
            print('Accept: ', accept)
        if u < accept:
            X_ =  X_T
        all_outputs.append([X_])

    return all_outputs


#------------------------------MAIN--------------------------------------------
if __name__ == '__main__':
    daphne_path = '/Users/tony/Documents/prog-prob/CS539-HW-2'

    # run_deterministic_tests()

    # run_probabilistic_tests()

    for i in range(1,2):
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

            output = HMC(graph=ast, S=1, T=1, eps=0.001)
            print("Gibbs Sampling with Metropolis-Hastings Updates: ")
            print("Evaluation Output: \n", output)
            print("\n")
