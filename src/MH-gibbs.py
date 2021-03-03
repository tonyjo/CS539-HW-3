import json
import math
import torch
import torch.distributions as distributions
from daphne import daphne
import matplotlib.pyplot as plt
# funcprimitives
from evaluation_based_sampling import evaluate_program
from tests import is_tol, run_prob_test,load_truth
# Useful functions
from utils import _hashmap, _vector, _totensor
from utils import _put, _remove, _append, _get
from utils import _squareroot, _mat_repmat, _mat_transpose

# 2 or more var OPS
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


# Global vars
global rho
rho = {}
DEBUG = False # Set to true to see intermediate outputs for debugging purposes
def sample_from_joint(graph):
    """
    This function does ancestral sampling starting from the prior.
    Args:
        graph: json Graph of FOPPL program
    Returns: sample from the prior of ast
    """
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
    # if DEBUG:
    #     print("Created Adjacency list: ", adj_list)

    # Create Graph
    G_ = {}
    for (n1, n2) in adj_list:
        G_ = make_link(G=G_, node1=n1, node2=n2)

    # Test
    # E = "observe8"

    if DEBUG:
        print("Constructed Graph: ", G_)
        print("Evaluation Expression: ", E)

    if isinstance(E, str):
        # output = torch.zeros(0, dtype=torch.float32)
        path = []
        path = traverse(G=G_, node=E, visit={}, path=path)
        if DEBUG:
            print('Evaluated graph path: ', path)
        # List Reverse
        path.reverse()
        # Evaluate path
        output, sigma = eval_path(path, l={}, Y=Y, P=P)
        if DEBUG:
            print('Evaluated reverse graph path: ', path)
            print('Evaluated graph output: ', output)
        return output

    elif isinstance(E, list):
        # Setup Local vars
        l = {}
        sigma = {}
         # Add Y to local vars
        for y in Y.keys():
            l[y] = Y[y]
         # Evaluate and add link functions to local vars
        for idk in range(2):
            for pi in P.keys():
                p = P[pi]
                root = p[0]
                tail = p[1]
                if root == "sample*":
                    sample_eval = ["sample", tail]
                    output_, sigma = evaluate_program(ast=[sample_eval], sig=sigma, l=l)
                    l[pi] = output_
                else:
                    continue

        # Evalute
        root_expr, *tail = E
        if DEBUG:
            print('Root OP: ', root_expr)
            print('TAIL: ', tail)
            print('Local vars: ', l)
            print('\n')

        eval_outputs = []
        # Conditonal
        if root_expr == 'if':
            # (if e1 e2 e3)
            if DEBUG:
                print('Conditonal Expr1 :  ', tail[0])
                print('Conditonal Expr2 :  ', tail[1])
                print('Conditonal Expr3 :  ', tail[2])
            e1_, sigma = evaluate_program([tail[0]], sig=sigma, l=l)
            if DEBUG:
                print('Conditonal eval :  ', e1_)
            if e1_:
                if tail[1] in V:
                    path = []
                    op_eval, sigma = eval_node(node=tail[1], path=path, G=G_, Y=Y, P=P)
                    op_eval = op_eval[-1]
                    if DEBUG:
                        print('If eval: ', op_eval)
                else:
                    op_eval, sigma = evaluate_program([tail[1]], sig=sigma, l=l)
                    if DEBUG:
                        print('If eval: ', op_eval)
            else:
                if tail[2] in V:
                    path = []
                    op_eval, sigma  = eval_node(node=tail[2], path=path, G=G_, Y=Y, P=P)
                    op_eval = op_eval[-1]
                else:
                    op_eval, sigma = evaluate_program([tail[1]], sig=sigma, l=l)
        # Vector
        elif root_expr == "vector":
            # import pdb; pdb.set_trace()
            if DEBUG:
                print('Data Structure data: ', tail)
            # Eval tails:
            op_eval = torch.zeros(0, dtype=torch.float32)
            for T in range(len(tail)):
                # Check for single referenced string
                if isinstance(tail[T], str):
                    if tail[T] in V:
                        path = []
                        eval_T = eval_node(node=tail[T], path=path, G=G_, Y=Y, P=P)
                    else:
                        eval_T = evaluate_program([tail[T]], sig=sigma, l=l)
                else:
                    eval_T = evaluate_program([tail[T]], sig, l=l)
                if DEBUG:
                    print('Evaluated Data Structure data: ', eval_T)
                try:
                    eval_T = eval_T[0]
                except:
                    # In case of functions returning only a single value & not sigma
                    pass
                try:
                    eval_T = eval_T[-1]
                except:
                    # In case of functions returning only a single value & not full trace
                    pass
                # IF sample object then take a sample
                try:
                    eval_T = eval_T.sample()
                except:
                    pass
                # Check if not torch tensor
                if not torch.is_tensor(eval_T):
                    if isinstance(eval_T, list):
                        eval_T = torch.tensor(eval_T, dtype=torch.float32)
                    else:
                        eval_T = torch.tensor([eval_T], dtype=torch.float32)
                # Check for 0 dimensional tensor
                elif eval_T.shape == torch.Size([]):
                    eval_T = torch.tensor([eval_T.item()], dtype=torch.float32)
                # Concat
                try:
                    op_eval = torch.cat((op_eval, eval_T))
                except:
                    raise AssertionError('Cannot append the torch tensors')
            if DEBUG:
                print('Eval Data Structure data: ', op_eval)
        # Others
        else:
            sigma = {}
            for ei in range(len(tail)):
                if ei in V:
                    path = []
                    output, sigma_ = eval_node(node=ei, path=path, G=G_, Y=Y, P=P)
                    output = output[-1]
                else:
                    output, sigma = evaluate_program([ei], sig=sigma, l=l)
                # Collect
                eval_outputs.append([output])
            if DEBUG:
                print('For eval: ', eval_outputs)
            # Evaluate expression
            if root_expr in one_ops.keys():
                op_func = one_ops[root_expr]
                op_eval = op_func(eval_outputs[0])
            else:
                op_func = two_ops[root_expr]
                if DEBUG:
                    print('Final output: ', eval_outputs[0])
                    print('Final output: ', eval_outputs[1])
                op_eval = op_func(eval_outputs[0], eval_outputs[1])

        if DEBUG:
            print('Final output: ', op_eval)

        return op_eval

    else:
        raise AssertionError('Invalid input of E!')

    return None

#--------------------------------GIBBS-with-MH----------------------------------
def eval_free_vars(v, X, Y, P):
    # TODO: depend on the X_var and not a new sample
    # Setup Local vars
    l = {}
     # Add Y to local vars
    for y in Y.keys():
        l[y] = Y[y]
    # Add evaluated X to to local vars
    for x in X.keys():
        l[x] = X[x]

    p = P[v] # [sample* [n, 5, [sqrt, 5]]]
    root = p[0]
    tail = p[1]

    if DEBUG:
        print('PMF for Node: ', p)
        print('Empty sample Root: ', root)
        print('Empty sample Root: ', tail)
        print('Local vars: ', l)
        print('\n')

    if root == "sample*":
        sample_eval = ["sample", tail]
        if DEBUG:
            print('Sample AST: ', sample_eval)
        output_, sigma = evaluate_program(ast=[sample_eval], sig={}, l=l)
        if DEBUG:
            print('Evaluated sample: ', output_)

    elif root == "observe*":
        try:
            sample_eval = ["observe", tail, p[2]]
        except:
            sample_eval = ["observe", tail]
        if DEBUG:
            print('Sample AST: ', sample_eval)
        output_, sigma = evaluate_program(ast=[sample_eval], sig={}, l=l)
        if DEBUG:
            print('Evaluated sample: \n', output_, '\n', sigma)
            print('\n')

    else:
        raise AssertionError('Unsupported operation!')

    return output_, sigma


def GibbsAccept(x, X_, X_n, Q_x, V, X, O, A, P, Y, G):
    d    = Q_x[x]
    d_   = Q_x[x]
    x_x  = X_[x]
    x_nx = X_n[x]
    if isinstance(x_x, list):
        x_x = x_x[0]
    if isinstance(x_x, list):
        x_nx = x_nx[0]
    log_alpha = d.log_prob(x_x) - d_.log_prob(x_nx)
    # TODO: Check this
    Vx = A[x]
    # import pdb; pdb.set_trace()
    if DEBUG:
        print('Pre-Log Alpha:', x_x, x_nx, log_alpha)
        print('Distribution-1: ', d)
        print('Distribution-2: ', d_)
        print('Verticies: ', Vx, ' that depend on ', x)
        print('\n')

    for v in Vx:
        _, sig  = eval_free_vars(v=v, X=X_n, Y=Y, P=P)
        log_alpha = log_alpha + sig["logW"]
        _, sig  = eval_free_vars(v=v, X=X_, Y=Y, P=P)
        log_alpha = log_alpha - sig["logW"]

    # import pdb; pdb.set_trace()
    if torch.is_tensor(log_alpha):
        log_alpha = log_alpha.tolist()
    if DEBUG:
        print('Log Alpha:', log_alpha)

    if isinstance(log_alpha, list):
        log_alpha = log_alpha[0]
    # Change from log
    try:
        alpha = math.exp(log_alpha)
    except OverflowError:
        alpha = 0.5

    return alpha


def GibbsStep(X_, Q_x, V, X, O, A, P, Y, G):
    uniform_dist = distributions.uniform.Uniform(low=0.0, high=1.0)
    for x in X:
        d = Q_x[x]
        X_n = {**X_}
        X_n[x] = d.sample()
        alpha = GibbsAccept(x=x, X_=X_, X_n=X_n, Q_x=Q_x, V=V, X=X, O=O, A=A, P=P, Y=Y, G=G)
        u = (uniform_dist.sample()).item()
        if DEBUG:
            print('Alpha: ', alpha)
            print('Unf: ', u)
            print('X_:', X_)
        if u < alpha:
            X_ = {**X_n}
    return X_

def Gibbs(graph, S):
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

    X_s_minus_1 = {}
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
        X_s_minus_1[lvar] = output[-1]
    # import pdb; pdb.set_trace()
    if DEBUG:
        print('Evaluted Latent Vars: ', X_s_minus_1)
        print('\n')

    Q_x = {}
    # Setup local vars
    l = {**X_s_minus_1}
    for lvar in X:
        p = P[lvar]
        root = p[0]
        tail = p[1]
        if root == "sample*":
            output_, sigma = evaluate_program(ast=[tail], sig={}, l=l)
            Q_x[lvar] = output_
        else:
            continue
    # import pdb; pdb.set_trace()
    if DEBUG:
        print('Evaluted Latent Vars dist.: ', Q_x)

    all_outputs = []
    for x in range(S):
        X_s = GibbsStep(X_=X_s_minus_1, Q_x=Q_x, V=V, X=X, O=O, A=A, P=P, Y=Y, G=G_)
        # Compute joint
        joint_log_prob = 0.0
        for lvar in X_s.keys():
            p = P[lvar]
            x1 = X_s[lvar]
            root = p[0]
            tail = p[1]
            if root == "sample*":
                dist_, sigma = evaluate_program(ast=[tail], sig={}, l=l)
                if isinstance(x1, list):
                    x1 = x1[0]
                x1_value = dist_.log_prob(x1)
                try:
                    joint_log_prob += x1_value
                except:
                    joint_log_prob += x1_value[0]
            else:
                continue
        # Update
        X_s_minus_1 = {**X_s}
        # Collect
        all_outputs.append([X_s, joint_log_prob])

    return all_outputs

#------------------------------MAIN--------------------------------------------
if __name__ == '__main__':
    daphne_path = '/Users/tony/Documents/prog-prob/CS539-HW-2'

    # run_deterministic_tests()

    # run_probabilistic_tests()

    for i in range(1,5):
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

            # output = Gibbs(graph=ast, S=1)
            # print("Gibbs Sampling with Metropolis-Hastings Updates: ")
            # print("Evaluation Output: \n", output)
            # print("\n")

            print("--------------------------------")
            print("Gibbs Sampling with Metropolis-Hastings Updates Evaluation: ")
            num_samples = 10000
            all_outputs = Gibbs(graph=ast, S=num_samples)

            EX = 0.0
            ee = []
            joint_log_prob = []
            for i in range(num_samples):
                Xs, joint_prob = all_outputs[i]
                try:
                    EX += Xs['sample2']
                    ee.extend(Xs['sample2'])
                except:
                    EX += Xs['sample2'][0]
                    ee.extend(Xs['sample2'][0])
                try:
                    joint_prob = joint_prob[0].tolist()
                except:
                    pass
                joint_log_prob.append(-joint_prob)

            print("Posterior Mean: ", EX/num_samples)
            print("--------------------------------")
            print("\n")

            plt.plot(joint_log_prob)
            plt.xlabel("Iterations")
            plt.ylabel("Joint-Log-Probability")
            plt.savefig(f'plots/1_MH_1.png')
            plt.clf()

            plt.plot(ee)
            plt.xlabel("Iterations")
            plt.ylabel("Traces")
            plt.savefig(f'plots/1_MH_2.png')
            plt.clf()

        elif i == 2:
            print('Running Graph-Based-sampling for Task number {}:'.format(str(i)))
            ast_path = f'./jsons/HW3/graph/{i}.json'
            with open(ast_path) as json_file:
                ast = json.load(json_file)
            # print(ast)
            # print("Single Run Evaluation: ")
            # output = sample_from_joint(ast)
            # print("Evaluation Output: \n", output)
            # print("\n")

            # output = Gibbs(graph=ast, S=1)
            # print("Gibbs Sampling with Metropolis-Hastings Updates: ")
            # print("Evaluation Output: \n", output)
            # print("\n")

            print("--------------------------------")
            print("Gibbs Sampling with Metropolis-Hastings Updates Evaluation: ")
            num_samples = 10000
            all_outputs = Gibbs(graph=ast, S=num_samples)

            EX1 = 0.0
            EX2 = 0.0
            ex1 = []
            ex2 = []
            joint_log_prob = []
            for i in range(num_samples):
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

            print("Posterior Bias Mean: ",  EX2/num_samples)
            print("Posterior Slope Mean: ", EX1/num_samples)
            print("--------------------------------")
            print("\n")

            plt.plot(joint_log_prob)
            plt.xlabel("Iterations")
            plt.ylabel("Joint-Log-Probability")
            plt.savefig(f'plots/2_MH_1.png')
            plt.clf()

            plt.plot(ex1)
            plt.xlabel("Iterations")
            plt.ylabel("Slope-Trace")
            plt.savefig(f'plots/2_MH_2.png')
            plt.clf()

            plt.plot(ex2)
            plt.xlabel("Iterations")
            plt.ylabel("Bias-Trace")
            plt.savefig(f'plots/2_MH_3.png')
            plt.clf()


        elif i == 3:
            print('Running Graph-Based-sampling for Task number {}:'.format(str(i)))
            ast_path = f'./jsons/HW3/graph/{i}.json'
            with open(ast_path) as json_file:
                ast = json.load(json_file)
            # print(ast)
            # print("Single Run Evaluation: ")
            # output = sample_from_joint(ast)
            # print("Evaluation Output: ", output)
            # print("\n")

            print("--------------------------------")
            print("Gibbs Sampling with Metropolis-Hastings Updates Evaluation: ")
            num_samples = 10000
            all_outputs = Gibbs(graph=ast, S=num_samples)

            EX = 0.0
            joint_log_prob = []
            ee = []
            for i in range(num_samples):
                Xs, joint_prob = all_outputs[i]
                if isinstance(Xs['sample7'], list):
                    e1 = Xs['sample7'][0]
                else:
                    e1 = Xs['sample7']
                if isinstance(Xs['sample9'], list):
                    e2 = Xs['sample9'][0]
                else:
                    e2 = Xs['sample9']
                EX += float(torch.eq(e1, e2).item()) # ("=", "sample7", "sample9")
                ee.append(float(torch.eq(e1, e2).item()))
                try:
                    joint_prob = joint_prob[0].tolist()
                except:
                    pass
                joint_log_prob.append(-joint_prob)

            print("Posterior Mean: ", EX/num_samples)
            print("--------------------------------")
            print("\n")

            plt.plot(joint_log_prob)
            plt.xlabel("Iterations")
            plt.ylabel("Joint-Log-Probability")
            plt.savefig(f'plots/3_MH_1.png')
            plt.clf()

            plt.plot(ee)
            plt.xlabel("Iterations")
            plt.ylabel("Traces")
            plt.savefig(f'plots/3_MH_2.png')
            plt.clf()


        elif i == 4:
            print('Running Graph-Based-sampling for Task number {}:'.format(str(i)))
            ast_path = f'./jsons/HW3/graph/{i}.json'
            with open(ast_path) as json_file:
                ast = json.load(json_file)
            # print(ast)
            # print("Single Run Evaluation: ")
            # output = sample_from_joint(ast)
            # print("Evaluation Output: ", output)
            # print("\n")

            print("--------------------------------")
            print("Gibbs Sampling with Metropolis-Hastings Updates Evaluation: ")
            num_samples = 10000
            all_outputs = Gibbs(graph=ast, S=num_samples)

            EX = 0.0
            joint_log_prob = []
            ee = []
            for i in range(num_samples):
                Xs, joint_prob = all_outputs[i]
                # ["if",["=","sample2",true],"sample3","sample4"]
                if isinstance(Xs['sample2'], list):
                    e1 = Xs['sample2'][0].item()
                else:
                    e1 = Xs['sample2'].item()
                if isinstance(Xs['sample3'], list):
                    e2 = Xs['sample3'][0].item()
                else:
                    e2 = Xs['sample3'].item()
                if isinstance(Xs['sample4'], list):
                    e3 = Xs['sample4'][0].item()
                else:
                    e3 = Xs['sample4'].item()

                EX += e1
                if e1 == True:
                    ee.append(e2)
                    EX += e2
                else:
                    ee.append(e3)
                    EX += e3

                try:
                    joint_prob = joint_prob[0].tolist()
                except:
                    pass
                joint_log_prob.append(joint_prob)

            print("Posterior Mean: ", EX/num_samples)
            print("--------------------------------")
            print("\n")

            plt.plot(joint_log_prob)
            plt.xlabel("Iterations")
            plt.ylabel("Joint-Log-Probability")
            plt.savefig(f'plots/4_MH_1.png')
            plt.clf()

            plt.plot(ee)
            plt.xlabel("Iterations")
            plt.ylabel("Traces")
            plt.savefig(f'plots/4_MH_2.png')
            plt.clf()
#-------------------------------------------------------------------------------
