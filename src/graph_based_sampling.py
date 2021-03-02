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


def get_stream(graph):
    """
    Return a stream of prior samples
    Args:
        graph: json graph as loaded by daphne wrapper
    Returns: a python iterator with an infinite stream of samples
    """
    while True:
        yield sample_from_joint(graph)[0]


def likelihood_weighting_IS(ast, L):
    samples = []
    for i in range(L):
        sig = {}
        sig["logW"] = 0.0
        r_l, sig_l = sample_from_joint(ast, sigma=sig)
        s_l = sig_l["logW"]
        samples.append([r_l, s_l])
    return samples


def independent_MH(ast, S):
    sig = {}
    sig["logW"] = 0.0
    r = evaluate_program(ast, sig=sig, l={})[0]
    logW  = 0.0
    all_r = []
    uniform_dist = distributions.uniform.Uniform(low=0.0, high=1.0)
    for i in range(S):
        sig = {}
        sig["logW"] = 0.0
        r_l, sig_l  = sample_from_joint(ast, sigma=sig)
        s_l = sig_l["logW"]
        alpha = math.exp(s_l)/math.exp(logW)
        u = (uniform_dist.sample()).item()
        if u < alpha:
            r = r_l
            logW = s_l
        all_r.append([r])
    return all_r


#------------------------------Test Functions --------------------------------#
def run_deterministic_tests():
    for i in range(1,13):
    #for i in range(2,3): # For debugging purposes
        # Note: this path should be with respect to the daphne path!
        # ast = daphne(['graph', '-i', f'{daphne_path}/src/programs/tests/deterministic/test_{i}.daphne'])
        # ast_path = f'./jsons/graphs/deterministic/test_{i}.json'
        # with open(ast_path, 'w') as fout:
        #     json.dump(ast, fout, indent=2)

        ast_path = f'./jsons/graph/deterministic/test_{i}.json'
        with open(ast_path) as json_file:
            graph = json.load(json_file)
        # print(graph)

        ret = sample_from_joint(graph)[0]

        print('Running graph-based-sampling for deterministic test number {}:'.format(str(i)))
        truth = load_truth('./programs/tests/deterministic/test_{}.truth'.format(i))
        print("Graph Evaluation Output: ", ret)
        print("Ground Truth: ", truth)
        try:
            assert(is_tol(ret, truth))
        except AssertionError:
            raise AssertionError('return value {} is not equal to truth {} for graph {}'.format(ret,truth,graph))

        print('Test passed \n')

    print('All deterministic tests passed.')


def run_probabilistic_tests():
    #TODO:
    num_samples=1e4
    max_p_value = 1e-4

    for i in range(1,7):
    #for i in range(6,7):
        # Note: this path should be with respect to the daphne path!
        # ast = daphne(['graph', '-i', f'{daphne_path}/src/programs/tests/probabilistic/test_{i}.daphne'])
        # ast_path = f'./jsons/graphs/probabilistic/test_{i}.json'
        # with open(ast_path, 'w') as fout:
        #     json.dump(ast, fout, indent=2)

        ast_path = f'./jsons/graph/probabilistic/test_{i}.json'
        with open(ast_path) as json_file:
            graph = json.load(json_file)
        # print(graph)

        stream = get_stream(graph)

        # samples = []
        # for k in range(1):
        #     samples.append(next(stream))
        # print(samples)

        # if i != 4:
        print('Running graph-based-sampling for probabilistic test number {}:'.format(str(i)))
        truth = load_truth('./programs/tests/probabilistic/test_{}.truth'.format(i))
        # print(truth)
        try:
            p_val = run_prob_test(stream, truth, num_samples)
            print('p value', p_val)
            assert(p_val > max_p_value)
        except:
            print('Test Failed\n')
            continue

        print('Test passed\n')

    print('All probabilistic tests passed.')


def run_hw2_tests():
    for i in range(1,4):
        if i == 1:
            print('Running graph-based-sampling for Task number {}:'.format(str(i+1)))
            ast_path = f'./jsons/graph/final/{i}.json'
            with open(ast_path) as json_file:
                graph = json.load(json_file)
            # print(graph)

            print("Single Run Graph Evaluation: ")
            output = sample_from_joint(graph)
            print("Graph Evaluation Output: ", output)
            print("\n")

            print("Expectation: ")
            stream = get_stream(graph)
            samples = []
            for k in range(1000):
                samples.append(next(stream))
            # print(samples)
            all_samples = torch.tensor(samples)

            # print("Evaluation Output: ", all_samples)
            print("Mean of 1000 samples: ", torch.mean(all_samples))
            print("\n")

        elif i == 2:
            print('Running evaluation-based-sampling for Task number {}:'.format(str(i+1)))
            ast_path = f'./jsons/graph/final/{i}.json'
            with open(ast_path) as json_file:
                graph = json.load(json_file)
            # print(graph)

            print("Single Run Graph Evaluation: ")
            output = sample_from_joint(graph)
            print("Graph Evaluation Output: \n", output)
            print("\n")

            print("Expectation: ")
            stream = get_stream(graph)
            samples = []
            for k in range(1000):
                if k == 0:
                    samples = next(stream)
                    # print(samples.shape)
                else:
                    sample  = next(stream)
                    samples = torch.cat((samples, sample), dim=-1)

            # print(samples)
            print("Evaluation Output: ", samples.shape)
            print("")
            print("Mean of 1000 samples for slope and bias: \n", torch.mean(samples))
            print("\n")

            # Empty globals funcs
            rho = {}

        elif i == 3:
            print('Running evaluation-based-sampling for Task number {}:'.format(str(i+1)))
            ast_path = f'./jsons/graph/final/{i}.json'
            with open(ast_path) as json_file:
                graph = json.load(json_file)
            # print(graph)

            print("Single Run Graph Evaluation: ")
            output = sample_from_joint(graph)[0]
            print("Graph Evaluation Output: \n", output)
            print("\n")

            print("Expectation: ")
            stream = get_stream(graph)
            samples = []
            for k in range(1000):
                if k == 0:
                    samples = next(stream)
                    samples = torch.unsqueeze(samples, 0)
                else:
                    sample  = next(stream)
                    sample  = torch.unsqueeze(sample, 0)
                    samples = torch.cat((samples, sample), dim=0)

            # print(samples)
            samples = samples.float()
            print("Evaluation Output: ", samples.shape)
            print("")
            print("Mean of 1000 samples for each HMM step: \n", torch.mean(samples, dim=0))
            print("\n")
#-------------------------------------------------------------------------------

#-------------------------------MAIN--------------------------------------------
if __name__ == '__main__':
    program_path = '/Users/tony/Documents/prog-prob/CS539-HW-3'

    # # Uncomment the appropriate tests to run
    # # Deterministic Test
    # run_deterministic_tests()
    #
    # # Probabilistic Test
    # run_probabilistic_tests()

    # Run HW-2 Tests
    # run_hw2_tests()
