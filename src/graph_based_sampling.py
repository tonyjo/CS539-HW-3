import json
import torch
import numpy as np
import torch.distributions as dist
import matplotlib.pyplot as plt

from daphne import daphne
from primitives import PRIMITIVES
from tests import is_tol, run_prob_test,load_truth

# OPS
env = PRIMITIVES
DEBUG = False # Set to true to see intermediate outputs for debugging purposes
#----------------------------Evaluation Functions -----------------------------#
def deterministic_eval(exp):
    "Evaluation function for the deterministic target language of the graph based representation."
    if type(exp) is list:
        op = exp[0]
        args = exp[1:]
        return env[op](*map(deterministic_eval, args))
    elif type(exp) in [int, float]:
        # We use torch for all numerical objects in our evaluator
        return torch.Tensor([float(exp)]).squeeze()
    elif type(exp) is torch.Tensor:
        return exp
    else:
        raise Exception("Expression type unknown.", exp)

def topological_sort(nodes, edges):
    result = []
    visited = {}
    def helper(node):
        if node not in visited:
            visited[node] = True
            if node in edges:
                for child in edges[node]:
                    helper(child)
            result.append(node)
    for node in nodes:
        helper(node)
    return result[::-1]

def plugin_parent_values(expr, trace):
    if type(expr) == str and expr in trace:
        return trace[expr]
    elif type(expr) == list:
        return [plugin_parent_values(child_expr, trace) for child_expr in expr]
    else:
        return expr

def sample_from_joint(graph, sigma={}):
    """
    This function does ancestral sampling starting from the prior.
    1. Run topological sort on V using V and A, resulting in an array of v's
    2. Iterate through sample sites of the sorted array, and save sampled results on trace dictionary using P and Y
    - If keyword is sample*, first recursively replace sample site names with trace values in the expression from P. Then, run deterministic_eval.
    - If keyword is observe*, put the observation value in the trace dictionary
    3. Filter the trace dictionary for things sample sites you should return
    Obtained From: https://github.com/truebluejason/prob_prog_project/blob/master/graph_based_sampling.py
    """
    procs, model, expr = graph[0], graph[1], graph[2]
    nodes, edges, links, obs = model['V'], model['A'], model['P'], model['Y']
    sorted_nodes = topological_sort(nodes, edges)

    trace = {}
    for node in sorted_nodes:
        keyword = links[node][0]
        if keyword == "sample*":
            # import pdb; pdb.set_trace()
            link_expr = links[node][1]
            if DEBUG:
                print('Link Expression without parent vals: ', link_expr)
            link_expr = plugin_parent_values(link_expr, trace)
            if DEBUG:
                print('Link Expression: ', link_expr)
            dist_obj  = deterministic_eval(link_expr)
            trace[node] = dist_obj.sample()
        elif keyword == "observe*":
            # import pdb; pdb.set_trace()
            value = obs[node]
            link_expr = links[node][1]
            link_expr = plugin_parent_values(link_expr, trace)
            if DEBUG:
                print('Link Expression: ', link_expr)
            dist_obj = deterministic_eval(link_expr)
            if DEBUG:
                print('Distribution Object: ', dist_obj)
            # Obtain likelihood
            if "logW" in sigma.keys():
                sigma["logW"] += dist_obj.log_prob(value)
            else:
                sigma["logW"]  = dist_obj.log_prob(value)
            # Trace
            trace[node] = value

    expr = plugin_parent_values(expr, trace)

    return deterministic_eval(expr), sigma


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

    # for i in range(1,6):
    for i in range(1,2):
        # # Note: this path should be with respect to the daphne path!
        # ast = daphne(['graph', '-i', f'{program_path}/src/programs/{i}.daphne'])
        # ast_path = f'./jsons/HW3/graph/{i}.json'
        # with open(ast_path, 'w') as fout:
        #     json.dump(ast, fout, indent=2)
        # print('\n\n\nSample of prior of program {}:'.format(i))

        if i == 1:
            print('Running Graph-Based-sampling for Task number {}:'.format(str(i+1)))
            ast_path = f'./jsons/HW3/graph/{i}.json'
            with open(ast_path) as json_file:
                ast = json.load(json_file)
            # print(ast)
            output = sample_from_joint(ast)
            print(output)

            print("--------------------------------")
            print("Importance sampling Evaluation: ")
            num_samples = 1000
            all_output = likelihood_weighting_IS(ast=ast, L=num_samples)

            W_k = 0.0
            for k in range(num_samples):
                r_l, W_l = all_output[k]
                W_k += W_l

            expected_output = 0.0
            for l in range(num_samples):
                r_l, W_l = all_output[l]
                expected_output += ((W_l/W_k) * r_l)
            print("Output: ", expected_output)
            print("--------------------------------")
            print("\n")
