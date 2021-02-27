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
DEBUG = True # Set to true to see intermediate outputs for debugging purposes
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

def sample_from_joint(graph):
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

    sigma = {}
    trace = {}
    for node in sorted_nodes:
        keyword = links[node][0]
        if keyword == "sample*":
            link_expr = links[node][1]
            if DEBUG:
                print('Link Expression without parent vals: ', link_expr)
            link_expr = plugin_parent_values(link_expr, trace)
            if DEBUG:
                print('Link Expression: ', link_expr)
            dist_obj  = deterministic_eval(link_expr)
            trace[node] = dist_obj.sample()
        elif keyword == "observe*":
            trace[node] = obs[node]
        import pdb; pdb.set_trace()
        if DEBUG:
            print('Trace: ', trace)

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


#------------------------------MAIN--------------------------------------------
if __name__ == '__main__':
    daphne_path = '/Users/tony/Documents/prog-prob/CS539-HW-2'

    # run_deterministic_tests()
    #
    # run_probabilistic_tests()

    for i in range(1,5):
        ## Note: this path should be with respect to the daphne path!
        # ast = daphne(['graph', '-i', f'{daphne_path}/src/programs/{i}.daphne'])
        # ast_path = f'./jsons/graphs/final/{i}.json'
        # with open(ast_path, 'w') as fout:
        #     json.dump(ast, fout, indent=2)
        # print('\n\n\nSample of prior of program {}:'.format(i))

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


    #     elif i == 2:
    #         print('Running evaluation-based-sampling for Task number {}:'.format(str(i+1)))
    #         ast_path = f'./jsons/graphs/final/{i}.json'
    #         with open(ast_path) as json_file:
    #             graph = json.load(json_file)
    #         # print(graph)
    #
    #         print("Single Run Graph Evaluation: ")
    #         output = sample_from_joint(graph)
    #         print("Graph Evaluation Output: \n", output)
    #         print("\n")
    #
    #         print("Expectation: ")
    #         stream = get_stream(graph)
    #         samples = []
    #         for k in range(1000):
    #             if k == 0:
    #                 samples = next(stream)
    #                 # print(samples.shape)
    #             else:
    #                 sample  = next(stream)
    #                 samples = torch.cat((samples, sample), dim=-1)
    #
    #         # print(samples)
    #         print("Evaluation Output: ", samples.shape)
    #         print("")
    #         print("Mean of 1000 samples for slope and bias: \n", torch.mean(samples, dim=1))
    #         print("\n")
    #
    #         # Empty globals funcs
    #         rho = {}
    #
    #     elif i == 3:
    #         print('Running evaluation-based-sampling for Task number {}:'.format(str(i+1)))
    #         ast_path = f'./jsons/graphs/final/{i}.json'
    #         with open(ast_path) as json_file:
    #             graph = json.load(json_file)
    #         # print(graph)
    #
    #         print("Single Run Graph Evaluation: ")
    #         output = sample_from_joint(graph)
    #         output = torch.transpose(output, 0, 1)
    #         print("Graph Evaluation Output: \n", output)
    #         print("\n")
    #
    #         print("Expectation: ")
    #         stream = get_stream(graph)
    #         samples = []
    #         for k in range(1000):
    #             if k == 0:
    #                 samples = next(stream)
    #                 samples = torch.transpose(samples, 0, 1)
    #                 print(samples.shape)
    #             else:
    #                 sample  = next(stream)
    #                 sample  = torch.transpose(sample, 0, 1)
    #                 samples = torch.cat((samples, sample), dim=0)
    #
    #         # print(samples)
    #         print("Evaluation Output: ", samples.shape)
    #         print("")
    #         print("Mean of 1000 samples for each HMM step: \n", torch.mean(samples, dim=0))
    #         print("\n")
    #
    #     elif i == 4:
    #         print('Running evaluation-based-sampling for Task number {}:'.format(str(i+1)))
    #         ast_path = f'./jsons/graphs/final/{i}.json'
    #         with open(ast_path) as json_file:
    #             graph = json.load(json_file)
    #         # print(graph)
    #
    #         print("Single Run Graph Evaluation: ")
    #         output = sample_from_joint(graph)
    #         output = torch.transpose(output, 0, 1)
    #         print("Graph Evaluation Output: \n", output)
    #         print("\n")
    #
    #         print("Expectation: ")
    #         stream = get_stream(graph)
    #         samples = []
    #         for k in range(1000):
    #             if k == 0:
    #                 samples = next(stream)
    #                 samples = torch.transpose(samples, 0, 1)
    #                 # print(samples.shape)
    #             else:
    #                 sample  = next(stream)
    #                 sample  = torch.transpose(sample, 0, 1)
    #                 samples = torch.cat((samples, sample), dim=0)
    #
    #         # print(samples)
    #         print("Evaluation Output: ", samples.shape)
    #         print("")
    #         print("Mean of 1000 samples for each Neural Network weight(s): \n", torch.mean(samples, dim=0))
    #         print("\n")
#-------------------------------------------------------------------------------
