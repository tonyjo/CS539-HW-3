import os
import json
import torch
import torch.distributions as distributions
from daphne import daphne
from tests import is_tol, run_prob_test,load_truth
import matplotlib.pyplot as plt
# Useful functions
from utils import _hashmap, _vector, _totensor
from utils import _put, _remove, _append, _get
from utils import _squareroot, _mat_repmat, _mat_transpose

# OPS
basic_ops = {'+':torch.add,
             '-':torch.sub,
             '*':torch.mul,
             '/':torch.div
}

math_ops = {'sqrt': lambda x: _squareroot(x)
}

data_struct_ops = {'vector': lambda x: _vector(x),
                   'hash-map': lambda x: _hashmap(x)
}

data_interact_ops = {'first': lambda x: x[0],      # retrieves the first element of a list or vector e
                     'second': lambda x: x[1],     # retrieves the second element of a list or vector e
                     'last': lambda x: x[-1],      # retrieves the last element of a list or vector e
                     'rest': lambda x: x[1:],      # retrieves the rest of the element of a list except the first one
                     'get': lambda x, idx: _get(x, idx),              # retrieves an element at index e2 from a list or vector e1, or the element at key e2 from a hash map e1.
                     'append': lambda x, y: _append(x, y),           # (append e1 e2) appends e2 to the end of a list or vector e1
                     'remove': lambda x, idx: _remove(x, idx),        # (remove e1 e2) removes the element at index/key e2 with the value e2 in a vector or hash-map e1.
                     'put': lambda x, idx, value: _put(x, idx, value) # (put e1 e2 e3) replaces the element at index/key e2 with the value e3 in a vector or hash-map e1.
}

dist_ops = {"normal":lambda mu, sig: distributions.normal.Normal(loc=mu, scale=sig),
            "beta":lambda a, b: distributions.beta.Beta(concentration1=a, concentration0=b),
            "exponential":lambda rate: distributions.exponential.Exponential(rate=rate),
            "uniform": lambda low, high: distributions.uniform.Uniform(low=low, high=high),
            "discrete": lambda probs: distributions.categorical.Categorical(probs=probs)
}

cond_ops={"<":  lambda a, b: a < b,
          ">":  lambda a, b: a > b,
          ">=": lambda a, b: a >= b,
          "<=": lambda a, b: a <= b,
          "|":  lambda a, b: a or b
}

nn_ops={"mat-tanh": lambda a: torch.tanh(a),
        "mat-add": lambda a, b: torch.add(a, b),
        "mat-mul": lambda a, b: torch.matmul(a, b),
        "mat-repmat": lambda a, b, c: _mat_repmat(a, b, c),
        "mat-transpose": lambda a: _mat_transpose(a)
}

# Global vars
global rho;
rho = {}
DEBUG = False # Set to true to see intermediate outputs for debugging purposes
#----------------------------Evaluation Functions -----------------------------#
def evaluate_program(ast, sig=None, l={}):
    """
    Evaluate a program as desugared by daphne, generate a sample from the prior
    Args:
        ast: json FOPPL program
    Returns: sample from the prior of ast
    """
    # Empty list
    if not ast:
        return [False, sig]

    if DEBUG:
        print('Current AST: ', ast)

    # import pdb; pdb.set_trace()
    if len(ast) == 1:
        # Check if a single string ast [['mu']]
        single_val = False
        if isinstance(ast[0], str):
            root = ast[0]
            tail = []
            single_val = True
        else:
            ast = ast[0]

        if DEBUG:
            print('Current program: ', ast)
        try:
            # Check if a single string such as ast = ['mu']
            if not single_val:
                if len(ast) == 1:
                    if isinstance(ast[0], str):
                        root = ast[0]
                        tail = []
                else:
                    root, *tail = ast
            if DEBUG:
                print('Current OP: ', root)
                print('Current TAIL: ', tail)
            # Basic primitives
            if root in basic_ops.keys():
                op_func = basic_ops[root]
                eval_1  = evaluate_program([tail[0]], sig, l=l)[0]
                 # Make sure in floating point
                if torch.is_tensor(eval_1):
                    eval_1 = eval_1.type(torch.float32)
                elif isinstance(eval_1, int):
                    eval_1 = float(eval_1)
                if DEBUG:
                    print('Basic OP eval-1: ', eval_1)
                    print('Basic OP Tail: ', tail[1:])
                return [op_func(eval_1, evaluate_program(tail[1:], sig, l=l)[0]), sig]
            # Math ops
            elif root in math_ops.keys():
                op_func = math_ops[root]
                return [op_func(tail), sig]
            # NN ops
            elif root in nn_ops.keys():
                # import pdb; pdb.set_trace()
                op_func = nn_ops[root]
                if root == "mat-add" or root == "mat-mul":
                    e1, e2 = tail
                    # Operand-1
                    if isinstance(e1, list) and len(e1) == 1:
                        a, _ = evaluate_program(e1, sig, l=l)
                    elif isinstance(e1, list):
                        a, _ = evaluate_program([e1], sig, l=l)
                    else:
                        # Most likely a pre-defined varibale in l
                        a = l[e1]
                    # Operand-2
                    if isinstance(e2, list) and len(e2) == 1:
                        b, _ = evaluate_program(e2, sig, l=l)
                    elif isinstance(e2, list):
                        b, _ = evaluate_program([e2], sig, l=l)
                    else:
                        b = l[e2] # Most likely a pre-defined varibale in l
                    if DEBUG:
                        print('Evaluated MatMul-1: ', a)
                        print('Evaluated MatMul-2: ', b)
                    # OP
                    return [op_func(a, b), sig]
                # ["mat-repmat", "b_0", 1, 5]
                elif root == "mat-repmat":
                    e1, e2, e3 = tail
                    # Initial MAT
                    if isinstance(e1, list) and len(e1) == 1:
                        a, _ = evaluate_program(e1, sig, l=l)
                    elif isinstance(e1, list):
                        a, _ = evaluate_program([e1], sig, l=l)
                    else:
                        a = l[e1] # Most likely a pre-defined varibale in l
                    # Repeat axis 1
                    if isinstance(e2, list) and len(e2) == 1:
                        b, _ = evaluate_program(e2, sig, l=l)
                    elif isinstance(e2, list):
                        b, _ = evaluate_program([e2], sig, l=l)
                    elif isinstance(e2, float) or isinstance(e2, int):
                        b = int(e2)
                    else:
                        b = l[e2] # Most likely a pre-defined varibale in l
                    # Repeat axis 2
                    if isinstance(e3, list) and len(e3) == 1:
                        c, _ = evaluate_program(e3, sig, l=l)
                    elif isinstance(e3, list):
                        c, _ = evaluate_program([e3], sig, l=l)
                    elif isinstance(e3, float) or isinstance(e3, int):
                        c = int(e3)
                    else:
                        c = l[e3] # Most likely a pre-defined varibale in l
                    # OP
                    return [op_func(a, b, c), sig]
                else:
                    e1 = tail
                    if isinstance(e1, list) and len(e1) == 1:
                        a, _ = evaluate_program(e1, sig, l=l)
                    elif isinstance(e1, list):
                        a, _ = evaluate_program([e1], sig, l=l)
                    else:
                        a = l[e1] # Most likely a pre-defined varibale in l
                    if DEBUG:
                        print('Evaluated Matrix: ', a)
                    # OP
                    return [op_func(a), sig]
            # Data structures-- Vector
            elif root == "vector":
                # import pdb; pdb.set_trace()
                op_func = data_struct_ops[root]
                if DEBUG:
                    print('Data Structure data: ', tail)
                # Eval tails:
                tail_data = torch.zeros(0, dtype=torch.float32)
                for T in range(len(tail)):
                    # Check for single referenced string
                    if isinstance(tail[T], str):
                        VT = [tail[T]]
                    else:
                        VT = tail[T]
                    if DEBUG:
                        print('Pre-Evaluated Data Structure data: ', VT)
                    eval_T = evaluate_program([VT], sig, l=l)
                    if DEBUG:
                        print('Evaluated Data Structure data: ', eval_T)
                    try:
                        eval_T = eval_T[0]
                    except:
                        # In case of functions returning only a single value & not sigma
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
                        tail_data = torch.cat((tail_data, eval_T))
                    except:
                        raise AssertionError('Cannot append the torch tensors')
                if DEBUG:
                    print('Eval Data Structure data: ', tail_data)
                return [tail_data, sig]
            # Data structures-- hash-map
            elif root == "hash-map":
                op_func = data_struct_ops[root]
                return [op_func(tail), sig]
            # Data structures interaction
            elif root in data_interact_ops.keys():
                op_func = data_interact_ops[root]
                # ['put', ['vector', 2, 3, 4, 5], 2, 3]
                if root == 'put':
                    e1, e2, e3 = tail
                    if isinstance(e1, list):
                        get_data_struct, _ = evaluate_program([e1], sig, l=l)
                    else:
                        # Most likely a pre-defined varibale in l
                        get_data_struct = l[e1]
                    # Get index
                    if isinstance(e2, list):
                        e2_idx, _ = evaluate_program([e2], sig, l=l)
                    elif isinstance(e2, float) or isinstance(e2, int):
                        e2_idx = int(e2)
                    else:
                        # Most likely a pre-defined varibale in l
                        e2_idx = l[e2]
                    # Get Value
                    if isinstance(e3, list):
                        e3_val, _ = evaluate_program([e3], sig, l=l)
                    elif isinstance(e3, float) or isinstance(e3, int):
                        e3_val = e3
                    else:
                        # Most likely a pre-defined varibale in l
                        e3_val = l[e3]
                    if DEBUG:
                        print('Data : ', get_data_struct)
                        print('Index: ', e2_idx)
                        print('Value: ', e3_val)
                    return [op_func(get_data_struct, e2_idx, e3_val), sig]
                # ['remove'/'get', ['vector', 2, 3, 4, 5], 2]
                elif root == 'remove' or root == 'get':
                    # import pdb; pdb.set_trace()
                    e1, e2 = tail
                    if isinstance(e1, list):
                        get_data_struct, _ = evaluate_program([e1], sig, l=l)
                    else:
                        # Most likely a pre-defined varibale in l
                        get_data_struct = l[e1]
                    if isinstance(e2, list):
                        e2_idx, _ = evaluate_program([e2], sig, l=l)
                    elif isinstance(e2, float) or isinstance(e2, int):
                        e2_idx = e2
                    else:
                        # Otherwise Most likely a pre-defined varibale in l
                        e2_idx = l[e2]
                    # Convert index to type-int
                    if torch.is_tensor(e2_idx):
                        e2_idx = e2_idx.long()
                    else:
                        e2_idx = int(e2_idx)
                    if DEBUG:
                        print('Data : ', get_data_struct)
                        print('Index/Value: ', e2_idx)
                    return [op_func(get_data_struct, e2_idx), sig]
                # ['append', ['vector', 2, 3, 4, 5], 2]
                elif root == 'append':
                    # import pdb; pdb.set_trace()
                    get_list1, get_list2 = tail
                    # Evalute exp1
                    if isinstance(get_list1, list):
                        get_data_eval_1, _ = evaluate_program([get_list1], sig, l=l)
                    elif isinstance(get_list1, float) or isinstance(get_list1, int):
                        get_data_eval_1 = get_list1
                    else:
                        get_data_eval_1 = l[get_list1] # Most likely a pre-defined varibale in l
                    if DEBUG:
                        print('Op Eval-1: ', get_data_eval_1)
                    # Evalute exp2
                    if isinstance(get_list2, list):
                        get_data_eval_2, _ = evaluate_program([get_list2], sig, l=l)
                    elif isinstance(get_list2, float) or isinstance(get_list2, int):
                        get_data_eval_2 = get_list2
                    else:
                        get_data_eval_2 = l[get_list2] # Most likely a pre-defined varibale in l
                    if DEBUG:
                        print('Op Eval-2: ', get_data_eval_2)
                    # Check if not torch tensor
                    if not torch.is_tensor(get_data_eval_1):
                        if isinstance(get_data_eval_1, list):
                            get_data_eval_1 = torch.tensor(get_data_eval_1, dtype=torch.float32)
                        else:
                            get_data_eval_1 = torch.tensor([get_data_eval_1], dtype=torch.float32)
                    # Check for 0 dimensional tensor
                    elif get_data_eval_1.shape == torch.Size([]):
                        get_data_eval_1 = torch.tensor([get_data_eval_1.item()], dtype=torch.float32)
                    # Check if not torch tensor
                    if not torch.is_tensor(get_data_eval_2):
                        if isinstance(get_data_eval_2, list):
                            get_data_eval_2 = torch.tensor(get_data_eval_2, dtype=torch.float32)
                        else:
                            get_data_eval_2 = torch.tensor([get_data_eval_2], dtype=torch.float32)
                    # Check for 0 dimensional tensor
                    elif get_data_eval_2.shape == torch.Size([]):
                        get_data_eval_2 = torch.tensor([get_data_eval_2.item()], dtype=torch.float32)
                    # Append
                    try:
                        all_data_eval = torch.cat((get_data_eval_1, get_data_eval_2))
                    except:
                        raise AssertionError('Cannot append the torch tensors')
                    if DEBUG:
                        print('Appended Data : ', all_data_eval)
                    return [all_data_eval, sig]
                else:
                    # ['First'/'last'/'rest', ['vector', 2, 3, 4, 5]]
                    e1 = tail
                    if isinstance(e1, list):
                        get_data_struct, _ = evaluate_program(e1, sig, l=l)
                    else:
                        # Most likely a pre-defined varibale in l
                        get_data_struct = l[e1]
                    if DEBUG:
                        print('Data : ', get_data_struct)
                    return [op_func(get_data_struct), sig]
            # Conditionals
            elif root in cond_ops.keys():
                # (< a b)
                op_func = cond_ops[root]
                if DEBUG:
                    print('Conditional param-1: ', tail[0])
                    print('Conditional param-2: ', tail[1])
                a = evaluate_program([tail[0]], sig, l=l)
                b = evaluate_program([tail[1]], sig, l=l)
                # In case of functions returning only a single value & not sigma
                try:
                    a = a[0]
                except:
                    pass
                try:
                    b = b[0]
                except:
                    pass
                # If torch tensors convert to python data types for comparison
                if torch.is_tensor(a):
                    a = a.tolist()
                    if isinstance(a, list):
                        a = a[0]
                if torch.is_tensor(b):
                    b = b.tolist()
                    if isinstance(b, list):
                        b = b[0]
                if DEBUG:
                    print('Eval Conditional param-1: ', a)
                    print('Eval Conditional param-2: ', b)
                return [op_func(a, b), sig]
            # Assign
            elif root == 'let':
                # (let [params] body)
                let_param_name = tail[0][0]
                let_param_value = tail[0][1]
                let_body = tail[1]
                if DEBUG:
                    print('Let param name: ', let_param_name)
                    print('Let params value: ', let_param_value)
                    print('Let body: ', let_body)
                # Evaluate params
                let_param_value_eval, _ = evaluate_program([let_param_value], sig, l=l)
                # Add to local variables
                l[let_param_name] = let_param_value_eval
                # Check for single instance string
                if isinstance(let_body, str):
                    let_body = [let_body]
                if DEBUG:
                    print('Local Params :  ', l)
                    print('Recursive Body: ', let_body, "\n")
                # Evaluate body
                return evaluate_program([let_body], sig, l=l)
            # Assign
            elif root == "if":
                # (if e1 e2 e3)
                if DEBUG:
                    print('Conditonal Expr1 :  ', tail[0])
                    print('Conditonal Expr2 :  ', tail[1])
                    print('Conditonal Expr3 :  ', tail[2])
                e1_, sig = evaluate_program([tail[0]], sig, l=l)
                if DEBUG:
                    print('Conditonal eval :  ', e1_)
                if e1_:
                    return evaluate_program([tail[1]], sig, l=l)
                else:
                    return evaluate_program([tail[2]], sig, l=l)
            # Functions
            elif root == "defn":
                # (defn name[param] body, )
                if DEBUG:
                    print('Defn Tail: ', tail)
                try:
                    fnname   = tail[0]
                    fnparams = tail[1]
                    fnbody   = tail[2]
                except:
                    raise AssertionError('Failed to define function!')
                if DEBUG:
                    print('Function Name : ', fnname)
                    print('Function Param: ', fnparams)
                    print('Function Body : ', fnbody)
                # Check if already present
                if fnname in rho.keys():
                    return [fnname, None]
                else:
                    # Define functions
                    rho[fnname] = [fnparams, fnbody]
                    if DEBUG:
                        print('Local Params : ', l)
                        print('Global Funcs : ', rho, "\n")
                    return [fnname, None]
            # Get distribution
            elif root in dist_ops.keys():
                op_func = dist_ops[root]
                if len(tail) == 2:
                    # Check for single referenced string
                    if isinstance(tail[0], str):
                        param1 = [tail[0]]
                    else:
                        param1 = tail[0]
                    if isinstance(tail[1], str):
                        param2 = [tail[1]]
                    else:
                        param2 = tail[1]
                    if DEBUG:
                        print('Sampler Parameter-1: ', param1)
                        print('Sampler Parameter-2: ', param2)
                    # Eval params
                    para1 = evaluate_program([param1], sig, l=l)[0]
                    para2 = evaluate_program([param2], sig, l=l)[0]
                    # Make sure to have it in torch tensor
                    para1 = _totensor(x=para1)
                    para2 = _totensor(x=para2)
                    if DEBUG:
                        print('Eval Sampler Parameter-1: ', para1)
                        print('Eval Sampler Parameter-2: ', para2, "\n")
                    return [op_func(para1, para2), sig]
                else:
                    # Exponential has only one parameter
                    # Check for single referenced string
                    if isinstance(tail[0], str):
                        param1 = [tail[0]]
                    else:
                        param1 = tail[0]
                    if DEBUG:
                        print('Sampler Parameter-1: ', param1)
                    para1 = evaluate_program([param1], sig, l=l)[0]
                    if DEBUG:
                        print('Eval Sampler Parameter-1: ', para1)
                    # Make sure to have it in torch tensor
                    para1 = _totensor(x=para1)
                    if DEBUG:
                        print('Tensor Sampler Parameter-1: ', para1, "\n")
                    return [op_func(para1), sig]
            # Sample
            elif root == 'sample':
                if DEBUG:
                    print('Sampler program: ', tail)
                sampler = evaluate_program(tail, sig, l=l)[0]
                if DEBUG:
                    print('Sampler: ', sampler)
                # IF sample object then take a sample
                try:
                    sampler_ = sampler.sample()
                except:
                    sampler_ = sampler
                return [sampler_, sig]
            # Observe
            elif root == 'observe':
                if len(tail) == 2:
                    # Check for single referenced string
                    if isinstance(tail[0], str):
                        ob_pm1 = [tail[0]]
                    else:
                        ob_pm1 = tail[0]
                    if isinstance(tail[1], str):
                        ob_pm2 = [tail[1]]
                    else:
                        ob_pm2 = tail[1]
                else:
                    raise AssertionError('Unknown list of observe params!')
                if DEBUG:
                    print('Observe Param-1: ', ob_pm1)
                    print('Observe Param-2: ', ob_pm2)
                # Evaluate observe params
                dist, sig  = evaluate_program([ob_pm1], sig, l=l)
                value, sig = evaluate_program([ob_pm2], sig, l=l)
                value = _totensor(x=value)
                if DEBUG:
                    print('Observed Value: ', value, "\n")
                return [value, sig]
            # Most likely a single element list or function name
            else:
                if DEBUG:
                    print('End case Root Value: ', root)
                    print('End case Tail Value: ', tail)
                # Check in local vars
                if root in l.keys():
                    return [l[root], sig]
                # Check in Functions vars
                elif root in rho.keys():
                    # import pdb; pdb.set_trace()
                    fnparams_ = {**l}
                    fnparams, fnbody =rho[root]
                    if len(tail) != len(fnparams):
                        raise AssertionError('Function params mis-match!')
                    else:
                        for k in range(len(tail)):
                            fnparams_[fnparams[k]] = evaluate_program([tail[k]], sig, l=l)[0]
                    if DEBUG:
                        print('Function Params :', fnparams_)
                        print('Function Body :', fnbody)
                    # Evalute function body
                    eval_output = [evaluate_program([fnbody], sig, l=fnparams_)[0], sig]
                    if DEBUG:
                        print('Function evaluation output: ', eval_output)
                    return eval_output
                else:
                    return [root, sig]
        except:
            # Just a single element
            return [ast, sig]
    else:
        # Parse functions
        for func in range(len(ast)-1):
            if DEBUG:
                print('Function: ', ast[func])
            fname, _ = evaluate_program([ast[func]], sig=None, l={})
            if DEBUG:
                print('Parsed function: ', fname)
                print("\n")
        # Evaluate Expression
        try:
            outputs_, sig = evaluate_program([ast[-1]], sig=None, l={})
        except:
            raise AssertionError('Failed to evaluate expression!')
        if DEBUG:
            print('Final output: ', outputs_)
        # Return
        return [outputs_, sig]

    return [None, sig]


def get_stream(ast):
    """
    Return a stream of prior samples
    """
    while True:
        yield evaluate_program(ast)[0]


#------------------------------Test Functions --------------------------------#
def run_deterministic_tests():
    for i in range(1,14):
        # Note: this path should be with respect to the daphne path!
        # ast = daphne([f'desugar', '-i', f'{daphne_path}/src/programs/tests/deterministic/test_{i}.daphne'])
        #
        # ast_path = f'./jsons/tests/deterministic/test_{i}.json'
        # with open(ast_path, 'w') as fout:
        #     json.dump(ast, fout, indent=2)

        ast_path = f'./jsons/tests/deterministic/test_{i}.json'
        with open(ast_path) as json_file:
            ast = json.load(json_file)
        # print(ast)

        ret, sig = evaluate_program(ast)
        print('Running evaluation-based-sampling for deterministic test number {}:'.format(str(i)))
        truth = load_truth('./programs/tests/deterministic/test_{}.truth'.format(i))
        print("Evaluation Output: ", ret)
        print("Ground Truth: ", truth)
        try:
            assert(is_tol(ret, truth))
        except AssertionError:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,ast))

        print('Test passed \n')

    print('All deterministic tests passed.')


def run_probabilistic_tests():

    num_samples=1e4
    #num_samples=10
    max_p_value =1e-4

    for i in range(1,7):
    #for i in range(6,7):
        # Note: this path should be with respect to the daphne path!
        # ast = daphne(['desugar', '-i', f'{daphne_path}/src/programs/tests/probabilistic/test_{i}.daphne'])
        # ast_path = f'./jsons/tests/probabilistic/test_{i}.json'
        # with open(ast_path, 'w') as fout:
        #     json.dump(ast, fout, indent=2)

        ast_path = f'./jsons/tests/probabilistic/test_{i}.json'
        with open(ast_path) as json_file:
            ast = json.load(json_file)
        # print(ast)

        stream = get_stream(ast)

        # samples = []
        # for k in range(1):
        #     samples.append(next(stream))
        # print(samples)

        print('Running evaluation-based-sampling for probabilistic test number {}:'.format(str(i)))

        truth = load_truth('./programs/tests/probabilistic/test_{}.truth'.format(i))
        p_val = run_prob_test(stream, truth, num_samples)

        # Empty globals funcs
        rho = {}

        assert(p_val > max_p_value)
        print('P-Value: ', p_val)
        print('Test passed \n')

    print('All probabilistic tests passed.')

#------------------------------MAIN--------------------------------------------
if __name__ == '__main__':
    # Change the path
    daphne_path = '/Users/tony/Documents/prog-prob/CS539-HW-2'

    # Uncomment the appropriate tests to run
    # Deterministic Test
    # run_deterministic_tests()

    # Probabilistic Test
    # run_probabilistic_tests()

    #for i in range(1,5):
    for i in range(4,5):
        # Note: this path should be with respect to the daphne path!
        # ast = daphne(['desugar', '-i', f'{daphne_path}/src/programs/{i}.daphne'])
        # ast_path = f'./jsons/tests/final/{i}.json'
        # with open(ast_path, 'w') as fout:
        #     json.dump(ast, fout, indent=2)
        # print('\n\n\nSample of prior of program {}:'.format(i))

        if i == 1:
            print('Running evaluation-based-sampling for Task number {}:'.format(str(i+1)))
            ast_path = f'./jsons/tests/final/{i}.json'
            with open(ast_path) as json_file:
                ast = json.load(json_file)
            # print(ast)

            print("Single Run Evaluation: ")
            ret, sig = evaluate_program(ast)
            print("Evaluation Output: ", ret)
            print("\n")

            print("Expectation: ")
            stream = get_stream(ast)
            samples = []
            for k in range(1000):
                samples.append(next(stream))

            # print(samples)
            all_samples = torch.tensor(samples)

            # print("Evaluation Output: ", all_samples)
            print("Mean of 1000 samples: ", torch.mean(all_samples))
            print("\n")

            # Empty globals funcs
            rho = {}

        elif i == 2:
            print('Running evaluation-based-sampling for Task number {}:'.format(str(i+1)))
            ast_path = f'./jsons/tests/final/{i}.json'
            with open(ast_path) as json_file:
                ast = json.load(json_file)
            # print(ast)
            # print(len(ast))

            print("Single Run Evaluation: ")
            ret, sig = evaluate_program(ast)
            print("Evaluation Output: ", ret)
            print("\n")

            print("Expectation: ")
            stream = get_stream(ast)
            samples = []
            for k in range(1000):
                if k == 0:
                    samples = next(stream)
                    samples = samples.unsqueeze(0)
                    print(samples.shape)
                else:
                    sample  = next(stream)
                    sample  = sample.unsqueeze(0)
                    samples = torch.cat((samples, sample), dim=0)
            print("Evaluation Output: ", samples.shape)
            print("Mean of 1000 samples: ", torch.mean(samples, dim=0))
            print("\n")

            # print(samples)
            fig, (ax1, ax2) = plt.subplots(2, 1)
            ax1.hist([a[0] for a in samples])
            ax2.hist([a[1] for a in samples])
            plt.savefig(f'plots/2.png')

            # Empty globals funcs
            rho = {}

        elif i == 3:
            print('Running evaluation-based-sampling for Task number {}:'.format(str(i+1)))
            ast_path = f'./jsons/tests/final/{i}.json'
            with open(ast_path) as json_file:
                ast = json.load(json_file)
            # print(ast)
            # print("Single Run Evaluation: ")
            # ret, sig = evaluate_program(ast)
            # print("Evaluation Output: ", ret)
            # print("\n")

            print("Expectation: ")
            stream  = get_stream(ast)
            samples = []
            for k in range(1000):
                if k == 0:
                    samples = next(stream)
                    samples = samples.unsqueeze(0)
                    # print(samples.shape)
                else:
                    sample  = next(stream)
                    sample  = sample.unsqueeze(0)
                    samples = torch.cat((samples, sample), dim=0)
            # print(samples)
            print("Evaluation Output: ", samples.shape)
            print("Mean of 1000 samples for each HMM step: \n", torch.mean(samples, dim=0))
            print("\n")

            fig, axs = plt.subplots(3,6)
            png = [axs[i//6,i%6].hist([a[i] for a in samples]) for i in range(17)]
            plt.tight_layout()
            plt.savefig(f'plots/p3.png')

            # Empty globals funcs
            rho = {}

        elif i == 4:
            print('Running evaluation-based-sampling for Task number {}:'.format(str(i+1)))
            ast_path = f'./jsons/tests/final/{i}.json'
            with open(ast_path) as json_file:
                ast = json.load(json_file)
            # print(ast)

            # print("Single Run Evaluation: ")
            # ret, sig = evaluate_program(ast)
            # print("Evaluation Output: ", ret.shape)
            # print("\n")

            print("Expectation: ")
            stream  = get_stream(ast)
            samples = []
            for k in range(1000):
                if k == 0:
                    samples = next(stream)
                    samples = samples.unsqueeze(0)
                    # print(samples.shape)
                else:
                    sample  = next(stream)
                    sample  = sample.unsqueeze(0)
                    samples = torch.cat((samples, sample), dim=0)
            # print(samples)
            print("Evaluation Output: ", samples.shape)
            W_0 = samples[:, 0:10]
            b_0 = samples[:, 10:20]
            W_1 = samples[:, 20:120]
            b_1 = samples[:, 120:]

            print("W_0: ", W_0.shape)
            print("b_0: ", b_0.shape)
            print("W_1: ", W_1.shape)
            print("b_1: ", b_1.shape)
            print("Mean of 1000 samples for each Neural Network weight(s): \n", torch.mean(samples, dim=0))
            print("\n")

            # Empty globals funcs
            rho = {}
#-------------------------------------------------------------------------------
