import time
import numpy as np

# Profiler decorator

PROF_DATA = {}


def profile(fn):
    def with_profiler(*args, **kwargs):

        start_time = time.time()

        ret = fn(*args, **kwargs)

        elapsed_time = time.time() - start_time

        if fn.__name__ not in PROF_DATA:
            PROF_DATA[fn.__name__] = [1, [elapsed_time]]
        else:
            PROF_DATA[fn.__name__][0] += 1
            PROF_DATA[fn.__name__][1].append(elapsed_time)

        return ret
    return with_profiler


def print_data():
    for fn_name, data in PROF_DATA.items():
        max_time = max(data[1])
        max_place = data[1].index(max_time)
        avg_time = (sum(data[1]) - data[1][0])/(data[0] - 1)
        print("Max execution time for function %s was %0.3f s at %d." % (fn_name, max_time, max_place))
        print("Average execution time: %0.5f s." % avg_time)


def write_file(message, file_name):
    with open(file_name, 'a') as f:
        for fn_name, data in PROF_DATA.items():
            max_time = max(data[1])
            max_place = data[1].index(max_time)
            avg_time = (sum(data[1]) - data[1][0]) / (data[0] - 1)
            f.write(message + "\n")
            f.write("Max execution time for function %s was %0.3f s at %d.\n" % (fn_name, max_time, max_place))
            f.write("Average execution time: %0.5f s.\n" % avg_time)


def print_data2():
    for fn_name, data in PROF_DATA.items():
        print(fn_name)
        for d in data[1]:
            print(d)
        print("------")


def clear_data():
    global PROF_DATA
    PROF_DATA = {}
    
# Data generators

def generate_random_data(input_size):  # input_size = [1, h, w, ch]
    return np.float32(np.random.random(input_size) * 255.0)

def generate_data_KERAS(shape):
    def generator():
        return generate_random_data(shape)

    return generator


def generate_data_TF(var, shape):
    def generator():
        return {var: generate_random_data(shape)}

    return generator


def generate_data_CNTK(var, shape):
    def generator():
        return {var: generate_random_data(shape)}

    return generator