from preproc import PreProcessing as  prep 
import gym

# Creating variables to use it anywhere

env = gym.make("Breakout-v0")



# Test the preprocessing functions:

def _init_preproc_test():
    return prep(env)

def test_preprocessing():
    
    p = _init_preproc_test()
    o,r,d,i = env.step(1)
    state = p.preprocessing(o)

    print ("test_preprocessing: successful")


def test_preproc():
    test_preprocessing()


# Test the A3C class:


# Test the NN class:




# ------------------------------------

def run_all():
    
    test_preproc()
    print( "All tests were SUCCESSFUL.")


# RUN TESTS

run_all()

