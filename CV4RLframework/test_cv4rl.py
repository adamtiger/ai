from cv4rl.cv4pool import pool
from cv4rl.cv4env import BaseEnvironment as benv
from scipy import misc
import numpy as np

# Testing the ImagePool.

def __init_pool_tests():
    return pool.ImagePool(5, 6, "vTestImages")
    
def test_cv4pool_add():
    img_pool = __init_pool_tests()
    
    for i in range(0, 100):
        img_pool._add(i)
        
    assert getattr(img_pool, 'images')[0] == 95, 'images should be contain 95 at position 0 instead of %r' % getattr(img_pool, 'images')[0]
    assert getattr(img_pool, 'images')[1] == 96, 'images should be contain 96 at position 0 instead of %r' % getattr(img_pool, 'images')[1]
    assert getattr(img_pool, 'images')[2] == 97, 'images should be contain 97 at position 0 instead of %r' % getattr(img_pool, 'images')[2]
    assert getattr(img_pool, 'images')[3] == 98, 'images should be contain 98 at position 0 instead of %r' % getattr(img_pool, 'images')[3]
    assert getattr(img_pool, 'images')[4] == 99, 'images should be contain 99 at position 0 instead of %r' % getattr(img_pool, 'images')[4]
    
    print "The test_c4pool_add was successful!"
    
def test_cv4pool_get_pixel_base():
    img_pool = __init_pool_tests()
    img = img_pool.next_image()
    
    assert img.get_pixel_base(1,1).size == 3, 'Base picture is not an RGB format.'
    
    print "The test_c4pool_get_pixel_base was successful!"
    
def test_cv4pool():
    test_cv4pool_add()
    test_cv4pool_get_pixel_base()

# Testing the Environment.

def __init_env_tests():
    return benv.BaseEnvironment(2, 5, 100, 200, "vTestImages") 

def __map2img(path, pth):
    for i in range(0, len(pth)):
        path[pth[i][0], pth[i][1]] = 250.0
        
def __check_generated_curve(sgm_img, pth):
    correct = True
    for idx in range(0, len(pth)):
        if (int(sgm_img[pth[idx][0], pth[idx][1]]) < 200.0):
            correct = False
    return correct
    
def test_environment():
    env = __init_env_tests()
    img = getattr(env, 'image')
    sgm_img = getattr(img, 'sgm_img')
    path = np.zeros(sgm_img.shape)
    
    env.generate_new_situation()
    pth = env.get_correct()
    __map2img(path, pth)
    env.generate_new_situation()
    pth = env.get_correct()
    __map2img(path, pth)
    env.generate_new_situation()
    pth = env.get_correct()
    __map2img(path, pth)
    misc.imsave('path.png', path)
    assert __check_generated_curve(sgm_img, pth) == True, 'Wrong generated path.'
    
    print "The test_environment was successful!"

def run_all_tests():
    
    test_cv4pool()
    test_environment()
    
    print "All tests were successful."
    
# RUN THE TESTS

run_all_tests()
