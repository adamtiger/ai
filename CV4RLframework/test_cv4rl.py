from cv4rl.cv4pool import pool

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
    

def run_all_tests():
    
    test_cv4pool()
    
    
# RUN THE TESTS

run_all_tests()