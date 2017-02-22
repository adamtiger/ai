from cv4rl.cv4pool import pool as p
import random as r
import numpy as np

class Rect:
    
    """
    
    During path generation a rectangular is generated.
    
    """
    
    def __init__(self, top, left, height, width):
        self.top = top
        self.left = left
        self.height = height
        self.width = width
        
    def get_top(self):
        return self.top
        
    def get_left(self):
        return self.left
    
    def get_height(self):
        return self.height
        
    def get_width(self):
        return self.width


class BaseEnvironment:
    
    """
    All the environments used by the RL agents should inherit
    from this class.
    
    generate_new_situation:
      definition: situation is a curve with a start and target point.
      -> Choose a random white pixel as a start point.
      -> Generate a rectengular around the start point.
      -> Find all the white pixels in the rectengular in a parent-child chain manner.
      -> The first point choosen from the boundary will be a target.
      -> Following the chain from the target to the start point gives the generated curve.
    """
    
    def __init__(self, pool_size, refresh_freq, min_rect_size, max_rect_size, folder):
        assert min_rect_size < max_rect_size and min_rect_size > 0, "Wrong lower and upper bound for the size of rectengular."
        
        # Init the variables for the class.
        self.pool = p.ImagePool(pool_size, refresh_freq, folder)
        self.image = self.pool.next_image()
        self.correct_path = []
        self.max_rect_size = max_rect_size
        self.min_rect_size = min_rect_size
        self.stop = False
        self.rect = Rect(0, 0, 0, 0)
        self.target = 0
        self.start = 0
        self.canvas = 0

    def generate_new_situation(self):
        lngth = self.image.get_white_vec_length()
        st_idx = r.randint(0, lngth-1)
        coord = self.image.get_coord_from_white_vec(st_idx)
        self.stop = False
        self.start = coord
        counter = 0
        max_iter = 5000
        
        # Generate rectangle.
        height = r.randint(self.min_rect_size, self.max_rect_size)
        width  = r.randint(self.min_rect_size, self.max_rect_size)
        self.rect = Rect(coord[0] - height, coord[1] - width, 2*height, 2*width)
        
        # Flood it. BFS like algorithm. Parent-child structure.
        tree = [(coord, np.array([-1, -1]))]
        idx_0 = 0
        idx_1 = 1
        self.canvas = np.zeros((self.rect.get_height(), self.rect.get_width()), dtype=int)
        
        while (not self.stop and counter < max_iter):
            for idx in range(idx_0, idx_1):
                neighbors = self.__neighbors(tree[idx])
                neighbors = self.__check_conditions(neighbors)
                tree = tree + neighbors
            idx_0 = idx_1
            idx_1 = len(tree)
            counter += 1
        
        if counter >= max_iter:
            self.target = tree[len(tree)-1]
        
        # Propagate back the sequence.
        generated_line = [self.target[0]]
        end = False
        current_point = self.target[1]
        while not end:
            next_point = self.__find(tree, current_point)
            assert not np.array_equal(next_point, np.array([-2, -2])), "Wrong tree!"
            if np.array_equal(next_point, np.array([-1, -1])):
                end = True
            generated_line.append(current_point)
            current_point = next_point

        self.correct_path = generated_line
        

    def __neighbors(self, coord):
        n_st = np.array([coord[0][0], coord[0][1]])
        n_1 = (n_st + np.array([-1, -1]), coord[0])
        n_2 = (n_st + np.array([0, -1]), coord[0])
        n_3 = (n_st + np.array([1, -1]), coord[0])
        n_4 = (n_st + np.array([-1, 0]), coord[0])
        n_5 = (n_st + np.array([1, 0]), coord[0])
        n_6 = (n_st + np.array([-1, 1]), coord[0])
        n_7 = (n_st + np.array([0, 1]), coord[0])
        n_8 = (n_st + np.array([1, 1]), coord[0])
                
        return [n_1, n_2, n_3, n_4, n_5, n_6, n_7, n_8]

    def __check_conditions(self, neighbors):
        should_remove = []
        for i in range(0, 8):
            x = neighbors[i][0][0]
            y = neighbors[i][0][1]
            
            if (x < 0.0 or x >= self.image.shape()[0]-1):
                should_remove.append(i)
            elif (y < 0.0 or y >= self.image.shape()[1]-1):
                should_remove.append(i)
            elif (x < self.rect.get_top() or x >= self.rect.get_top() + self.rect.get_height()-1):
                should_remove.append(i)
                self.stop = True
                self.target = neighbors[i]
            elif (y < self.rect.get_left() or y >= self.rect.get_left() + self.rect.get_width()-1):
                should_remove.append(i)
                self.stop = True
                self.target = neighbors[i]
            elif (self.canvas[x - self.rect.get_top(), y - self.rect.get_left()] == 1):
                should_remove.append(i)
            elif (self.image.get_pixel_sgm(x, y) < 200.0):
                should_remove.append(i)
                
        checked_neighbors = []   
        for i in range(0, 8):
            if not(i in should_remove):
                checked_neighbors.append(neighbors[i])
            self.canvas[neighbors[i][0][0] - self.rect.get_top(), neighbors[i][0][1] - self.rect.get_left()] = 1
        return checked_neighbors

    def __find(self, tree, xy):
        for i in range(0, len(tree)):
            if np.array_equal(tree[i][0], xy):
                return tree[i][1]

        return np.array([-2, -2]) 
       
    def update_base_image(self):
        self.image = self.pool.next_image()

    def get_correct(self):
        return self.correct_path
