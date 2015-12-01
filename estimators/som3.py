from __future__ import division

import numpy as np
import sys

class SOM:

    def __init__(self, height=10, width=10, FV_size=10, learning_rate=0.005):
        self.height = height
        self.width = width
        self.FV_size = FV_size
        self.radius = (height+width)/3
        self.learning_rate = learning_rate
        self.nodes = np.array([[ [np.random.random()*255 for i in range(FV_size)] for x in range(width)] for y in range(height)])

    # train_vector: [ FV0, FV1, FV2, ...] -> [ [...], [...], [...], ...]
    # train vector may be a list, will be converted to a list of scipy arrays
    def train(self, train_vector, iterations=1000):
        for t in range(len(train_vector)):
            train_vector[t] = np.array(train_vector[t])
        time_constant = iterations/np.log(self.radius)
        delta_nodes = np.array([[[0 for i in range(self.FV_size)] for x in range(self.width)] for y in range(self.height)])
        
        for i in range(1, iterations+1):
            delta_nodes.fill(0)
            radius_decaying = self.radius*np.exp(-1.0*i/time_constant)
            rad_div_val = 2 * radius_decaying * i
            learning_rate_decaying=self.learning_rate*np.exp(-1.0*i/time_constant)
            sys.stdout.write("\rTraining Iteration: " + str(i) + "/" + str(iterations))
            
            for j in range(len(train_vector)):
                best = self.best_match(train_vector[j])
                for loc in self.find_neighborhood(best, radius_decaying):
                    influence = np.exp( (-1.0 * (loc[2]**2)) / rad_div_val)
                    inf_lrd = influence*learning_rate_decaying
                    
                    delta_nodes[loc[0],loc[1]] += inf_lrd * (train_vector[j] - self.nodes[loc[0],loc[1]])
            self.nodes += delta_nodes
        sys.stdout.write("\n")
    
    # Returns a list of points which live within 'dist' of 'pt'
    # Uses the Chessboard distance
    # pt is (row, column)
    def find_neighborhood(self, pt, dist):
        min_y = max(int(pt[0] - dist), 0)
        max_y = min(int(pt[0] + dist), self.height)
        min_x = max(int(pt[1] - dist), 0)
        max_x = min(int(pt[1] + dist), self.width)
        neighbors = []
        for y in range(min_y, max_y):
            for x in range(min_x, max_x):
                dist = abs(y-pt[0]) + abs(x-pt[1])
                neighbors.append((y,x,dist))
        return neighbors
    
    # Returns location of best match, uses Euclidean distance
    # target_FV is a scipy array
    def best_match(self, target_FV):
        loc = np.argmin((((self.nodes - target_FV)**2).sum(axis=2))**0.5)
        r = 0
        while loc > self.width:
            loc -= self.width
            r += 1
        c = loc
        return (r, c)

    # returns the Euclidean distance between two Feature Vectors
    # FV_1, FV_2 are scipy arrays
    def FV_distance(self, FV_1, FV_2):
        return (sum((FV_1 - FV_2)**2))**0.5

if __name__ == "__main__":
    print "Initialization..."
    colors = [ [0, 0, 0], [0, 0, 255], [0, 255, 0], [0, 255, 255], [255, 0, 0], [255, 0, 255], [255, 255, 0], [255, 255, 255]]
    width = 32
    height = 32
    color_som = SOM(width,height,3,0.05)
    print "Training colors..."
    color_som.train(colors, 1000)
    try:
        from PIL import Image
        print "Saving Image: sompy_test_colors.png..."
        img = Image.new("RGB", (width, height))
        for r in range(height):
            for c in range(width):
                img.putpixel((c,r), (int(color_som.nodes[r,c,0]), int(color_som.nodes[r,c,1]), int(color_som.nodes[r,c,2])))
        img = img.resize((width*10, height*10),Image.NEAREST)
        img.save("/home/matteo/sompy_test_colors.png")
    except:
        print "Error saving the image, do you have PIL (Python Imaging Library) installed?"