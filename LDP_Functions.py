import random
import math
import numpy as np
from sortedcontainers import SortedList
import time


class GRR:
    def __init__(self, c, r, epsilon, precision):
        self.c = c
        self.r = r
        self.epsilon = epsilon
        self.p = precision
        
        self.domain_size = 2*r*(10 ** precision) + 1
        self.same_prob = (math.exp(epsilon))/(self.domain_size - 1 + math.exp(epsilon))
        
        self.domain = [round(c - r + i*(10**-precision), precision) for i in range(round(2*r*(10**precision))+1)]
        self.sorted_domain =  SortedList(self.domain)
        self.choice_dict = {}
        for val in self.domain:
            self.sorted_domain.remove(val)
            self.choice_dict[val] = list(self.sorted_domain)
            self.sorted_domain.add(val)
        return
    
    def _perturb(self, w):
        
        rand_val = random.uniform(0,1)
        
        
        closest_value = self.sorted_domain.bisect_left(w)

        if closest_value == 0:
            w_transformed = self.sorted_domain[0]
        elif closest_value == len(self.sorted_domain):
            w_transformed = self.sorted_domain[-1]
            closest_value = len(self.sorted_domain)-1
        else:
            before = self.sorted_domain[closest_value - 1]
            after = self.sorted_domain[closest_value]
            if after - w < w - before:
                w_transformed = after
            else:
                w_transformed = before
                closest_value -= 1
        
        if rand_val <= self.same_prob:
            return w
        else:
            return random.choice(self.choice_dict[w_transformed])
        
        return

def LDP_FL(weight, c = 0.0, r = 0.075, epsilon = 1):
    
    rand_val = random.uniform(0, 1)
    
    boundary = ( r*(math.exp(epsilon) + 1) + (weight-c)*(math.exp(epsilon)-1) )/( 2*r*(math.exp(epsilon) + 1) )
    
    if rand_val <= boundary:
        return c + r*( (math.exp(epsilon)+1)/(math.exp(epsilon)-1) )
    else:
        return c - r*( (math.exp(epsilon)+1)/(math.exp(epsilon)-1) )

    
class LDPFL:
    def __init__(self, c, r, epsilon):
        self.c = c
        self.r = r
        self.epsilon = epsilon
    
    def perturb(self, w):
        
        rand_val = random.uniform(0, 1)
    
        boundary = ( self.r*(math.exp(self.epsilon) + 1) + (w-self.c)*(math.exp(self.epsilon)-1) )/( 2*self.r*(math.exp(self.epsilon) + 1) )

        if rand_val <= boundary:
            return self.c + self.r*( (math.exp(self.epsilon)+1)/(math.exp(self.epsilon)-1) )
        else:
            return self.c - self.r*( (math.exp(self.epsilon)+1)/(math.exp(self.epsilon)-1) )
    
    def check(self, w):
        if abs( w - (self.c - self.r*( (math.exp(self.epsilon)+1)/(math.exp(self.epsilon)-1) )) ) <= 0.001:
            return 0
        elif abs( w - (self.c + self.r*( (math.exp(self.epsilon)+1)/(math.exp(self.epsilon)-1) )) ) <= 0.001:
            return 1
        else:
            print("ERROR UNKNOWN INPUT TO CHECK()")

    def estimate(self, value):
        
        temp = float(value[1]) / (value[0] + value[1])
        temp = temp * (2*self.r*(math.exp(self.epsilon) + 1)) - self.r*(math.exp(self.epsilon) + 1)
        temp = temp / (math.exp(self.epsilon) - 1) + self.c
        return temp
    
    def get_domain_size(self):
        return 2*(self.r)

    
class SRR:
    def __init__(self, c, r, epsilon, delta_d, precision, m):
        
        start = time.time()
        
        self.c = round(c, precision)
        self.r = round(r, precision)
        self.epsilon = epsilon
        self.delta_d = int(delta_d)
        self.p = precision
        self.m = m
        
        self.domain_size = 2*r*(10 ** precision) + 1
        self.k = math.exp(epsilon)
        
        self.group_sizes = []
        g1 = ( (2 * self.domain_size)/m - (m-1)*delta_d )/2
        self.group_sizes.append(int(g1)+1)
        for i in range(m-1):
            g1 += delta_d
            self.group_sizes.append(int(g1))
        
        temp_sum = 0
        for j in range(2, m+1):
            temp_sum += (j-1)*self.group_sizes[j-1]
        
        self.alpha_min = (m-1)/( (m-1)*self.domain_size*self.k - (self.k -1)*temp_sum )
        self.alpha_max = self.k * self.alpha_min
        
        self.delta_p = (self.alpha_min * (self.k -1))/(m-1)
        
        self.probs = []
        a1 = self.alpha_min
        for i in range(m):
            self.probs.append(a1)
            a1 += self.delta_p
        self.probs.reverse()
        
        self.domain = SortedList([round(c - r + i*(10**-precision), precision) for i in range(round(2*r*(10**precision))+1)])
#         self.sorted_domain = SortedList(self.domain)
        
        self.groups = []
        curr_idx = 0

        # iterate over number of groups
        for i in range(m):
            # get group size
            group_size = self.group_sizes[i]
            
            self.groups.append([curr_idx, curr_idx+group_size])
            curr_idx += group_size
        
        self.group_probs = [a*b for a, b in zip(self.group_sizes, self.probs)]
        s = sum(self.group_probs)
        if s != 1:
            diff = 1 - s
            if diff > 0:
                self.group_probs[0] += diff
            else:
                self.group_probs[0] -= abs(diff)
        
        self.data = {}
        
        for value in self.domain:
            self.data[value] = sorted(self.domain, key=lambda x: abs(x - value))
            
        end = time.time()
        
        print(f"Time taken for SRR Precomputation : {end-start} seconds.")
        
        return
    
    def find_group_and_sample(self, idx, w):
        
        l_max = self.c - self.r
        r_max = self.c + self.r
        curr_l = w
        curr_r = w
        hit_r = False
        hit_l = False
        
        for i in range(idx):
            temp_l = curr_l
            temp_r = curr_r
            if not hit_r and not hit_l:
                temp_l -= (self.group_sizes[i]/2)*(10**-self.p)
                temp_r += (self.group_sizes[i]/2)*(10**-self.p)
            elif hit_r:
                temp_l -=  (self.group_sizes[i])*(10**-self.p)
            elif hit_l:
                temp_r += (self.group_sizes[i])*(10**-self.p)
            
            if temp_l <= l_max:
                diff = l_max - temp_l
                temp_r += diff
                temp_l = l_max
                hit_l = True
            if temp_r >= r_max:
                diff = temp_r - r_max
                temp_l -= diff
                temp_r = r_max
                hit_r = True
            
            curr_l = temp_l
            curr_r = temp_r
        
        temp_l = curr_l
        temp_r = curr_r
        
        if not hit_r and not hit_l:
            temp_l -= (self.group_sizes[idx]/2)*(10**-self.p)
            temp_r += (self.group_sizes[idx]/2)*(10**-self.p)
        elif hit_r:
            temp_l -=  (self.group_sizes[idx])*(10**-self.p)
        elif hit_l:
            temp_r += (self.group_sizes[idx])*(10**-self.p)
        
        if temp_l < l_max:
            diff = l_max - temp_l
            temp_r += diff
            temp_l = l_max
        if temp_r > r_max:
            diff = temp_r - r_max
            temp_l -= diff
            temp_r = r_max
        
        pos = random.randint(0, self.group_sizes[idx]-1)
        
        size_l = int( (curr_l - temp_l)/(10**-self.p) )
        size_r = int( (temp_r - curr_r)/(10**-self.p) )
        
        if pos > size_l:
            return curr_r + (pos-size_l)*(10**-self.p)
        else:
            return temp_l + (pos)*(10**-self.p)
        
        
        groupl = []
        
        
        while temp_l <= curr_l:
            groupl.append(temp_l)
            temp_l += 10**-self.p
        
        groupr = []
        
        while curr_r <= temp_r:
            groupr.append(curr_r)
            curr_r += 10**-self.p
        
        group = groupl + groupr
        
        return random.choice(group)
    
    
    def perturb(self, w):
        
        closest_value = self.domain.bisect_left(w)

        if closest_value == 0:
            w = self.domain[0]
        elif closest_value == len(self.domain):
            w = self.domain[-1]
            closest_value = len(self.domain)-1
        else:
            before = self.domain[closest_value - 1]
            after = self.domain[closest_value]
            if after - w < w - before:
                w = after
            else:
                w = before
                closest_value -= 1
        
        rand_val = random.uniform(0,1)
        
        temp = self.group_probs[0]
        idx = 0
        while idx < self.m:
            if rand_val <= temp:
                pos = random.randint(self.groups[idx][0], self.groups[idx][1]-1)
                return self.data[w][pos]
            if idx == self.m - 1:
                pos = random.randint(self.groups[idx][0], self.groups[idx][1]-1)
                return self.data[w][pos]
            idx += 1
            temp += self.group_probs[idx]
    
    def _perturb(self, w):
        
        rand_val = random.uniform(0,1)

        # find the closest value to w
        closest_value = self.domain.bisect_left(w)

        if closest_value == 0:
            w_transformed = self.domain[0]
        elif closest_value == len(self.domain):
            w_transformed = self.domain[-1]
            closest_value = len(self.domain)-1
        else:
            before = self.domain[closest_value - 1]
            after = self.domain[closest_value]
            if after - w < w - before:
                w_transformed = after
            else:
                w_transformed = before
                closest_value -= 1
        
        temp = self.group_probs[0]
        idx = 0
        while idx < self.m:
            if rand_val <= temp:
                return self.find_group_and_sample(idx, w_transformed)
            if idx == self.m - 1:
                return self.find_group_and_sample(idx, w_transformed)
            idx += 1
            temp += self.group_probs[idx]
        
    
    def np_perturb(self, w):
        
        # iterate over each weight
        for i in range(len(w)):
            group = sorted(self.domain, key=lambda x: abs(x - w[i]))
        
            rand_val = random.uniform(0,1)
        
            temp = self.group_probs[0]
            idx = 0
            while idx < self.m:
                if rand_val <= temp:
                    pos = random.randint(self.groups[idx][0], self.groups[idx][1]-1)
                    w[i] = group[pos]
                    break
                if idx == self.m - 1:
                    pos = random.randint(self.groups[idx][0], self.groups[idx][1]-1)
                    w[i] = group[pos]
                    break
                idx += 1
                temp += self.group_probs[idx]
        
        return w