__author__ = 'yoyomyo'

import pdb

class HoughArray:

    def __init__(self, neighborhood):
        self.data = []  # a sorted array to keep track of data
        self.vote = []  # vote correspond to each data
        self.size = 0   # number of data
        self.neighborhood = neighborhood  #the point can vote for a neighborhood

    def __add__(self, toAdd):
        # if toAdd already belongs to an entry's neighborhood,
        # then increment the vote of that entry
        # else create data's own entry and insert it at the right index
        # to maintain the requirement that arrays are sorted
        if self.size == 0:
            self.data.append(toAdd)
            self.vote.append(1)
            self.size = 1
        else:
            # self.__add_helper__(toAdd, 0, self.size-1)

            diffs = [abs(toAdd-x) for x in self.data]
            min_diff, min_idx = min((val, idx) for (idx, val) in enumerate(diffs))

            if min_diff < self.neighborhood:
                self.vote[min_idx] += 1
            else:
                # either add to max(0, min_idx -1)
                # or add to min(self.size, min_idx + 1)
                if toAdd < self.data[min_idx]:
                    pos = max(0, min_idx -1)
                    self.data.insert(pos, toAdd)
                    self.vote.insert(pos, 1)

                else:
                    pos = min(self.size, min_idx + 1)
                    self.data.insert(pos, toAdd)
                    self.vote.insert(pos, 1)
                self.size += 1
            # print self.size
            # print self.data
            # print self.vote


    # def __add_helper__(self, toAdd, left, right):
    #     if left > right:
    #         # print left, right
    #         # did not find a place to insert toAdd
    #         # need to insert toAdd as its own entry
    #
    #         # for some reason add overflows
    #         # trying to access index in list overflows
    #         # how to prevent it?
    #         # if the smaller is less than the beginning
    #         # if the larger is greater than the end of the list
    #
    #         if right < 0:
    #             self.data.insert(0, toAdd)
    #             self.vote.insert(0, 1)
    #         elif left >= len(self.data):
    #             self.data.append(toAdd)
    #             self.vote.append(1)
    #
    #         # ..., right, left,...
    #
    #         elif toAdd < self.data[right]:
    #             self.data.insert(right-1, toAdd)
    #             self.vote.insert(right-1, 1)
    #         elif toAdd > self.data[right] and toAdd < self.data[left]:
    #             self.data.insert(right+1, toAdd)
    #             self.vote.insert(right+1, 1)
    #         else:
    #             self.data.insert(left+1, toAdd)
    #             self.vote.insert(left+1, 1)
    #
    #         # increment size when a new entry is added to data
    #         self.size += 1
    #         return
    #
    #     middle = left+right
    #     if self.__within_entry_neighborhood(toAdd, middle):
    #         self.vote[middle] += 1
    #         return
    #
    #     if toAdd < self.data[middle]:
    #         self.__add_helper__(toAdd, left, middle -1)
    #     else:
    #         self.__add_helper__(toAdd, middle+1, right)
    #
    # def __within_entry_neighborhood(self, toAdd, i):
    #     entry = self.data[i]
    #     left = max(0, entry - self.neighborhood)
    #     right = entry + self.neighborhood
    #
    #     if toAdd > left and toAdd < right:
    #         return True
    #
    #     return False


ha = HoughArray(10)
ha.__add__(20)
ha.__add__(15)
ha.__add__(30)
ha.__add__(50)
pdb.set_trace()
