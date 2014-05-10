__author__ = 'yoyomyo'

import pdb

class HoughArray:

    def __init__(self, neighborhood):
        self.data = []  # data points to vote for
        self.vote = []  # vote count for each data
        self.size = 0   # number of data
        self.neighborhood = neighborhood  #a point may fall into the neighborhood around a data entry

    def add(self, toAdd):
        # if toAdd already belongs to an entry's neighborhood,
        # increment the vote of that entry
        # else create data's own entry and insert it
        if self.size == 0:
            self.data.append(toAdd)
            self.vote.append(1)
            self.size = 1
        else:
            # self.__add_helper__(toAdd, 0, self.size-1)

            # find the data entry closest to toAdd
            diffs = [abs(toAdd-x) for x in self.data]
            min_diff, min_idx = min((val, idx) for (idx, val) in enumerate(diffs))

            if min_diff < self.neighborhood:
                self.vote[min_idx] += 1
            else:
                # create a new data entry
                self.data.append(toAdd)
                self.vote.append(1)
                self.size += 1

    def get_high_votes(self):
        # return a subset of data that received a lot of votes
        # the threshold is 0.5 * (total vote/size)
        total_vote = sum(self.vote)
        filter_result =  filter(lambda x: x[0] >= 0.5* total_vote / (self.size+0.0), zip(self.vote, self.data))
        return map(lambda x: x[1], filter_result)

# ha = HoughArray(10)
# ha.__add__(20)
# ha.__add__(15)
# ha.__add__(30)
# ha.__add__(50)
# pdb.set_trace()
