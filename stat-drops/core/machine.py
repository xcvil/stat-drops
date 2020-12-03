import random
import numpy as np


# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


# Definition for linked list and its append method
class LinkedList:
    def __init__(self):
        self.head = None
        self.tail = None

    def is_empty(self):
        return self.head is None

    def append(self, data):
        node = ListNode(data)
        if self.head is None:
            self.head = node
            self.tail = node
        else:
            self.tail.next = node
            self.tail = node


class Sampling:
    def __init__(self, head: ListNode):
        """
        @param head The linked list's head.
        Note that the head is guaranteed to be not null, so it contains at least one node.
        """
        self.head = head

    def getRandom(self) -> int:
        """
        Returns a random node's value.
        """
        ans = cnt = 0
        head = self.head
        while head:
            if random.randint(0, cnt) == 0:
                ans = head.val
            head, cnt = head.next, cnt+1
        return ans


def population_creator(diameter_class, num_drops):

    population = []
    for i in range(len(diameter_class)):
        n = 0
        while n < num_drops[i]:
            population.append(diameter_class[i])
            n += 1

    return population


def observation_generator(population, num_obervations):
    return np.random.choice(population, num_obervations, replace=True)
