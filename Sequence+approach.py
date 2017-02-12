
# coding: utf-8

# In[3]:

import array

import random
import numpy as np

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

import operator
import math

import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout

import networkx as nx


# In[4]:

def draw_tree(expr):
    nodes, edges, labels = gp.graph(expr)
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    pos = graphviz_layout(g, prog="dot")

    nx.draw_networkx_nodes(g, pos)
    nx.draw_networkx_edges(g, pos)
    nx.draw_networkx_labels(g, pos, labels)
    plt.show()


# In[5]:


class RegexObject(object):
    index = 0
    def __init__(self):
        self.index = RegexObject.index
        RegexObject.index += 1
        self.next = None
        self.value = ""

    def add_next(self, next_object):
        self.next = next_object
        return self

    @staticmethod
    def create():
        return RegexObject()

    @staticmethod
    def create_with_value(value):
        regex = RegexObject()
        regex.value = value
        return regex

    @property
    def regex(self):
        result = self.value
        if self.next:
            result += self.next.regex
        return result


class Factory(object):
    def __init__(self, value):
        self.value = ""

    @staticmethod
    def create_value(value):
        return lambda x:RegexObject.create_with_value(value).add_next(x)

    @staticmethod
    def create_group():
        return lambda x, y:Group().add_child(x, y)


class Group(RegexObject):
    def __init__(self):
        self.child = None

    def add_child(self, child, next_object):
        self.child = child
        return self.add_next(next_object)

    @property
    def group_value(self):
        return

    @property
    def regex(self):
        result = "(" + self.child.regex + ")+"
        if self.next:
            result += self.next.regex
        return result



# In[6]:

pset = gp.PrimitiveSetTyped("main", [], RegexObject)

pset.addPrimitive(Factory.create_value("A"), [RegexObject], RegexObject, "RegexObject_A")
pset.addPrimitive(Factory.create_value("B"), [RegexObject], RegexObject, "RegexObject_B")
pset.addPrimitive(Factory.create_value("C"), [RegexObject], RegexObject, "RegexObject_C")
pset.addPrimitive(Factory.create_value("D"), [RegexObject], RegexObject, "RegexObject_D")
pset.addPrimitive(Factory.create_value("E"), [RegexObject], RegexObject, "RegexObject_E")

pset.addPrimitive(Factory.create_group(), [RegexObject, RegexObject], RegexObject, "Group")

pset.addTerminal(RegexObject.create(), RegexObject, "RegexEnd")


# In[11]:

creator.create("Fitness", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.Fitness)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genFull, pset=pset, min_=5, max_=10)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


# In[12]:

expr = toolbox.individual()

draw_tree(expr)


# In[13]:

tree = gp.PrimitiveTree(expr)
print gp.compile(tree, pset).regex


# In[1]:

PERFECT_SCALE = 1.0

import re, sys
items = ["AADDEE", "AADDEE", "ADEEE", "AAADDE", "AAAAAADEEEE", "AAAADDDDEE", "ADDDDDEEEE", "AAADDDDE"]
def evaluate(individual):
    tree = gp.PrimitiveTree(individual)
    regex = "({})".format(gp.compile(tree, pset).regex)
    value = 0
    for item in items:
        try:
            print regex
            best = re.findall(regex, item)[-1]
            current_value = len(best) / float(len(item))
            print current_value
            print '-'*10
            if current_value == 1.0:
                current_value *= PERFECT_SCALE
            value += current_value

        except:
            continue
    return value,


# In[2]:

toolbox.register("evaluate", evaluate)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=1, max_=5)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
stats_size = tools.Statistics(len)
mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
mstats.register("avg", np.mean)
mstats.register("std", np.std)
mstats.register("min", np.min)
mstats.register("max", np.max)


# In[ ]:

NGEN = 50
pop = toolbox.population(n=100)
hof = tools.HallOfFame(10)
pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.1, ngen=NGEN, stats=mstats,
                               halloffame=hof, verbose=True)


# In[ ]:

for item in hof.items:
    tree = gp.PrimitiveTree(item)
    print gp.compile(tree, pset).regex, toolbox.evaluate(tree)


# In[668]:


items = ["AADDEE", "AADDEE", "ADEEE", "AAADDE", "AAAAAADEEEE", "AAAADDDDEE", "ADDDDDEEEE", "AAADDDDE"]


# In[640]:

for item in items:
    best = re.findall("(AD*E*)", item)[-1]
    print len(best) / float(len(item))


# In[ ]:




# In[ ]:



