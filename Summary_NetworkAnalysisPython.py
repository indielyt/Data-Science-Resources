# Datacamp NETWORK ANALYSIS IN PYTHON WITH NETWORKX

# Chapter 1: Introduction to Networks

	# Basics of NetworkX API, using Twitter network
	# Basic drawing of a network using NetworkX
	# Queries on a graph
	# Checking the un/directed status of a graph
	# Specifying a weight on edges
	# Checking whether there are self-loops in the graph
	# Visualizing using Matrix plots
	# Visualizing using Circos plots
	# Visualizing using Arc plots

# Chapter 2: Important Nodes

	# Compute number of neighbors for each node
	# Compute degree distribution
	# Degree centrality distribution
	# Shortest Path I
    # Shortest Path II
    # Shortest Path III
    # NetworkX betweenness centrality on a social network
    # Deep dive - Twitter network

# Chapter 3: Structures

    # Identifying triangle relationships
    # Finding nodes involved in triangles
    # Finding open triangles
    # Finding all maximal cliques of size "n"
    # Subgraphs
    # Subgraphs II






		











############################ Chapter 1: Introduction to Networks

















### Basics of NetworkX API, using Twitter network

# To get you up and running with the NetworkX API, we will run through some basic functions 
# that let you query a Twitter network that has been pre-loaded for you and is available in 
# the IPython Shell as T. The Twitter network comes from KONECT, and shows a snapshot of a 
# subset of Twitter users. It is an anonymized Twitter network with metadata.

# You're now going to use the NetworkX API to explore some basic properties of the network, 
# and are encouraged to experiment with the data in the IPython Shell.  Wait for the IPython 
# shell to indicate that the graph that has been preloaded under the 
# variable name T (representing a Twitter network), and then answer the following question:

# What is the size of the graph T, the type of T.nodes(), and the data structure of the third 
# element of the last entry of T.edges(data=True)? The len() and type() functions will be 
# useful here. To access the last entry of T.edges(data=True), you can use T.edges(data=True)[-1].

# (In the shell):
len(T)
Out[1]: 23369

type(T.nodes())
Out[4]: list

(T.edges(data=True)[-1])
Out[5]: (23324, 23327, {'date': datetime.date(2008, 2, 9)})






### Basic drawing of a network using NetworkX

# NetworkX provides some basic drawing functionality that works for small graphs. We have 
# selected a subset of nodes from the graph for you to practice using NetworkX's drawing facilities. 
# It has been pre-loaded as T_sub.

# Import necessary modules
import matplotlib.pyplot as plt
import networkx as nx

# Draw the graph to screen
nx.draw(T_sub)
plt.show()

# see drawingNetwork.svg





### Queries on a graph

# Now that you know some basic properties of the graph and have practiced using 
# NetworkX's drawing facilities to visualize components of it, it's time to 
# explore how you can query it for nodes and edges. Specifically, you're going to 
# look for "nodes of interest" and "edges of interest". To achieve this, you'll 
# make use of the .nodes() and .edges() methods that Eric went over in the 
# video. The .nodes() method returns a list of nodes, while the .edges() method 
# returns a list of tuples, in which each tuple shows the nodes that are 
# present on that edge. Recall that passing in the keyword argument data=True 
# in these methods retrieves the corresponding metadata associated with the 
# nodes and edges as well.

# You'll write list comprehensions to effectively build these queries in one 
# line. For a refresher on list comprehensions, refer to Part 2 of DataCamp's 
# Python Data Science Toolbox course. Here's the recipe for a list comprehension:

[ output expression for iterator variable in iterable if predicate expression ].

# You have to fill in the _iterable_ and the _predicate expression_. Feel free 
# to prototype your answer by exploring the graph in the IPython Shell before 
# submitting your solution.

# Input
T.nodes(data=True):
...(999, {'category': 'P', 'occupation': 'scientist'}),
 (1000, {'category': 'I', 'occupation': 'celebrity'}),
 (1001, {'category': 'I', 'occupation': 'scientist'}),
T.edges(data=True):
 [(1, 3, {'date': datetime.date(2012, 11, 17)}),
 (1, 4, {'date': datetime.date(2007, 6, 19)}),
 (1, 5, {'date': datetime.date(2014, 3, 18)}),...


# Use a list comprehension to get the nodes of interest: noi
noi = [n for n, d in T.nodes(data=True) if d['occupation'] == 'scientist']

# Use a list comprehension to get the edges of interest: eoi
eoi = [(u, v) for u, v, d in T.edges(data=True) if d['date'] < date(2010, 1, 1)]







### Checking the un/directed status of a graph

# In the video, Eric described to you different types of graphs. Which type of 
# graph do you think the Twitter network data you have been working with 
# corresponds to? Use Python's built-in type() function in the IPython Shell 
# to find out. The network, as before, has been pre-loaded as T.

# Of the four below choices below, which one corresponds to the type of graph that T is?

type(T)

# Output:
Out[1]: networkx.classes.digraph.DiGraph







### Specifying a weight on edges

# Weights can be added to edges in a graph, typically indicating the "strength" 
# of an edge. In NetworkX, the weight is indicated by the 'weight' key in the 
# metadata dictionary.

# Before attempting the exercise, use the IPython Shell to access the dictionary 
# metadata of T and explore it, for instance by running the 
# commands T.edge[1][10] and then T.edge[10][1]. Note how there's only one 
# field, and now you're going to add another field, called 'weight'.

# NOTE: the first item [] is the first node, the second item [] is the second node.

# Input:
In [4]: T.edge[1]
Out[4]: 
{3: {'date': datetime.date(2012, 11, 17)},
 4: {'date': datetime.date(2007, 6, 19)},
 5: {'date': datetime.date(2014, 3, 18)},
 6: {'date': datetime.date(2007, 3, 18)},

 # Set the weight of the edge
T.edge[1][10]['weight'] = 2

# Iterate over all the edges (with metadata)
for u, v, d in T.edges(data=True):

    # Check if node 293 is involved
    if 293 in [u,v]:
    
        # Set the weight to 1.1
        T.edge[u][v]['weight'] = 1.1







### Checking whether there are self-loops in the graph

# As Eric discussed, NetworkX also allows edges that begin and end on the same 
# node; while this would be non-intuitive for a social network graph, it is useful 
# to model data such as trip networks, in which individuals begin at one location 
# and end in another.

# It is useful to check for this before proceeding with further analyses, and 
# NetworkX graphs provide a method for this purpose: .number_of_selfloops().

# In this exercise as well as later ones, you'll find the assert statement 
# useful. An assert-ions checks whether the statement placed after it evaluates 
# to True, otherwise it will return an AssertionError.

# To begin, use the .number_of_selfloops() method on T in the IPython Shell to 
# get the number of edges that begin and end on the same node. A number of 
# self-loops have been synthetically added to the graph. Your job in this exercise 
# is to write a function that returns these edges.

# Define find_selfloop_nodes()
def find_selfloop_nodes(G):
    """
    Finds all nodes that have self-loops in the graph G.
    """
    nodes_in_selfloops = []
    
    # Iterate over all the edges of G
    for u, v in G.edges():
    
    # Check if node u and node v are the same
        if u == v:
        
            # Append node u to nodes_in_selfloops
            nodes_in_selfloops.append(u)
            
    return nodes_in_selfloops

# Check whether number of self loops equals the number of nodes in self loops
assert T.number_of_selfloops() == len(find_selfloop_nodes(T))

# output
42







### Visualizing using Matrix plots

# It is time to try your first "fancy" graph visualization method: a matrix plot. 
# To do this, nxviz provides a MatrixPlot object.

# nxviz is a package for visualizing graphs in a rational fashion. Under the hood, 
# the MatrixPlot utilizes nx.to_numpy_matrix(G), which returns the matrix form of 
# the graph. Here, each node is one column and one row, and an edge between the 
# two nodes is indicated by the value 1. In doing so, however, only the weight 
# metadata is preserved; all other metadata is lost, as you'll verify using an 
# assert statement.

# A corresponding nx.from_numpy_matrix(A) allows one to quickly create a graph 
# from a NumPy matrix. The default graph type is Graph(); if you want to make 
# it a DiGraph(), that has to be specified using the create_using keyword 
# argument, e.g. (nx.from_numpy_matrix(A, create_using=nx.DiGraph)).

# One final note, matplotlib.pyplot and networkx have already been imported 
# as plt and nx, respectively, and the graph T has been pre-loaded. For simplicity 
# and speed, we have sub-sampled only 100 edges from the network.

# Import nxviz
import nxviz as nv

# Create the MatrixPlot object: m
m = nv.MatrixPlot(T)

# Draw m to the screen
m.draw()

# Display the plot
plt.show()

# Convert T to a matrix format: A
A = nx.to_numpy_matrix(T)

# Convert A back to the NetworkX form as a directed graph: T_conv
T_conv = nx.from_numpy_matrix(A, create_using=nx.DiGraph())

# Check that the `category` metadata field is lost from each node
for n, d in T_conv.nodes(data=True):
    assert 'category' not in d.keys()

see MatrixPlot.svg







### Visualizing using Circos plots

# Circos plots are a rational, non-cluttered way of visualizing graph data, in 
# which nodes are ordered around the circumference in some fashion, and the edges 
# are drawn within the circle that results, giving a beautiful as well as 
# informative visualization about the structure of the network.

# In this exercise, you'll continue getting practice with the nxviz API, this time 
# with the CircosPlot object. matplotlib.pyplot has been imported for you as plt.

# Import necessary modules
import matplotlib.pyplot as plt
from nxviz import CircosPlot 

# Create the CircosPlot object: c
c = CircosPlot(T)

# Draw c to the screen
c.draw()

# Display the plot
plt.show()

see CircosPlot.svg







### Visualizing using Arc plots

# Following on what you've learned about the nxviz API, now try making an ArcPlot 
# of the network. Two keyword arguments that you will try here are 
# node_order='keyX' and node_color='keyX', in which you specify a key in 
# the node metadata dictionary to color and order the nodes by.

# matplotlib.pyplot has been imported for you as plt.

# Import necessary modules
import matplotlib.pyplot as plt
from nxviz import ArcPlot

# Create the un-customized ArcPlot object: a
a = ArcPlot(T)

# Draw a to the screen
a.draw()

# Display the plot
plt.show()

# Create the customized ArcPlot object: a2
a2 = ArcPlot(T, node_order='category', node_color='category')

# Draw a2 to the screen
a2.draw()

# Display the plot
plt.show()


see ArcPlot_custom.svg















######## Chapter 2: Important nodes ########















### Compute number of neighbors for each node

# How do you evaluate whether a node is an important one or not? There are a 
# few ways to do so, and here, you're going to look at one metric: the number 
# of neighbors that a node has.

# Every NetworkX graph G exposes a .neighbors(n) method that returns a list 
# of nodes that are the neighbors of the node n. To begin, use this method in 
# the IPython Shell on the Twitter network T to get the neighbors of of node 1. 
# This will get you familiar with how the function works. Then, your job in this 
# exercise is to write a function that returns all nodes that have m neighbors.

# Define nodes_with_m_nbrs()
def nodes_with_m_nbrs(G, m):
    """
    Returns all nodes in graph G that have m neighbors.
    """
    nodes = set()
    
    # Iterate over all nodes in G
    for n in G.nodes():
    
        # Check if the number of neighbors of n matches m
        if len(G.neighbors(n)) == m:
        
            # Add the node n to the set
            nodes.add(n)
            
    # Return the nodes with m neighbors
    return nodes

# Compute and print all nodes in T that have 6 neighbors
six_nbrs = nodes_with_m_nbrs(T, 6)
print(six_nbrs)


<script.py> output:
    {22533, 1803, 11276, 11279, 6161, 4261, 10149, 3880, 16681, 5420, 14898, 
    64, 14539, 6862, 20430, 9689, 475, 1374, 6112, 9186, 17762, 14956, 
    2927, 11764, 4725}







 ### Compute degree distribution

 # The number of neighbors that a node has is called its "degree", and it's 
 # possible to compute the degree distribution across the entire graph. In 
 # this exercise, your job is to compute the degree distribution across T.

 # Compute the degree of every node: degrees
degrees = [len(T.neighbors(n)) for n in T.nodes()]

# Print the degrees
print(degrees)

# Output
[47, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 0, 10, 27, 0, 0, 0, 0, 0, 0, 0, 0, 34, 0, 0, 0, 0, 0, 60, 0, 11, 4, 0, 12, 0, 0, 56, 53, 0, 30, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 6, 0, 0, 0, 0, 0, 33, 0, 0, 0, 0, 39, 8, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0,







### Degree centrality distribution

# The degree of a node is the number of neighbors that it has. The degree 
# centrality is the number of neighbors divided by all possible neighbors that 
# it could have. Depending on whether self-loops are allowed, the set of possible 
# neighbors a node could have could also include the node itself.

# The nx.degree_centrality(G) function returns a dictionary, where the keys are 
# the nodes and the values are their degree centrality values.

# The degree distribution degrees you computed in the previous exercise using 
# the list comprehension has been pre-loaded.

# Import matplotlib.pyplot
import matplotlib.pyplot as plt

# Compute the degree centrality of the Twitter network: deg_cent
deg_cent = nx.degree_centrality(T)

# Plot a histogram of the degree centrality distribution of the graph.
plt.figure()
plt.hist(list(deg_cent.values()))
plt.show()

# Plot a histogram of the degree distribution of the graph
plt.figure()
plt.hist(degrees)
plt.show()

# Plot a scatter plot of the centrality distribution and the degree distribution
plt.figure()
plt.scatter(degrees, list(deg_cent.values()))
plt.show()







### Shortest Path I

# You can leverage what you know about finding neighbors to try finding paths in 
# a network. One algorithm for path-finding between two nodes is the 
# "breadth-first search" (BFS) algorithm. In a BFS algorithm, you start from a 
# particular node and iteratively search through its neighbors and neighbors' 
# neighbors until you find the destination node.

# Pathfinding algorithms are important because they provide another way of 
# assessing node importance; you'll see this in a later exercise.

# In this set of 3 exercises, you're going to build up slowly to get to the 
# final BFS algorithm. The problem has been broken into 3 parts that, if you 
# complete in succession, will get you to a first pass implementation of the 
# BFS algorithm.

# Note: Breadth-first-search is a neat concept and easy to understand... google 
# for additional explanation

# Define path_exists()
def path_exists(G, node1, node2):
    """
    This function checks whether a path exists between two nodes (node1, node2) in graph G.
    """
    visited_nodes = set()
    
    # Initialize the queue of cells to visit with the first node: queue
    queue = [node1]  
    
    # Iterate over the nodes in the queue
    for node in queue:
    
        # Get neighbors of the node
        neighbors = G.neighbors(node) 
        
        # Check to see if the destination node is in the set of neighbors
        if node2 in neighbors:
            print('Path exists between nodes {0} and {1}'.format(node1, node2))
            return True
            break







### Shortest Path II

# Now that you've got the code for checking whether the destination node is 
# present in neighbors, next up, you're going to extend the same function to 
# write the code for the condition where the destination node is not present 
# in the neighbors.

# All the code you need to write is in the else condition; that is, if node2 
# is not in neighbors.

def path_exists(G, node1, node2):
    """
    This function checks whether a path exists between two nodes (node1, node2) in graph G.
    """
    visited_nodes = set()
    queue = [node1]
    
    for node in queue:  
        neighbors = G.neighbors(node)
        if node2 in neighbors:
            print('Path exists between nodes {0} and {1}'.format(node1, node2))
            return True
            break
        
        else:
            # Add current node to visited nodes
            visited_nodes.add(node)
            
            # Add neighbors of current node that have not yet been visited
            queue.extend([n for n in neighbors if n not in visited_nodes])







### Shortest Path III

# This is the final exercise of this trio! You're now going to complete the 
# problem by writing the code that returns False if there's no path between two nodes.

# NOTE: this is the full implementation of the breadth-first algorithm.

def path_exists(G, node1, node2):
    """
    This function checks whether a path exists between two nodes (node1, node2) in graph G.
    """
    visited_nodes = set()
    queue = [node1]
    
    for node in queue:  
        neighbors = G.neighbors(node)
        if node2 in neighbors:
            print('Path exists between nodes {0} and {1}'.format(node1, node2))
            return True
            break

        else:
            visited_nodes.add(node)
            queue.extend([n for n in neighbors if n not in visited_nodes])
        
        # Check to see if the final element of the queue has been reached
        if node == queue[-1]:
            print('Path does not exist between nodes {0} and {1}'.format(node1, node2))

            # Place the appropriate return statement
            return False







### NetworkX betweenness centrality on a social network

# Betweenness centrality is a node importance metric that uses information about 
# the shortest paths in a network. It is defined as the fraction of all possible 
# shortest paths between any pair of nodes that pass through the node.

# NetworkX provides the nx.betweenness_centrality(G) function for computing the 
# betweenness centrality of every node in a graph, and it returns a dictionary 
# where the keys are the nodes and the values are their betweenness centrality 
# measures.

# Compute the betweenness centrality of T: bet_cen
bet_cen = nx.betweenness_centrality(T)

# Compute the degree centrality of T: deg_cen
deg_cen = nx.degree_centrality(T)

# Create a scatter plot of betweenness centrality and degree centrality
plt.scatter(list(bet_cen.values()), list(deg_cen.values()))

# Display the plot
plt.show()

see betweenessCentrality_vs_degreeCentrality.svg
# Note: degree Centrality is a poor indicator of betweenness centrality.  they
# measure different aspects of node importance as seen from the poor correlation 
# in the graph.







### Deep dive - Twitter network

# You're going to now take a deep dive into a Twitter network, which will help 
# reinforce what you've learned earlier. First, you're going to find the nodes 
# that can broadcast messages very efficiently to lots of people one degree of 
# separation away.

# NetworkX has been pre-imported for you as nx.

# Define find_nodes_with_highest_deg_cent()
def find_nodes_with_highest_deg_cent(G):

    # Compute the degree centrality of G: deg_cent
    deg_cent = nx.degree_centrality(G)
    
    # Compute the maximum degree centrality: max_dc
    max_dc = max(list(deg_cent.values()))
    
    nodes = set()
    
    # Iterate over the degree centrality dictionary
    for k, v in deg_cent.items():
    
        # Check if the current value has the maximum degree centrality
        if v == max_dc:
        
            # Add the current node to the set of nodes
            nodes.add(k)
            
    return nodes
    
# Find the node(s) that has the highest degree centrality in T: top_dc
top_dc = find_nodes_with_highest_deg_cent(T)
print(top_dc)

# Write the assertion statement
for node in top_dc:
    assert nx.degree_centrality(T)[node] == max(nx.degree_centrality(T).values())

# Output: (node with highest degree centrality)
<script.py> output:
    {11824}







### Deep dive - Twitter network part II

# Next, you're going to do an analogous deep dive on betweenness centrality! Just 
# a few hints to help you along: remember that betweenness centrality is 
# computed using nx.betweenness_centrality(G).

# Define find_node_with_highest_bet_cent()
def find_node_with_highest_bet_cent(G):

    # Compute betweenness centrality: bet_cent
    bet_cent = nx.betweenness_centrality(G)
    
    # Compute maximum betweenness centrality: max_bc
    max_bc = max(list(bet_cent.values()))
    
    nodes = set()
    
    # Iterate over the betweenness centrality dictionary
    for k, v in bet_cent.items():
    
        # Check if the current value has the maximum betweenness centrality
        if v == max_bc:
        
            # Add the current node to the set of nodes
            nodes.add(k)
    print(nodes)        
    return nodes

# Use that function to find the node(s) that has the highest betweenness centrality in the network: top_bc
top_bc = find_node_with_highest_bet_cent(T)

# Write an assertion statement that checks that the node(s) is/are correctly identified.
for node in top_bc:
    assert nx.betweenness_centrality(T)[node] == max(nx.betweenness_centrality(T).values())

# Output: (node with highest degree centrality)
{1}















######## Chapter 3: Structures ########















### Identifying triangle relationships

# Now that you've learned about cliques, it's time to try leveraging what you know 
# to find structures in a network. Triangles are what you'll go for first. We may 
# be interested in triangles because they're the simplest complex clique. Let's 
# write a few functions; these exercises will bring you through the fundamental 
# logic behind network algorithms.

# In the Twitter network, each node has an 'occupation' label associated with it, 
# in which the Twitter user's work occupation is divided into celebrity, 
# politician and scientist. One potential application of triangle-finding 
# algorithms is to find out whether users that have similar occupations are more 
# likely to be in a clique with one another.

from itertools import combinations

# Define is_in_triangle() 
def is_in_triangle(G, n):
    """
    Checks whether a node `n` in graph `G` is in a triangle relationship or not. 
    
    Returns a boolean.
    """
    in_triangle = False
    
    # Iterate over all possible triangle relationship combinations
    for n1, n2 in combinations(G.neighbors(n), 2):
    
        # Check if an edge exists between n1 and n2
        if G.has_edge(n1, n2):
            in_triangle = True
            break
    return in_triangle







### Finding nodes involved in triangles

# NetworkX provides an API for counting the number of triangles that every node 
# is involved in: nx.triangles(G). It returns a dictionary of nodes as the keys 
# and number of triangles as the values. Your job in this exercise is to modify 
# the function defined earlier to extract all of the nodes involved in a 
# triangle relationship with a given node.

from itertools import combinations

# Write a function that identifies all nodes in a triangle relationship with a given node.
def nodes_in_triangle(G, n):
    """
    Returns the nodes in a graph `G` that are involved in a triangle relationship with the node `n`.
    """
    triangle_nodes = set([n])
    
    # Iterate over all possible triangle relationship combinations
    for n1, n2 in combinations(G.neighbors(n), 2):
    
        # Check if n1 and n2 have an edge between them
        if G.has_edge(n1, n2):
        
            # Add n1 to triangle_nodes
            triangle_nodes.add(n1)
            
            # Add n2 to triangle_nodes
            triangle_nodes.add(n2)
            
    return triangle_nodes
    
# Write the assertion statement
assert len(nodes_in_triangle(T, 1)) == 35

# Output: Great work! Your function correctly identified that node 1 is in a triangle relationship with 35 other nodes.
True







### Finding open triangles

# Let us now move on to finding open triangles! Recall that they form the basis 
# of friend recommendation systems; if "A" knows "B" and "A" knows "C", then it's 
# probable that "B" also knows "C".

from itertools import combinations

# Define node_in_open_triangle()
def node_in_open_triangle(G, n):
    """
    Checks whether pairs of neighbors of node `n` in graph `G` are in an 'open triangle' relationship with node `n`.
    """
    in_open_triangle = False
    
    # Iterate over all possible triangle relationship combinations
    for n1, n2 in combinations(G.neighbors(n), 2):
    
        # Check if n1 and n2 do NOT have an edge between them
        if not G.has_edge(n1, n2):
        
            in_open_triangle = True
            
            break
            
    return in_open_triangle

# Compute the number of open triangles in T
num_open_triangles = 0

# Iterate over all the nodes in T
for n in T.nodes():

    # Check if the current node is in an open triangle
    if node_in_open_triangle(T, n):
    
        # Increment num_open_triangles
        num_open_triangles += 1
        
print(num_open_triangles)

# Output:
22







### Finding all maximal cliques of size "n"

# Now that you've explored triangles (and open triangles), let's move on to the 
# concept of maximal cliques. Maximal cliques are cliques that cannot be extended 
# by adding an adjacent edge, and are a useful property of the graph when finding 
# communities. NetworkX provides a function that allows you to identify the nodes 
# involved in each maximal clique in a graph: nx.find_cliques(G). Play around 
# with the function by using it on T in the IPython Shell, and then try answering 
# the exercise.

# Define maximal_cliques()
def maximal_cliques(G, size):
    """
    Finds all maximal cliques in graph `G` that are of size `size`.
    """
    mcs = []
    for clique in nx.find_cliques(G):
        if len(clique) == size:
            mcs.append(clique)
    return mcs

# Check that there are 33 maximal cliques of size 3 in the graph T
assert len(maximal_cliques(T, 3)) == 33







### Subgraphs

# There may be times when you just want to analyze a subset of nodes in a network. 
# To do so, you can copy them out into another graph object using G.subgraph(nodes), 
# which returns a new graph object (of the same type as the original graph) that is 
# comprised of the iterable of nodes that was passed in.

# matplotlib.pyplot has been imported for you as plt


nodes_of_interest = [29, 38, 42]

# Define get_nodes_and_nbrs()
def get_nodes_and_nbrs(G, nodes_of_interest):
    """
    Returns a subgraph of the graph `G` with only the `nodes_of_interest` and their neighbors.
    """
    nodes_to_draw = []
    
    # Iterate over the nodes of interest
    for n in nodes_of_interest:
    
        # Append the nodes of interest to nodes_to_draw
        nodes_to_draw.append(n)
        
        # Iterate over all the neighbors of node n
        for nbr in G.neighbors(n):
        
            # Append the neighbors of n to nodes_to_draw
            nodes_to_draw.append(nbr)
            
    return G.subgraph(nodes_to_draw)

# Extract the subgraph with the nodes of interest: T_draw
T_draw = get_nodes_and_nbrs(T, nodes_of_interest)

# Draw the subgraph to the screen
nx.draw(T_draw)
plt.show()

see subgraph.svg







### Subgraphs II

# In the previous exercise, we gave you a list of nodes whose neighbors we 
# asked you to extract.

# Let's try one more exercise in which you extract nodes that have a particular 
# metadata property and their neighbors. This should hark back to what you've 
# learned about using list comprehensions to find nodes. The exercise will 
# also build your capacity to compose functions that you've already 
# written before.

# Extract the nodes of interest: nodes
nodes = [n for n, d in T.nodes(data=True) if d['occupation'] == 'celebrity']

# Create the set of nodes: nodeset
nodeset = set(nodes)

# Iterate over nodes
for n in nodes:

    # Compute the neighbors of n: nbrs
    nbrs = T.neighbors(n)
    
    # Compute the union of nodeset and nbrs: nodeset
    nodeset = nodeset.union(nbrs)

# Compute the subgraph using nodeset: T_sub
T_sub = T.subgraph(nodeset)

# Draw T_sub to the screen
nx.draw(T_sub)
plt.show()

see subgraphsII.svg





