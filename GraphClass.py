import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skewnorm, truncnorm, poisson

#Helper functions
def prune(G):
        isolate = []

        for i in G.nodes:
            if len(nx.node_connected_component(G, i)) < int(len(G.nodes)/3):
                isolate.append(i)
        G.remove_nodes_from(isolate)
        return G

#Get mean sentiment of a graph
def getSentimentStatistics(graph):
    sentiment = []
    for i in graph.G.nodes:
        sentiment.append(graph.G.nodes[i]["sentiment"])

    return np.mean(sentiment), np.std(sentiment)

#Graph class
class Graph:
    
    #Create a full random, guassian distributed, graph
    def createRandomGraph(self, numNodes, meanClusterSize, clusterVariance, probIntraConnection, probInterConnection, meanSkew = 0, random = False):
        G = nx.gaussian_random_partition_graph(numNodes, meanClusterSize, clusterVariance, probIntraConnection, probInterConnection, directed = False)
        G = prune(G)
        self.G = G
        self.initSentiment(meanSkew, random)
        self.initReplies(random)
    
    #Create Graph with single comment
    def createRandomComment(self, meanSkew, random = False):
        G = nx.Graph()
        self.G = G
        self.G.add_node(0)
        self.initSentiment(meanSkew, random = False)
    
    #Initialize sentiment in the graph
    def initSentiment(self, meanSkew, random = False):
        numNodes = len(self.G.nodes)

        if random:
            weights = np.round(np.random.uniform(low = -1, high = 1, size = numNodes),4)
        else:
            weights = np.round(truncnorm.rvs(-1 - meanSkew, 1 - meanSkew, loc = meanSkew, size = numNodes),4)
           
        j = 0
        for i in self.G.nodes:
            self.G.nodes[i]["sentiment"] = weights[j]
            j +=1
            
    #Initialize edge wieghts in the graph
    def initReplies(self, random = False):
        numEdges = len(self.G.edges)

        if random:
            weights = np.random.randint(low=1,high = 10, size = numEdges)
            j = 0
            for u,v in self.G.edges:
                self.G[u][v]["replies"] = weights[j]
                j += 1
        else:
            for u,v in self.G.edges:
                self.G[u][v]["replies"] = np.random.randint(low= 1,high = 2*len([n for n in self.G.neighbors(u)]))
                
    #add commenter (node) to graph        
    def addCommenter(self, meanSkew = 0, random = False, propagate = False, threshold = 0, variable = True):
        index = len(self.G.nodes)+1
        self.G.add_node(index)
        
        if random:
            self.G.nodes[index]["sentiment"] = np.round(np.random.uniform(low = -1, high = 1),4)
        else:
            self.G.nodes[index]["sentiment"] = np.round(truncnorm.rvs(-1 - meanSkew, 1 - meanSkew, loc = meanSkew),4)
            
        connection = np.random.choice(list(self.G.nodes)[:-1])
        self.G.add_edge(index,connection)
        self.G[index][connection]["replies"] = 1
        
        if propagate:
            self.propagateSentiment(connection, index, threshold, variable)
        
    #add reply (edge) to graph or add +1 to exisiting edge
    def addReply(self, random = False, propagate = False, threshold = 0, variable = True):
        if random:
            probability = None
        else:
            probability = []
            for u in self.G.nodes:
                probability.append(len([n for n in self.G.neighbors(u)]))
                
            probability = np.array(probability)/sum(probability)
        
        if len(self.G.nodes) > 2:
            commenters = np.random.choice(list(self.G.nodes),2,replace=False, p = probability)
        else:
            commenters = list(self.G.nodes)
        
        if self.G.has_edge(commenters[0],commenters[1]):
            self.G[commenters[0]][commenters[1]]["replies"] += 1
        else:
            self.G.add_edge(commenters[0],commenters[1])
            self.G[commenters[0]][commenters[1]]["replies"] = 1
            
        if propagate:
            self.propagateSentiment(commenters[0],commenters[1], threshold, variable)
            
    #Propgate sentiment from one node to another node1 -> node2
    def propagateSentiment(self, node1, node2, threshold = 0, variable = True):
        weight = 1/(self.G[node1][node2]["replies"] + 1)
        
        if variable:
            threshold *= 1/(self.G[node1][node2]["replies"])
        
        if np.random.uniform() > threshold:
             self.G.nodes[node2]["sentiment"] = np.round((weight*self.G.nodes[node2]["sentiment"] + (1-weight)*self.G.nodes[node1]["sentiment"]),4)
                
    #Propagate sentiment between two random connected nodes in graph
    def propagateSentimentRandom(self, threshold = 0, variable = True):
        edge = list(self.G.edges)[np.random.randint(0,len(self.G.edges))]
        
        if np.random.uniform() > .5:
            self.propagateSentiment(edge[0], edge[1], threshold, variable)
        else:
            self.propagateSentiment(edge[1], edge[0], threshold, variable)
    
    #plot out graph
    def plotGraph(self, showColor = True, showSentiment = True, showReplies = True):
        sentiments = None
        replies = None
        color = None
        cmap = None
        labels = False
        
        if showSentiment:
            sentiments = dict(self.G.nodes(data="sentiment"))
            labels = True
            
        if showColor:
            color = list(dict(self.G.nodes(data="sentiment")).values())
            cmap='RdYlGn'   
        pos=nx.kamada_kawai_layout(self.G)
        
        if showReplies:
            replies = nx.get_edge_attributes(self.G,'replies')
            nx.draw_networkx_edge_labels(self.G,pos,edge_labels=replies)
    
        nx.draw(self.G,labels = sentiments, with_labels = labels, node_color = color, cmap=cmap, pos = pos)
       