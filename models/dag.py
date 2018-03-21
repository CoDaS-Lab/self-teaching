from collections import deque

class DirectedGraph:
    def __init__(self, edges, transmission_rate=0.9, background_rate=0.05):
        self.adjacency_matrix = edges
        self.n = self.adjacency_matrix.shape[0]
        self.transmission_rate = transmission_rate
        self.background_rate = background_rate
        
        assert self.n >= 0
        assert self.transmission_rate >= 0.0
        assert self.background_rate >= 0.0
        
    def get_parents(self, node):
        """Calculate the parents of a given node"""
        return np.flatnonzero(self.adjacency_matrix[:, node])
        
    def get_children(self, node):
        """Calculate the children of a given node"""
        return np.flatnonzero(self.adjacency_matrix[node])
        
    def intervene(self, node):
        """Calculate the outcome from intervening on a particular node"""
        
        outcomes = np.zeros(self.n)  # array to store outcomes
        outcomes[node] = 1.0  # set intervened node to be on
        
        # temporarily remove edge between node and parent
        intervened_parents = self.get_parents(node) 
        self.adjacency_matrix[intervened_parents, node] = 0

        q = deque()  # create queue to store nodes
        
        # set root nodes to be bg rate and add to queue
        # root notes have no parents and are not the node intervened on
        for i in range(self.n):
            if len(self.get_parents(i)) == 0 and i != node:
                outcomes[i] = self.background_rate
                q.append(i)
        
        # append children of intervened node to queue
        children = self.get_children(node)
        q.append(children)  # append children of nodes to queue
        
        while len(q) is not 0:
            curr_node = q.popleft()  # remove first node from queue
            
            # calculate probability of turning on
            parents = self.get_parents(curr_node)
            outcomes[curr_node] = np.sum(outcomes[parents]) * \
                self.transmission_rate + self.background_rate
            
            # append children to queue
            children = self.get_children(curr_node)
            for child in children:
                q.append(child) 
        
        # add edges back in from parents to intervened node
        self.adjacency_matrix[intervened_parents, node] = 1

        # set any outcomes greater than 1 to 1
        outcomes[outcomes > 1.0] = 1.0
        
        return outcomes
        
    def likelihood(self):
        """Calculate the likelihood of a node being turned on?"""
        lik = np.zeros((self.n, self.n))
        for i in range(self.n):
            lik[i] = self.intervene(i)
            
        return lik
