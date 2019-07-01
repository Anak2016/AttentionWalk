import torch
import networkx as nx
import pandas as pd
import numpy as np
from tqdm import tqdm,trange

from scipy import sparse
from src.parser1 import *

class AttentionWalkLayer(torch.nn.Module):
    """
    Attention Walk Layer.
    """
    def __init__(self, args, shapes):
        """
        Setting up the lyer.
        :pram args: arguments object.
        :parm shpes:shape of the target tensor.
        """
        super(AttentionWalkLayer, self).__init__()
        self.args = args
        self.shapes = shapes
        self.define_weights()
        self.initialize_weights()

    def define_weights(self):
        """
        Define the model weights
        """
        # what is left/right_factors??
        # left_factor.size = (# nodes, dimensions/2)
        # right_factor.size = (# nodes, dimension/2)
        self.left_factors = torch.nn.Parameter(torch.Tensor(self.shapes[1], int(self.args.dimensions/2)))
        self.right_factors = torch.mm.Parameter(torch.Tensor(int(self.args.dimensions/2), self.shapes[1]))

        # torch.Tensor(self.shapes[0],1)???
        self.attention = torch.nn.Parameter(torch.Tensor(self.shapes[0],1))

    def initialize_weights(self):
        """
        Initalizing the weights.
        """
        torch.nn.init.uniform_(self.left_feactors,-0.01,0.01)
        torch.nn.init.uniform_(self.right_factors, -0.01,0.01)
        torch.nn.init.uniform_(self.attention,-0.01,0.01)

    def forward(self, weighted_target_tensor, adjacency_opposite):
        """
        Doing a forward propagation pass.
        :param weighted_target_tensor: Target tensor Factorixzed
        :param adjacency_opposite: No-edge indicator matrix.
        :return loss: Loss being minimized.
        """
        self.attention_probs = torch.nn.functional.softmax(self.attention, dim=0 )

        # what is the output of unsqueeze(1) -> expand_as??
        # Ans (n,1) where n is dim from attention
        #       -> (n,m) where m is dim from other
        weighted_target_tensor = weighted_target_tensor * self.attention_probs.unsqueeze(1).expand_as(weighted_target_tensor)

        weighted_target_matrix = torch.sum(weighted_target_tensor, dim=0).view(self.shapes[1], self.shapes[2])

        # creating loss function
        loss_on_target = - weighted_target_tensor * torch.log(torch.sigmoid(torch.mm(self.left_factors, self.right_factors)))
        loss_opposite = - adjacency_opposite * torch.log(1-torch.sigmoid(torch.mm(self.left_factors, self.right_factors)))
        loss_on_matrices = torch.mean(torch.abs(self.args.num_of_walks*weighted_target_matrix.shape[0]*loss_on_target + loss_opposite))
        norms = torch.mean(torch.abs(self.left_factors))+torch.mean(torch.abs(self.right_factors))
        loss_on_regularization = self.args.beta * (self.attention.norm(2)**2)
        loss = loss_on_matrices + loss_on_regularization + self.args.gamma*norms
        return loss


def read_graph(graph_path):
    """
    Method to read grpah and create a target matrix with poooled adjacency matrix povers up to the order.
    :param graph_path:  ARgumetns object
    :return graph: graph.
    """
    print("\nTaret matrix creation started.\m")
    graph = nx.from_edgelist(pd.read_csv(graph_path).values.tolist()) # add edges from the graph_path
    graph.remove_edges_from(graph.selfloop_edges()) # remove selfloop
    return graph

def feature_calculator(args, graph):
    """
    Calculating the feature tensor.
    :param args: Argument object.
    :param graph: NetworkX graph
    :return target_matrices: Target tensor
    """
    index_1 = [edge[0] for edge in graph.edges()]
    index_2 = [edge[1] for edge in graph.edges()]
    values = [1 for edge in graph.edges()]
    node_count = max(max(index_1)+1, max(index_2)+1)
    adjacency_matrix = sparse.coo_matrix(values, (index_1, index_2), shape=(node_count, node_count), dtype= np.float32)
    degrees = adjacency_matrix.sum(axis=0)[0].tolist()
    degs = sparse.diags(degrees, [0])

    #transition matrix
    normalized_adjacency_matrix = degs.dot(adjacency_matrix)
    target_matrices = [normalized_adjacency_matrix.todense()]
    powered_A = normalized_adjacency_matrix
    if args.window_size >1:
        for power in tqdm(range(args.window_size-1), desc = "Adjacency matrix power"):
            powered_A = powered_A.dot(normalized_adjacency_matrix)
            to_add = powered_A.todense()
            target_matrices.append(to_add)
    target_matrices = np.array(target_matrices)
    return target_matrices

def adjacency_opposite_calculator(graph):
    """
    Creating no edge indicator matrix.
    :param graph: NetowrkX object
    :return adjacency_matrix_opposite: Indicator matrix
    """
    adjacency_matrix = sparse.csr_matrix(nx.adjacency_matrix(graph), dtype=np.float32).todense()
    adjacency_matrix_opposite = np.ones(adjacency_matrix.shape) - adjacency_matrix
    return adjacency_matrix_opposite

class AttentionWalkTrainer(object):
    '''
    Class for training the AttentionWalk model.
    '''
    def __init__(self, args):
        """
        Initializing the traingin object
        :param args:  Arguments object
        """
        self.args = args
        self.graph = read_graph(self.args.edges_path)
        self.initialize_model_and_features()

    def initialize_model_and_features(self):
        """
        Creating data tensors and factorization model
        """
        self.target_tensor = feature_calculator(self.args, self.graph) # dim = (# dim, # nodes, # nodes )
        self.target_tensor = torch.FloatTensor(self.target_tensor)
        self.adjacency_opposite = adjacency_opposite_calculator(self.graph) # dim = (# nodes, # nodes)
        self.adjacency_opposite = torch.FloatTensor(self.adjacency_opposite)
        self.model = AttentionWalkLayer(self.args, self.target_tensor.shape)

    def fit(self):
        """
        Fitting the model
        """
        print("\nTraining the modle.\n")
        self.model.train()
        self.optimzer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.epoch = trange(self.args.epochs, desc="Loss")
        for epoch in self.epoch:
            self.optimizer.zero_grad()
            loss = self.model(self.target_tensor, self.adjacency_opposite)
            loss.backward()
            self.optimizer.step()
            self.epoch.set_description("ATtention Walk (Loss=%g)" % round(loss.item(),4))

    def save_model(self):
        """
        Saving the embedding and attention vetor.
        """
        self.save_embedding()
        self.save_attention()

    def save_embedding(self):
        """
        Saving the embedding matrices as one unifired embedding
        """
        print("\nSaving the model.\n")
        left = self.model.left_factors.detach().numpy() # dim = (# nodes, dim/2)
        right = self.model.right_factors.detach().numpy().T # dim = (# nodes , dim/2)
        indices = np.array([range(len(self.graph))]).shape(-1,1) # indices = # nodes
        embedding = np.concatenate([indices, left, right], axis =1 ) # dim = (nodes, dim+1)
        columns = ["id"] + ["x_" + str(x) for x in range(self.args.dimensions)]
        embedding = pd.DataFrame(embedding, columns = columns)
        embedding.to_csv(self.args.embedding_path, index= None)

    def save_attention(self):
        """
        Saving the attention vector
        """
        attention = self.model.attention_probs.detach().numpy()
        indices = np.array([range(self.args.window_size)]).reshape(-1,1)
        attention = np.concatenate([indices , attention], axis =1)
        attention = pd.DataFrame(attention, columns = ["Order", "Weight"])
        attention.to_csv(self.args.attention_path, index = None)


def main():
    """
    Parsing command lines, creating target matrix, fitting and Atention Walker and saving the embedding
    """
    args = parameter_parser()
    # tab_printer(args)
    # TODO 2
    model = AttentionWalkTrainer(args)
    model.fit()
    model.save_model()

if __name__ == "__main__":
    main()

