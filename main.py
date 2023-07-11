import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class random_walk_agent(object):
    def __init__(self):
        self.n_states = 7 # A, B, C, D, E, F, G
        self.first_state = 3 # D
        self.n_training_sets = 100
        self.n_sequence = 10 # number of sequences in a set
        self.w_initial = np.array([0, 0.5, 0.5, 0.5, 0.5, 0.5, 1])
        self.w_true = np.array([0.0, 1.0/6, 1.0/3 ,1.0/2, 2.0/3, 5.0/6, 1.0])

    def sample_sequence(self):
        sequence = [3]
        curr_state = 3

        while curr_state != 0 and curr_state != 6:
            rand_num = np.random.rand()

            if rand_num > 0.5:
                curr_state += 1
                sequence.append(curr_state)
            else:
                curr_state -= +1
                sequence.append(curr_state)

        return sequence
    
    def get_training_data(self):
        training_set = []

        for i in range(self.n_training_sets):
            one_training_set = []
            for j in range(self.n_sequence):
                one_training_set.append(self.sample_sequence())
            training_set.append(one_training_set)
        # print(training_set)
        # print(np.array(training_set).shape)

        return training_set
    
    def get_delta_w(self, sequence, w, lr, lam):
        delta_w = np.zeros([self.n_states])
        grad = np.zeros([self.n_states])
        
        for t in range(len(sequence)-1):
            # print("====")
            # print(t)
            # print(seq)
            # print(seq[t])
            state_vector = np.zeros([self.n_states])
            state_at_t = sequence[t]
            state_vector[state_at_t] = 1

            grad = state_vector + lam * grad

            delta_p = w[sequence[t+1]] - w[sequence[t]] 
            delta_w += lr * delta_p * grad
            
        return delta_w
    
    def train_single_pass(self, training_data_, lr, lam):
        w = self.w_initial.copy()

        for s in training_data_:
            w += self.get_delta_w(s, w, lr, lam)
        
        rmse = np.mean((w - self.w_true) ** 2) ** 0.5

        return rmse

    def train_to_convergence(self, training_data_, lr, lam):
        w = self.w_initial.copy()

        for i in range(9999):
            delta_w = np.zeros([self.n_states])
            for s in training_data_:
                delta_w += self.get_delta_w(s, w, lr, lam)
                
            # print("delta_w", delta_w)

            # When a vector norm is very small, it means the update is now very small, thus the algorithm converges
            # Setting is as 1e-3 to allow for faster testing; can set it to lower value to produce a smoother curve in the graph
            if np.linalg.norm(delta_w) < 1e-3:
                break
            w += delta_w
        
        # calculate root mean square error
        rmse = np.mean((w - self.w_true) ** 2) ** 0.5

        return rmse
    
    # First experiment in paper
    def convergence_experiment(self, training_data_, lr, lam):
        all_rmse = []

        for set in tqdm(training_data_):
            rmse = self.train_to_convergence(set, lr, lam)
            all_rmse.append(rmse)

        avg_rmse = np.mean(all_rmse)

        return avg_rmse

    # Second experiment in paper
    def one_pass_experiment(self, training_data, lr, lam):
        all_rmse = []

        for set in tqdm(training_data):
            rmse = self.train_single_pass(set, lr, lam)
            all_rmse.append(rmse)
        avg_rmse = np.mean(all_rmse)

        return avg_rmse

if __name__ == "__main__":
    # Fig 3
    sutton_agent = random_walk_agent()
    training_data = sutton_agent.get_training_data()

    exp1_lam = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    exp1_errors = []
    
    for l in exp1_lam:
        ex_error= sutton_agent.convergence_experiment(training_data, 0.01, l)
        exp1_errors.append(ex_error)

    # print(exp1_errors)

    plt.title("Figure 3")
    plt.plot(exp1_lam, exp1_errors, linestyle="-", marker="o")
    plt.xlabel("lambda")
    plt.ylabel("RMSE with learing rate (alpha) = 0.01")
    plt.show()

    # Fig 4
    exp2_lam = [0, 0.3, 0.8, 1]
    exp2_lr = np.linspace(0, 0.7, 20)

    exp2_all_errors = []
    for l in exp2_lam:
        exp2_errors = []
        for a in exp2_lr:
            ex_error = sutton_agent.one_pass_experiment(training_data, a, l)
            exp2_errors.append(ex_error)
        exp2_all_errors.append(exp2_errors)


    for i in range(len(exp2_lam)):
        plt.plot(exp2_lr, exp2_all_errors[i], linestyle="-", marker="o", label="lambda = {}".format(exp2_lam[i]))
        plt.annotate("lambda = {}".format(i), (0.7, exp2_all_errors[i][-1]))

    plt.title("Figure 4")
    plt.xlim(0, 0.6)
    plt.ylim(0, 0.7)
    plt.xlabel("learning rate (alpha)")
    plt.ylabel("RMSE")
    plt.legend()
    plt.show()

    # Fig 5
    # Looking at Fig 4, learning rate (alpha) = 0.3 minimizes the error best
    lr = 0.3 
    fig5_lam = np.linspace(0, 1, 20)
    fig5_error = []

    for l in fig5_lam:
        ex_error = sutton_agent.one_pass_experiment(training_data, 0.3, l)
        fig5_error.append(ex_error)

    plt.title("Figure 5")
    plt.plot(fig5_lam, fig5_error, linestyle="-", marker="o")
    plt.xlabel("lambda")
    plt.ylabel("RMSE with learning rate (alpha) = 0.3")
    plt.show()
