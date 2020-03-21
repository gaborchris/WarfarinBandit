import numpy as np

class LinearBandit:

    def __init__(self, feature_dims, learning_rate=0.1):
        self.num_actions = 3
        self.beta = []
        self.A = []
        self.b = []
        self.learning_rate = learning_rate
        for i in range(self.num_actions):
            self.beta.append(np.zeros((feature_dims, 1)))
            self.A.append(np.identity(feature_dims))
            self.b.append(np.zeros((feature_dims, 1)))

    def take_action(self, context):
        probs = []
        for i in range(self.num_actions):
            A_inv = np.linalg.inv(self.A[i])
            self.beta[i] = np.matmul(A_inv, self.b[i])
            A_prod = np.squeeze(np.sqrt(np.dot(np.matmul(context.T, A_inv), context)))
            probs.append(np.dot(self.beta[i].T, context) + self.learning_rate * A_prod)
        action = np.argmax(probs)
        return action

    def evaluate_beta(self, X):
        probs = []
        for i in range(self.num_actions):
            probs.append(np.dot(X, self.beta[i]))
        probs = np.array(probs).squeeze()
        actions = np.argmax(probs, axis=0)

        return actions



    def update_arm(self, k, context, reward):
        self.A[k] = self.A[k] + np.matmul(context, context.T)
        self.b[k] = self.b[k] + reward*context




