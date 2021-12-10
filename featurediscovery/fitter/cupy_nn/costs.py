import cupy as cp


class CrossEntropyCost():

    epsilon = 1e-12

    def compute_cost(self, AL, Y):
        """
            Implement the cost function defined by equation (7).

            Arguments:
            AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
            Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

            Returns:
            cost -- cross-entropy cost
            """

        m = Y.shape[1]

        # Compute loss from aL and y.
        AL = cp.clip(AL, self.epsilon, 1-self.epsilon)
        #log_al = cp.log(AL)
        #log_1m_al = cp.log(1-AL)

        cost = (-1 / m) * cp.sum(cp.multiply(Y, cp.log(AL)) + cp.multiply(1 - Y, cp.log(1 - AL)))

        cost = cp.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
        assert (cost.shape == ())

        return cost

    def loss_backward(self, AL, Y):
        #source: http://www.adeveloperdiary.com/data-science/deep-learning/neural-network-with-softmax-in-python/
        return cp.subtract(AL,Y)