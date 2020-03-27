from rl.policy import EpsGreedyQPolicy


class DecayingEpsGreedyQPolicy(EpsGreedyQPolicy):
    def __init__(self, eps=.1, decay=0):
        super().__init__(eps)
        self.decay = decay

    def select_action(self, q_values):
        action = super().select_action(q_values)
        self.eps = self.eps * self.decay
        return action
