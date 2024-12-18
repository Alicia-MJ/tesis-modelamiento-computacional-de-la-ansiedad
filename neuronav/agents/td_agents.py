import numpy as np
import numpy.random as npr
import neuronav.utils as utils
from neuronav.agents.base_agent import BaseAgent


class TDSR(BaseAgent):
    """
    Implementation of one-step temporal difference (TD) Successor Representation Algorithm
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        lr: float = 1e-1,
        gamma: float = 0.99,
        poltype: str = "softmax",
        beta: float = 1e4,
        epsilon: float = 1e-1,
        M_init=None,
        weights: str = "direct",
        goal_biased_sr: bool = True,
        w_value: float = 1.0,
    ):
        super().__init__(
            state_size,
            action_size,
            lr,
            gamma,
            poltype,
            beta,
            epsilon
        )
        self.weights = weights
        self.goal_biased_sr = goal_biased_sr
        self.w_value = w_value


        if M_init is None:
            self.M = np.stack([np.identity(state_size) for i in range(action_size)])
        elif np.isscalar(M_init):
            self.M = np.stack(
                [M_init * npr.randn(state_size, state_size) for i in range(action_size)]
            )
        else:
            self.M = M_init

        self.w = np.zeros(state_size)

    def m_estimate(self, state):
        return self.M[:, state, :]
    
    def q_convergence(self):
        q_matrix = self.M @ self.w
        return np.linalg.norm(q_matrix,2)


    def q_estimate(self, state):
        return self.M[:, state, :] @ self.w

    def sample_action(self, state):
        logits = self.q_estimate(state)
        return self.base_sample_action(logits)

    def update_w(self, state, state_1, reward, a):
        if self.weights == "direct":
            error = reward - self.w[state_1]
            self.w[state_1] += self.lr * error
       
        return np.linalg.norm(error)

    def update_sr(self, s, s_a, s_1, d, next_exp=None, prospective=False):
        # determines whether update is on-policy or off-policy
        if next_exp is None:
            
            s_a_1_optim = np.argmax(self.q_estimate(s_1))
            s_a_1_pessim = np.argmin(self.q_estimate(s_1))          

        #faltaría ajustar el código para cuando sí se pase el argumento de next_exp
        #else:
        #    s_a_1 = next_exp[1]

        I = utils.onehot(s, self.state_size)
        if d:
            m_error = (
                I + self.gamma * utils.onehot(s_1, self.state_size) - self.M[s_a, s, :]
            )
        else:
            if self.goal_biased_sr:

                next_m =  (       self.w_value * self.m_estimate(s_1)[s_a_1_optim]
                + (1 - self.w_value) * self.m_estimate(s_1)[s_a_1_pessim]            )

            else:
                next_m = self.m_estimate(s_1).mean(0)
            m_error = I + self.gamma * next_m - self.M[s_a, s, :]

        if not prospective:
            # actually perform update to SR if not prospective
            self.M[s_a, s, :] += self.lr * m_error
        return m_error

    def _update(self, current_exp, **kwargs):
        s, a, s_1, r, d = current_exp
        m_error = self.update_sr(s, a, s_1, d, **kwargs)
        w_error = self.update_w(s, s_1, r, a)
        #q_error = self.q_error(s, a, s_1, r, d)
        return m_error

    def get_policy(self, M=None, goal=None):
        if goal is None:
            goal = self.w

        if M is None:
            M = self.M

        Q = M @ goal
        return self.base_get_policy(Q)

    def get_M_states(self):
        # average M(a, s, s') according to policy to get M(s, s')
        policy = self.get_policy()
        M = np.diagonal(np.tensordot(policy.T, self.M, axes=1), axis1=0, axis2=1).T
        return M

    @property
    def Q(self):
        return self.M @ self.w




class TDSR_RP(BaseAgent):
    """
    Implementation of one-step temporal difference (TD) Successor Representation Algorithm
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        lr: float = 1e-1,
        gamma: float = 0.99,
        poltype: str = "softmax",
        beta: float = 1e4,
        epsilon: float = 1e-1,
        M_init=None,
        weights: str = "rew_pun",
        goal_biased_sr: bool = True,
        lr_p: float = 1e-1,
    ):
        super().__init__(
            state_size,
            action_size,
            lr,
            gamma,
            poltype,
            beta,
            epsilon
        )
        self.lr_p=lr_p
        self.weights = weights
        self.goal_biased_sr = goal_biased_sr



        if M_init is None:
            self.M = np.stack([np.identity(state_size) for i in range(action_size)])
        elif np.isscalar(M_init):
            self.M = np.stack(
                [M_init * npr.randn(state_size, state_size) for i in range(action_size)]
            )
        else:
            self.M = M_init


        self.w = np.zeros(state_size)

    def m_estimate(self, state):
        return self.M[:, state, :]
    
    def q_convergence(self):
        q_matrix = self.M @ self.w
        return np.linalg.norm(q_matrix,2)

    def q_estimate(self, state):
        return self.M[:, state, :] @ self.w

    def sample_action(self, state):
        logits = self.q_estimate(state)
        return self.base_sample_action(logits)

    def update_w(self, state, state_1, reward, a):
        
        if self.weights =="rew_pun":
            
            if reward>=0:
                error = reward - self.w[state_1]
                self.w[state_1] += self.lr * error
            elif reward<0:
                error = reward - self.w[state_1]
                self.w[state_1] += self.lr_p * error
                
        if self.weights == "direct":
            error = reward - self.w[state_1]
            self.w[state_1] += self.lr * error

        return np.linalg.norm(error)

    def update_sr(self, s, s_a, s_1, d, next_exp=None, prospective=False):
        # determines whether update is on-policy or off-policy
        if next_exp is None:
            
            s_a_1= np.argmax(self.q_estimate(s_1))


        #faltaría ajustar el código para cuando sí se pase el argumento de next_exp
        #else:
        #    s_a_1 = next_exp[1]

        I = utils.onehot(s, self.state_size)
        if d:
            m_error = (
                I + self.gamma * utils.onehot(s_1, self.state_size) - self.M[s_a, s, :]
            )
        else:
            if self.goal_biased_sr:

                next_m = self.m_estimate(s_1)[s_a_1]

            else:
                next_m = self.m_estimate(s_1).mean(0)

            m_error = I + self.gamma * next_m - self.M[s_a, s, :]

        if not prospective:
            # actually perform update to SR if not prospective
            self.M[s_a, s, :] += self.lr * m_error
        return m_error

    def _update(self, current_exp, **kwargs):
        s, a, s_1, r, d = current_exp
        m_error = self.update_sr(s, a, s_1, d, **kwargs)
        w_error = self.update_w(s, s_1, r, a)
        return m_error

    def get_policy(self, M=None, goal=None):
        if goal is None:
            goal = self.w

        if M is None:
            M = self.M

        Q = M @ goal
        return self.base_get_policy(Q)

    def get_M_states(self):
        # average M(a, s, s') according to policy to get M(s, s')
        policy = self.get_policy()
        M = np.diagonal(np.tensordot(policy.T, self.M, axes=1), axis1=0, axis2=1).T
        return M

    @property
    def Q(self):
        return self.M @ self.w




class TDSR_AB(BaseAgent):
    """
    Implementation of one-step temporal difference (TD) Successor Representation Algorithm
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        lr: float = 1e-1,
        gamma: float = 0.99,
        poltype: str = "softmax",
        beta: float = 1e4,
        epsilon: float = 1e-1,
        M_init=None,
        weights: str = "direct",
        goal_biased_sr: bool = True,
        w_value: float = 1.0,
        lr_p: float = 1e-1,
    ):
        super().__init__(
            state_size,
            action_size,
            lr,
            gamma,
            poltype,
            beta,
            epsilon,
            w_value,
        )
        self.lr_p=lr_p
        self.weights = weights
        self.goal_biased_sr = goal_biased_sr
        self.w_value = w_value

        if M_init is None:
            self.M = np.stack([np.identity(state_size) for i in range(action_size)])
        elif np.isscalar(M_init):
            self.M = np.stack(
                [M_init * npr.randn(state_size, state_size) for i in range(action_size)]
            )
        else:
            self.M = M_init

        self.w = np.zeros(state_size)

    def m_estimate(self, state):
        return self.M[:, state, :]

    def q_estimate(self, state):
        return self.M[:, state, :] @ self.w

    def sample_action(self, state):
        logits = self.q_estimate(state)
        return self.base_sample_action(logits)

    def update_w(self, state, state_1, reward, a):
        if self.weights =="rew_pun":

            if reward>=0:
                error = reward - self.w[state_1]
                self.w[state_1] += self.lr * error
            elif reward<0:
                error = reward - self.w[state_1]
                self.w[state_1] += self.lr_p * error
                
        if self.weights == "direct":
            error = reward - self.w[state_1]
            self.w[state_1] += self.lr * error

        return np.linalg.norm(error)

    def update_sr(self, s, s_a, s_1, d, r, next_exp=None, prospective=False):
        # determines whether update is on-policy or off-policy
        if next_exp is None:
            
            s_a_1_optim = np.argmax(self.q_estimate(s_1))
            s_a_1_pessim = np.argmin(self.q_estimate(s_1))          

        #faltaría ajustar el código para cuando sí se pase el argumento de next_exp
        #else:
        #    s_a_1 = next_exp[1]

        I = utils.onehot(s, self.state_size)
        if d:
            m_error = (
                I + self.gamma * utils.onehot(s_1, self.state_size) - self.M[s_a, s, :]
            )
        else:
            if self.goal_biased_sr:

                next_m =  (       self.w_value * self.m_estimate(s_1)[s_a_1_optim]
                + (1 - self.w_value) * self.m_estimate(s_1)[s_a_1_pessim]            )

            else:
                next_m = self.m_estimate(s_1).mean(0)
            m_error = I + self.gamma * next_m - self.M[s_a, s, :]

        if not prospective:
            # actually perform update to SR if not prospective
            if r >= 0:
                self.M[s_a, s, :] += self.lr * m_error
            elif r < 0:
                self.M[s_a, s, :] += self.lr_p * m_error

        return m_error

    def _update(self, current_exp, **kwargs):
        s, a, s_1, r, d = current_exp
        m_error = self.update_sr(s, a, s_1, d, r, **kwargs)
        w_error = self.update_w(s, s_1, r, a)
        return m_error

    def get_policy(self, M=None, goal=None):
        if goal is None:
            goal = self.w

        if M is None:
            M = self.M

        Q = M @ goal
        return self.base_get_policy(Q)

    def get_M_states(self):
        # average M(a, s, s') according to policy to get M(s, s')
        policy = self.get_policy()
        M = np.diagonal(np.tensordot(policy.T, self.M, axes=1), axis1=0, axis2=1).T
        return M

    @property
    def Q(self):
        return self.M @ self.w




class TDSR_ET(BaseAgent):
    """
    Implementation of one-step temporal difference (TD) Successor Representation Algorithm
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        lr: float = 1e-1,
        gamma: float = 0.99,
        poltype: str = "softmax",
        beta: float = 1e4,
        epsilon: float = 1e-1,
        M_init=None,
        weights: str = "direct",
        goal_biased_sr: bool = True,
        w_value: float = 1.0,
        lambd: float = 0.0,
        E_init=None,
    ):
        super().__init__(
            state_size,
            action_size,
            lr,
            gamma,
            poltype,
            beta,
            epsilon
        )
        self.weights = weights
        self.goal_biased_sr = goal_biased_sr
        self.w_value = w_value
        self.lambd = lambd

        if M_init is None:
            self.M = np.stack([np.identity(state_size) for i in range(action_size)])
        elif np.isscalar(M_init):
            self.M = np.stack(
                [M_init * npr.randn(state_size, state_size) for i in range(action_size)]
            )
        else:
            self.M = M_init

        self.w = np.zeros(state_size)


        if E_init is None:
            self.E = np.zeros((action_size, state_size))
        else:
            self.E = E_init
        

    def m_estimate(self, state):
        return self.M[:, state, :]
    
    def q_convergence(self):
        q_matrix = self.M @ self.w
        return np.linalg.norm(q_matrix,2)


    def q_estimate(self, state):
        return self.M[:, state, :] @ self.w

    def sample_action(self, state):
        logits = self.q_estimate(state)
        return self.base_sample_action(logits)

    def update_w(self, state, state_1, reward, a):
        if self.weights == "direct":
            error = reward - self.w[state_1]
            self.w[state_1] += self.lr * error
       
        return np.linalg.norm(error)


    def e_update(self, state, action, update):
        if update == "one":
           self.E[action, state] = self.E[action,state] +1
        else:
            for s in range(self.state_size):
                for a in range(self.action_size):
                    self.E[a, s] = self.gamma*self.lambd*self.E[a,s] 
       
    
    def e_estimate(self, s, s_a):
       return self.E[s_a, s]

    def update_sr(self, s, s_a, s_1, d, next_exp=None, prospective=False):
        # determines whether update is on-policy or off-policy
        if next_exp is None:
            
            s_a_1_optim = np.argmax(self.q_estimate(s_1))
            s_a_1_pessim = np.argmin(self.q_estimate(s_1))          

        #faltaría ajustar el código para cuando sí se pase el argumento de next_exp
        #else:
        #    s_a_1 = next_exp[1]

        I = utils.onehot(s, self.state_size)

        if d:
            m_error = (
                I + self.gamma * utils.onehot(s_1, self.state_size) - self.M[s_a, s, :]
            )
        else:
            if self.goal_biased_sr:

                next_m =  (       self.w_value * self.m_estimate(s_1)[s_a_1_optim]
                + (1 - self.w_value) * self.m_estimate(s_1)[s_a_1_pessim]            )

            else:
                next_m = self.m_estimate(s_1).mean(0)
            m_error = I + self.gamma * next_m - self.M[s_a, s, :]

        self.e_update(s,s_a,"one")

        if not prospective:
            # actually perform update to SR if not prospective
            
            for state in range(self.state_size):
                for action in range(self.action_size):
                    e_trace = self.e_estimate(state, action)
                    self.M[action, state, :] += self.lr * e_trace * m_error
            
        return m_error

    def _update(self, current_exp, **kwargs):
        s, a, s_1, r, d = current_exp
        m_error = self.update_sr(s, a, s_1, d, **kwargs)
        w_error = self.update_w(s, s_1, r, a)
        et_update = self.e_update( s, a, "all")

        return m_error

    def get_policy(self, M=None, goal=None):
        if goal is None:
            goal = self.w

        if M is None:
            M = self.M

        Q = M @ goal
        return self.base_get_policy(Q)

    def get_M_states(self):
        # average M(a, s, s') according to policy to get M(s, s')
        policy = self.get_policy()
        M = np.diagonal(np.tensordot(policy.T, self.M, axes=1), axis1=0, axis2=1).T
        return M

    @property
    def Q(self):
        return self.M @ self.w
