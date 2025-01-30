import numpy as np
import numpy.random as npr
import neuronav.utils as utils
from neuronav.agents.base_agent import BaseAgent


class TDSR(BaseAgent):
    """
    Implementación del método de diferencia temporal (considerando un paso) al algoritmo Successor Representator, 
    así como la variación B-pessimisitic. 
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        lr: float = 1e-1,          #tasa de aprendizaje
        gamma: float = 0.99,       #factor de descuento
        poltype: str = "softmax",  #política
        beta: float = 1e4,         #no se usa
        epsilon: float = 1e-1,     #valor de epsilon
        M_init=None,               #Matriz de representaciones sucesoras
        weights: str = "direct",   #vector de recompensas
        goal_biased_sr: bool = True,  #sirve para actualizar M considerando la variación B-pessimistic
        w_value: float = 1.0,         #omega
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

#se inicializa M
        if M_init is None:
            self.M = np.stack([np.identity(state_size) for i in range(action_size)])
        elif np.isscalar(M_init):
            self.M = np.stack(
                [M_init * npr.randn(state_size, state_size) for i in range(action_size)]
            )
        else:
            self.M = M_init

        self.w = np.zeros(state_size)

    def m_estimate(self, state):     #estimas la entrada de la matriz M[s,s',:]
        return self.M[:, state, :]
    
    def q_convergence(self):         #calculas la convergencia de la matriz de valores Q con la norma L2
        q_matrix = self.M @ self.w
        return np.linalg.norm(q_matrix,2)


    def q_estimate(self, state):       #estima el valor Q de una entrada
        return self.M[:, state, :] @ self.w

    def sample_action(self, state):    #selecciona una acción
        logits = self.q_estimate(state)
        return self.base_sample_action(logits)

    def update_w(self, state, state_1, reward, a):      #se actualizan la función de recompensa
        if self.weights == "direct":
            error = reward - self.w[state_1]
            self.w[state_1] += self.lr * error
       
        return np.linalg.norm(error)

    def update_sr(self, s, s_a, s_1, d, next_exp=None, prospective=False):  #se actualiza un vector de la matriz M

        if next_exp is None:
            
            s_a_1_optim = np.argmax(self.q_estimate(s_1))
            s_a_1_pessim = np.argmin(self.q_estimate(s_1))          

        I = utils.onehot(s, self.state_size)
        if d:
            m_error = (
                I + self.gamma * utils.onehot(s_1, self.state_size) - self.M[s_a, s, :]
            )
        else:
            if self.goal_biased_sr:     #implementación de B-pessimisitc

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

    def get_policy(self, M=None, goal=None):         #permite obtener la acción a realizar obteniendo los valores Q y llamando a un método de la clase padre
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
    def Q(self):         #se obtiene la matriz de valores Q
        return self.M @ self.w




class TDSR_RP(BaseAgent):
    """
    Implementation of one-step temporal difference (TD) Successor Representation Algorithm
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        lr: float = 1e-1,          #tasa de aprendizaje de las recompensas
        gamma: float = 0.99,       #factor de descuento
        poltype: str = "egp",      #politica
        beta: float = 1e4,         #no se usa
        epsilon: float = 1e-1,     #epsilon
        M_init=None,               #Matriz de representaciones sucesoras
        weights: str = "rew_pun",  #vector de recompensas/castigos
        goal_biased_sr: bool = True,  #sirve para considerar siempre los sucesores considerando la mejor acción del siguiente estado
        lr_p: float = 1e-1,        #tasa de aprendizaje de los castigos
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


        #se inicializa la matriz M
        if M_init is None:
            self.M = np.stack([np.identity(state_size) for i in range(action_size)])
        elif np.isscalar(M_init):
            self.M = np.stack(
                [M_init * npr.randn(state_size, state_size) for i in range(action_size)]
            )
        else:
            self.M = M_init

        #se inicializa el vector de las recompensas/castigos
        self.w = np.zeros(state_size)
        
    #estimas la entrada de la matriz M[s,s',:]
    def m_estimate(self, state):        
        return self.M[:, state, :]

    #calculas la convergencia de la matriz de valores Q con la norma L2
    def q_convergence(self):
        q_matrix = self.M @ self.w
        return np.linalg.norm(q_matrix,2)

    #estima el valor Q de una entrada
    def q_estimate(self, state):
        return self.M[:, state, :] @ self.w

    #selecciona una acción
    def sample_action(self, state):
        logits = self.q_estimate(state)
        return self.base_sample_action(logits)

    #se actualizan la función de recompensa
    def update_w(self, state, state_1, reward, a):
        
        if self.weights =="rew_pun":                #perimite usar diferentes tasas de aprendizare para recompensas o castigos
            
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

     #se actualiza un vector de la matriz M
    def update_sr(self, s, s_a, s_1, d, next_exp=None, prospective=False):
        # determines whether update is on-policy or off-policy
        if next_exp is None:
            
            s_a_1= np.argmax(self.q_estimate(s_1))

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
    def Q(self):             #obtienes toda la matriz de valores Q
        return self.M @ self.w






