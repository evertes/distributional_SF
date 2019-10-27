"""
Learning distributional successor features in a noisy 2d enviroment.

"""


import numpy as np
import numpy.random as rnd
from numpy import exp, log
import scipy as sc
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.linalg import pinvh

from scipy.special import i0
from scipy.special import softmax

def ridge_regression(X, Y, tau=1.e-4, pinv_Var_X=None, return_pinv=False):

    Var_X = X.dot(X.T)
    l = tau*np.mean(np.diag(Var_X))
    if pinv_Var_X is None:
        pinv_Var_X = pinvh(Var_X + l*np.eye(Var_X.shape[0]))
    Cov_XY = np.tensordot(X, Y.T, (-1,0) )

    alpha = np.tensordot(pinv_Var_X, Cov_XY, (1,0)).T

    
    if return_pinv==True:
        return alpha, pinv_Var_X
    else:
        return alpha





class environment:
    def __init__(self, K=10**2, sig_Psi=0.1, sig_s=0.02, sig_obs=0.1):
        
        self.wall_coords = [np.array([[0.5, 0.], [0.5, 0.7]])]

        self.K = K
        self.sig_Psi = sig_Psi
        
        self.W_filt0 = None
        self.W_filt = np.zeros((self.K, 2*self.K+1))
        
        self.sig_s = sig_s
        self.sig_obs = sig_obs
        self.T = np.maximum(0,rnd.randn(self.K, self.K+1)*0.001)



    def check_walls(self, p0, p1, ind=0):
        # checks whether stepping from p0 to p1 crosses the internal wall
        if (p0==p1).all():
            return 0

        s1_x = p1[0] - p0[0]
        s1_y = p1[1] - p0[1]

        s2_x = self.wall_coords[ind][1,0] - self.wall_coords[ind][0,0]   
        s2_y = self.wall_coords[ind][1,1] - self.wall_coords[ind][0,1]
        
        u = (-s1_y * (p0[0] - self.wall_coords[ind][0,0]) + s1_x * (p0[1] - self.wall_coords[ind][0,1])) / (-s2_x * s1_y + s1_x * s2_y)
        t = ( s2_x * (p0[1] - self.wall_coords[ind][0,1]) - s2_y * (p0[0] - self.wall_coords[ind][0,0])) / (-s2_x * s1_y + s1_x * s2_y)

        
        if (u >= 0 and u <= 1 and t >= 0 and t <= 1):
            
            return 1
        else:
            return 0
    
    def random_walk(self, s0, sigm, T, wall=True):
    
        # boundary conditions [0,1]x[0,1]
    
        s0  = s0.reshape(2,)
        traj = np.zeros((2, T))
        traj[:,0] = s0

        for t in range(1,T):

            # random step
            noise = rnd.randn(2,)
            noise = noise/np.linalg.norm(noise)
            traj[:,t] = traj[:,t-1] + sigm *noise


            # check boundary conditions ---- Bouncy wall!
            traj[:,t] -= 2*(traj[:,t]< np.zeros((2,)))*traj[:,t]

            traj[:,t] -= 2*(traj[:,t]> np.ones((2,)))*(traj[:,t]-1.)

            # check collision with wall
            if wall is True:
                for ind in range(len(self.wall_coords)):
                    if self.check_walls(traj[:,t-1], traj[:,t], ind=ind):
                        # change sign for first component of 'noise' (state transition)
                        if ind<2:
                            noise[0] = -noise[0]
                            traj[:,t] = traj[:,t-1] + sigm *noise #traj[:,t-1]
                        else:
                            t -= 1


        return traj

    def random_angle_walk(self, s0, delta, T, wall=True):

        s0  = s0.reshape(2,)
        traj = np.zeros((2, T))
        traj[:,0] = s0

        actions = np.zeros(T-1,)

        for t in range(1,T):

            # random step
            angle = rnd.rand(1,)*2*np.pi

            traj[:,t] = traj[:,t-1] + delta * np.array([np.cos(angle), np.sin(angle)]).squeeze()


            # check boundary conditions ---- Bouncy wall!
            traj[:,t] -= 2*(traj[:,t]< np.zeros((2,)))*traj[:,t]

            traj[:,t] -= 2*(traj[:,t]> np.ones((2,)))*(traj[:,t]-1.)

            # check collision with wall
            if wall is True:
                for ind in range(len(self.wall_coords)):
                    if self.check_walls(traj[:,t-1], traj[:,t], ind=ind):
                        # change sign for first component of 'noise' (state transition)
                        if ind<2:
                            angle = (angle + np.pi) % (2*np.pi)
                            traj[:,t] = traj[:,t-1] + delta * np.array([np.cos(angle), np.sin(angle)]).squeeze()
                            #traj[:,t] = traj[:,t-1]

                            # check boundary conditions ---- Bouncy wall!
                            traj[:,t] -= 2*(traj[:,t]< np.zeros((2,)))*traj[:,t]
                            traj[:,t] -= 2*(traj[:,t]> np.ones((2,)))*(traj[:,t]-1.)

                            #traj[:,t] = traj[:,t-1]
                        else:
                            t -= 1
                            #traj[:,t] = traj[:,t-1]

            actions[t-1] = angle

        return traj, actions





    
    def gen_observations(self, s0, Tmax, wall=True):
        
        traj, actions = self.random_angle_walk(s0, self.sig_s, Tmax, wall)
        
        obs = traj + self.sig_obs * rnd.randn(*traj.shape)
        
        return traj, obs, actions
    
    
    def gen_sleep_samples_rand_angles(self, s0, Tmax, alpha):
        
        
        s_sleep = np.zeros((2,Tmax))

        
        s_sleep[:,0] = s0
        
        Psi_s = np.concatenate([ self.Psi_fun(s0.reshape(-1,1)), np.ones((1,1))])
        
        for t in range(1,Tmax):
            noise = rnd.randn(2,)
            s_sleep[:,t] = (alpha.dot(self.T.dot(Psi_s)).squeeze() 
                            + self.sig_s * noise/np.linalg.norm(noise) )
            
            Psi_s = np.concatenate([ self.Psi_fun(s_sleep[:,t].reshape(-1,1)), np.ones((1,1))])
        
            
        o_sleep = s_sleep + self.sig_obs *  rnd.randn(2, Tmax)
        
        
        return s_sleep, o_sleep

    def gen_sleep_samples(self, s0, Tmax, alpha):
        
        
        s_sleep = np.zeros((2,Tmax))

        
        s_sleep[:,0] = s0
        
        Psi_s = np.concatenate([ self.Psi_fun(s0.reshape(-1,1)), np.ones((1,1))])
        
        for t in range(1,Tmax):
            
            s_sleep[:,t] = (alpha.dot(self.T.dot(Psi_s)).squeeze() 
                            + self.sig_s * rnd.randn(2,))
            
            Psi_s = np.concatenate([ self.Psi_fun(s_sleep[:,t].reshape(-1,1)), np.ones((1,1))])
        
            
        o_sleep = s_sleep + self.sig_obs *  rnd.randn(2, Tmax)
        
        
        return s_sleep, o_sleep

    
    def gen_sleep_samples_mod(self, s0, Tmax, alpha):
        
        
        s_sleep = np.zeros((2,Tmax))

        
        s_sleep[:,0] = s0
        
        noise = rnd.randn(2,1)
        Psi_s = np.concatenate([ self.Psi_fun(s0.reshape(-1,1)+ self.sig_s * noise), np.ones((1,1))])
        
        for t in range(1,Tmax):
            
            s_sleep[:,t] = alpha.dot(self.T.dot(Psi_s)).squeeze() 

            noise = rnd.randn(2,1)

            Psi_s = np.concatenate([ self.Psi_fun(s_sleep[:,t].reshape(-1,1) +  self.sig_s * noise), np.ones((1,1))])
        
            
        o_sleep = s_sleep + self.sig_obs *  rnd.randn(2, Tmax)
        
        
        return s_sleep, o_sleep
    
    
    
    def Psi_fun(self, s):
        # Gaussian RBF
        # returns K dimensional rbf features with centers arranged on a grid, truncated at the mid wall

        
        K = self.K
        sig_psi = self.sig_Psi

        x1, y1 = np.meshgrid(np.linspace(0, 1, int(np.sqrt(K))), np.linspace(0, 1, int(np.sqrt(K))))
        grid_coord = np.concatenate([x1.reshape(-1,1), y1.reshape(-1,1)], axis=1)

        rbf = 1./(np.sqrt(2)*sig_psi) * exp(-1./(sig_psi**2) *
                                    np.sum((s[np.newaxis,:,:] - grid_coord[:,:,np.newaxis])**2, axis=1) ) 

        
        border_list_left = list(range(4,69,10))
        border_list_right =  list(range(5,69,10))

        rbf[border_list_left] = rbf[border_list_left] * np.heaviside(0.5-s[0,:], 1.).reshape(1,-1)
        rbf[border_list_right] = rbf[border_list_right] * np.heaviside(s[0,:]-0.5, 1.).reshape(1,-1)


        return rbf
   
    
    def firststep_filtering(self, o, s):
        # learn W0 first step of the filtering weights
        # o_t, o_t+1 -> s_t+1
        
        # add bias 
        o_bias = np.concatenate([o, np.ones((1,o.shape[1]))], axis=0)
        s_bias =  np.concatenate([s, np.ones((1,s.shape[1]))], axis=0)
        
        #outerprod = np.einsum('i..., j...->ij...', s_bias[:,:-1], o_bias[:,1:]
        #                     ).reshape(s_bias.shape[0]*o_bias.shape[0], -1)
        
        concat  = np.concatenate([s[:,:-1], o_bias[:,1:]], axis=0)
        
        s_pred = s[:,1:]
        
        W0 = ridge_regression(X=concat, Y=s_pred, tau=1.e-4)
        
        mu0 = W0.dot(concat)
        
    
        return W0, mu0
        
        
    
    def learn_filtering_weights(self, s, o, tau):
        
    
        # inputs
        s_bias = np.concatenate([s, np.ones((1,s.shape[1]))], axis=0)
        o_bias = np.concatenate([o, np.ones((1,o.shape[1]))], axis=0)
        
        W0, mu = self.firststep_filtering(o, s)
        
        
        W_list = [W0]
        err_list = []
        
        for it in (range(1, 10)):
            
            mu_bias = np.concatenate([mu, np.ones((1,mu.shape[1]))], axis=0)
        
            concat = np.concatenate([self.T.dot(mu_bias[:,:-1]), o_bias[:,it+1:]], axis=0)

            
            # outputs
            s_pred = s[:,it+1:]


            W = ridge_regression(X=concat, Y=s_pred, tau=tau)
            W_list.append(W) 
            
            # compute new mu
            mu = W.dot(concat)
        
            err = np.sum((mu - s_pred)**2)/s_pred.shape[1]
            err_list.append(err)
            
        self.W_filt0 = W_list[0]
        self.W_filt = W_list[-1]
            
        return W_list, err_list
    

    
    def run_filter(self, o, s0):
        
        W0 = self.W_filt
        W = self.W_filt
        
        o_bias = np.concatenate([o, np.ones((1,o.shape[1]))], axis=0)
        mu_traj = np.zeros((W0.shape[0], o.shape[1]-1))
        
        for t in range(o.shape[1]-1):
            
            if t==0:
                
                mu0 = np.concatenate([s0, np.ones((1,)) ]).squeeze()
                
                #mu =  W0.dot(np.outer(mu0, o_bias[:,t+1]).reshape(len(o_bias[:,t])*len(mu0)))
                # replace outerprod with concat
                mu = W0.dot(np.concatenate([s0, o_bias[:,t+1] ], axis=0))
            else:
                mu_bias = np.concatenate([mu, np.ones((1,))], axis=0)

                #mu = W.dot(np.outer(mu_bias, o_bias[:,t+1]).reshape(-1,))
                mu = W.dot(np.concatenate([self.T.dot(mu_bias), o_bias[:,t+1]], axis=0))
            
            mu_traj[:,t] = mu   
        
        return mu_traj
    
    
    
    def decode_mean(self, s, s_feats):
        # return linear weights alpha to decode s from features Psi(s)
        
        alpha = ridge_regression(X=s_feats, Y=s, tau=1.e-4)
        err = np.sum((alpha.dot(s_feats) - s)**2)/s.shape[1]
        
        return alpha, err
    
    def update_dynamics_model(self, mu_feats, tau=1.e-4):
        
        # computes closed form updates for the transition matrix T, using DDC posteriors only
        
        mu_feats_prev = mu_feats[:,:-1]
        mu_feats_next = mu_feats[:,1:]

        #inputs
        inputs = np.concatenate([mu_feats_prev,  np.ones((1,mu_feats_prev.shape[1]))], axis=0)
        
        self.T = ridge_regression(X=inputs, Y=mu_feats_next, tau=tau)
        
        err = np.mean((self.T.dot(inputs) - mu_feats_next)**2)
        
        return err

    def grad_update_dynamics_model(self, mu_feats, eps = 1.e-3, tau=1.e-4):
        
        # computes gradient updates for the transition matrix T, using DDC posteriors only
        
        mu_feats_prev = mu_feats[:,:-1]
        mu_feats_next = mu_feats[:,1:]

        inputs = np.concatenate([mu_feats_prev,  np.ones((1,mu_feats_prev.shape[1]))], axis=0)

        for it in range(inputs.shape[1]):

        	self.T -= eps * np.outer((self.T.dot(inputs[:,it]) - mu_feats_next[:,it]), inputs[:,it])
                
        err = np.mean((self.T.dot(inputs) - mu_feats_next)**2)

        return err
    
    def wake_sleep(self, alpha, Niter, eps, tau):
                

        for it in tqdm(range(Niter)):
        
            # SLEEP
            # using fixed stepsize in s dynamics
            s_sleep, o_sleep = self.gen_sleep_samples_rand_angles(s0=rnd.rand(2,), Tmax=30000, alpha=alpha)

            traj_feats = self.Psi_fun(s_sleep)
            obs_feats = self.Psi_fun(o_sleep)

            W_list, err_list = self.learn_filtering_weights(traj_feats, obs_feats, tau=tau)


            # WAKE

            _, obs_wake, _ = self.gen_observations(s0=rnd.rand(2,), Tmax=50000, wall=True)

            obs_wake_feats = self.Psi_fun(obs_wake)
            # run filtering on real observations
            mu_feats =  self.run_filter( obs_wake_feats, traj_feats[:,0])

            # update dynamics model
            if eps is None:
                err = self.update_dynamics_model(mu_feats, tau=tau)
            else:
                err = self.grad_update_dynamics_model(mu_feats, eps = eps, tau=tau)
            

    
    def SF_TD_sleep(self, s_sleep, eta, gamma):

        # update SF parameters using TD on a latent sleep sequence
        # might want tu use several sequences/episodes

        #initalize m
        m = rnd.rand(self.K,self.K)

        #for n in range(Nepisode):
        
        #s =  random_walk(rnd.rand(2,), sigm=0.05, T=1000)
    
        deltas = []

        s_sleep_feats = self.Psi_fun(s_sleep)

        for t in range(s_sleep.shape[1]-1):
            
            delta_TD = s_sleep_feats[:,t+1] + gamma* m.dot(s_sleep_feats[:,t+1]) - m.dot(s_sleep_feats[:,t])
            
            delta_m = delta_TD.dot(s_sleep_feats[:,t].T)
            
            m += eta * delta_m

            deltas.append(np.linalg.norm(delta_m))

        return m, deltas

    def compute_reward_weights_obs(self, s_sleep, o_sleep, w_s):

        # true reward function defined by w_s
        r = w_s.T.dot(self.Psi_fun(s_sleep))

        # regression to find w_obs:
        w_obs = ridge_regression(self.Psi_fun(o_sleep), r, tau=1.e-4) 

        return w_obs



    def SF_TD_wake(self, mu_feats, eta, gamma):
        # ~ equivalent to TD_sleep if applied on trained model, matching the data p_data(o)
        # if p_model(o) != p_data(o): the two are not equivalent but might be better to use TD wake
        #                             -> does not rely on simulations from the incorrect model?

        m = self.SF_TD_sleep(mu_feats, eta, gamma)
        
        return m

    def SF_dynamics(self, mu_feats, gamma, Nit):
        # run dynamics with recurrent weights gamma* env.T


        u = np.zeros((self.K,1))
        dt = 0.1


        delta_norm = []

        A = self.T[:self.K, :self.K]

        for k in range(Nit):

            u  +=  dt *( -u + gamma* A.dot(u) + mu_feats)
            
            delta_norm.append(np.linalg.norm( -u + gamma* (A.dot(u)+ self.T[:self.K,-1] ) + mu_feats ))


        return u, delta_norm

    def MC_SF_estimate(Tep, Nepisode):

        # set of locations to evaluate the SF-s at
        x, y = np.meshgrid(np.linspace(0, 1, int(np.sqrt(self.K))), np.linspace(0, 1, int(np.sqrt(self.K))))
        grid_coord = np.concatenate([x.reshape(-1,1), y.reshape(-1,1)], axis=1)


        SF_grid = np.zeros((len(grid_coord),))
        #T = 1000
        #Nepisode = 100

        for ind, s0 in enumerate(grid_coord):
            
            SF = 0 

            for episode in range(Nepisode):

                s = self.random_walk(self, s0, self.sig_s, Tep, wall=True)

                for t in range(1, Tep):
                    #r = w00.T.dot(Psi_fun(s[:,t]))
                    SF += gamma**(t-1) * self.Psi_fun(s[:,t])
            
            SF_grid[ind] = SF/Nepisode

        return SF_grid


    def Phi_act(self, act, kappa=2.):
        
        centers = np.linspace(0, np.pi*2, 10, endpoint=False)
        y = np.exp(kappa*np.cos(act[np.newaxis,:] - centers[:,np.newaxis]))/(2*np.pi*i0(kappa))

        return y


    def train_state_action_model_offline(self, traj_feats, actions, tau=1.e-3 ):

        # traj, actions contain the sequence of sates and actions, traj[0] initial state

        # joint features on s and a

        traj_feats = np.concatenate([traj_feats,  np.ones((1,traj_feats.shape[1]))], axis=0)
        act_feats = np.concatenate([self.Phi_act(actions), np.ones((1,len(actions)))], axis=0)

        outer_prod = (traj_feats[:,np.newaxis,:-1] * act_feats[np.newaxis,:,:]).reshape(
                                              traj_feats.shape[0]*act_feats.shape[0],-1)
        
        # next states
        target =  traj_feats[:-1,1:] #self.Psi_fun(traj[:,1:])

        P = ridge_regression(X=outer_prod, Y=target, tau=tau)

        err = np.mean((P.dot(outer_prod) - target)**2)/np.mean(target**2)

        self.P = P

        return P, err

    def train_state_action_model(self, traj_feats, actions, eps=1.e-4 ):

        # traj, actions contain the sequence of sates and actions, traj[0] initial state

        # joint features on s and a

        traj_feats_b = np.concatenate([traj_feats,  np.ones((1,traj_feats.shape[1]))], axis=0)
        act_feats = self.Phi_act(actions)

        outer_prod = (traj_feats[:,np.newaxis,:-1] * act_feats[np.newaxis,:,:]).reshape(
                                            traj_feats_b.shape[0]*act_feats.shape[0],-1)

        # next states
        target =  traj_feats[:,1:] 

        P = np.zeros((target.shape[0], outer_prod.shape[0]))
        err = []
        batchsize = 1
        for epoch in range(1):
            for it in tqdm(range(int(target.shape[1]/batchsize))):

                P -= eps *np.outer((P.dot(outer_prod[:,it]) - target[:,it]), outer_prod[:,it])  


            err.append(np.mean((P.dot(outer_prod) - target)**2)/np.mean(target**2))

        self.P = P

        return P, err
       
    def predict_next_state(self, s, a, P, alpha):

        traj_feats =  np.concatenate([self.Psi_fun(s),  np.ones((1,s.shape[1]))], axis=0)
        act_feats =   np.concatenate([self.Phi_act(a), np.ones((1,len(a)))], axis=0)


        outer_prod = (traj_feats[:,np.newaxis,:] * act_feats[np.newaxis,:,:]).reshape(traj_feats.shape[0]*act_feats.shape[0],-1)

        Psi_next = P.dot(outer_prod)

        s_next = alpha.dot(Psi_next)

        return s_next
    
    def Q_sa(self, s_feats, a, P, SF_mtx, gamma, w_rew, r=0):

        # return state-action values for s and a, either s OR a can be a vector, returning a Q vector of the same size

        s_feats_b = np.concatenate([s_feats,  np.ones((1,s_feats.shape[1]))], axis=0)
        a_feats = np.concatenate([self.Phi_act(a), np.ones((1,len(a)))], axis=0)

        #self.Phi_act(a)


        outer_prod = (s_feats_b[:,np.newaxis,:] * a_feats[np.newaxis,:,:]).reshape(s_feats_b.shape[0]*a_feats.shape[0],-1)

        Q = r + gamma* w_rew.T.dot(SF_mtx.dot(P.dot(outer_prod)))

        return Q

    def choose_action(self, s_feats, r, P, SF_mtx, gamma, w_rew,  beta=None):
        # choose action either by argmax of Q or by softmax if beta is given

        actions = np.linspace(0,np.pi*2, 30)

        # compute Q values for actions
        Q = self.Q_sa(s_feats, actions, P, SF_mtx, gamma, w_rew, r=r)

        if beta is None:
            a = actions[np.argmax(Q)]
        else:
            prob = softmax(beta*Q)
            a = rnd.choice(actions, size=1, p=prob)

        return a

    def update_state(self, s, a, wall=True):
        r = 0
        delta = 0.06
        s_next = s + delta * np.array([np.cos(a), np.sin(a)]).squeeze()


        # check boundary conditions ---- bouncy wall
        if (s_next.squeeze()< np.zeros((2,))).any() or (s_next.squeeze()> np.ones((2,))).any(): 
            r -= 10

        s_next -= 2*(s_next.squeeze()< np.zeros((2,)))*s_next

        s_next -= 2*(s_next.squeeze()> np.ones((2,)))*(s_next-1.)

        # check collision with wall
        if wall is True:
            for ind in range(len(self.wall_coords)):
                if self.check_walls(s.squeeze(), s_next.squeeze(), ind=ind):

                    r-=10
                    # change sign for first component of 'noise' (state transition)
                    if ind<3:
                        a = (a + np.pi) % (2*np.pi)
                        s_next = s + delta * np.array([np.cos(a), np.sin(a)]).squeeze()

                        # check bounday again
                        s_next -= 2*(s_next.squeeze()< np.zeros((2,)))*s_next
                        s_next -= 2*(s_next.squeeze()> np.ones((2,)))*(s_next-1.)


                    else:
                        s_next = s

        return s_next, r


    def sample_episode_latent(self, Nsteps, s, SF_mtx, gamma, w_rew, P,  beta=None):
        # chooses actions according to current policy/SF and reward vector w
        # assumes that the true state s is known


        episode = np.zeros((Nsteps,2))
        episode[0,:] = s.squeeze()

        total_reward = 0
        r_walls = 0
        for it in range(1,Nsteps):
            # observe rewards if any
            r = w_rew.T.dot(self.Psi_fun(s.reshape(-1,1))) #+ r_walls
            total_reward += r
            # choose acton
            a = self.choose_action(self.Psi_fun(s.reshape(-1,1)), r, P, SF_mtx, gamma, w_rew,  beta=beta)

            s, r_walls = self.update_state(s, a)


            episode[it,:] = s.squeeze()

        return episode, total_reward


    def compute_T_offline(self, traj):
        Psi_t_bias = np.concatenate([self.Psi_fun(traj[:,:-1]), np.ones((1, traj[:,:-1].shape[1]))], axis=0)
        Psi_tnext = self.Psi_fun(traj[:,1:])
        T_samples =  ridge_regression(X=Psi_t_bias[:,:], Y = Psi_tnext[:,:], tau=1.e-2 ) 

        return T_samples


    def update_T_online(self, T, traj_feats, eps):

        #traj_feats = self.Psi_fun(traj)
        
        feats_prev = traj_feats[:,:-1]
        feats_next = traj_feats[:,1:]

        #inputs
        inputs = np.concatenate([feats_prev,  np.ones((1,feats_prev.shape[1]))], axis=0)

        for it in range(inputs.shape[1]):

            T -= eps * np.outer((T.dot(inputs[:,it]) - feats_next[:,it]), inputs[:,it])

        return T

        

    def get_SF_matrix(self, T, gamma):
        # returns the SF matrix  in closed form using the dynamics matrix T, ignores the bias term in T

        SF_analytic = np.linalg.inv(np.eye(self.K) - gamma * T[:,:self.K])

        return SF_analytic

    def policy_iteration_latent(self, N, T, gamma, w_rew, P, beta=None):

        SF = self.get_SF_matrix( T, gamma)

        rewards = []
        for it in tqdm(range(N)):
            
            s_start = rnd.rand(2,)
            isinwall = (s_start[0]>0.48 and s_start[0]<0.52) and (s_start[0]<0.7) # checks if starting point falls in the wall

            while isinwall:
                s_start = rnd.rand(2,)
                isinwall = (s_start[0]>0.48 and s_start[0]<0.52) and (s_start[0]<0.7)


            episode, total_reward = self.sample_episode_latent(Nsteps=500, s=s_start,
                                    SF_mtx=SF, gamma=gamma, w_rew=w_rew, P=P,  beta=beta)


            T = self.update_T_online(T, self.Psi_fun(episode.T), eps=1.e-4*(total_reward>1))

            SF = self.get_SF_matrix(T, gamma)

            rewards.append(total_reward)

        return episode, total_reward, T, SF



    ''' Functions for policy imrpovement using noisy observed states ''' 

    def sample_episode_obs(self, Nsteps, s, SF_mtx, gamma, w_rew, w_obs, P,  beta=None):
        # Chooses actions according to current policy/SF and reward vector w
        # having access to noisy observations only and actions updating the true latent state


        episode_latent = np.zeros((Nsteps,2))
        episode_latent[0,:] = s.squeeze()

        episode_obs = np.zeros((Nsteps,2))

        # get observable state o
        o = s +  self.sig_obs *  rnd.randn(2,)
        episode_obs[0,:] = o.squeeze()

        total_reward = 0

        for it in range(1,Nsteps):

            # observe rewards if any (according to true reward function)
            r = w_rew.T.dot(self.Psi_fun(s.reshape(-1,1)))
            total_reward += r
            
            # choose acton based on observed sate o
            a = self.choose_action(self.Psi_fun(o.reshape(-1,1)), r, P, SF_mtx, gamma, w_obs,  beta=beta)

            # update latent state using action a 
            s, r_walls = self.update_state(s, a)

            # get observable state o
            o = s +  self.sig_obs *  rnd.randn(2,)

            # save latent and observed states
            episode_latent[it,:] = s.squeeze()
            episode_obs[it,:] = o.squeeze()


        return episode_latent, episode_obs, total_reward

    def policy_iteration_observed(self, N, T_obs, gamma, w_rew,  w_obs, W_oa, beta=None):

        SF_obs = self.get_SF_matrix(T_obs, gamma)

        rewards = []
        for it in tqdm(range(N)):
            
            s_start = rnd.rand(2,)
            # Checks if starting point falls in the wall
            isinwall = (s_start[0]>0.48 and s_start[0]<0.52) and (s_start[0]<0.7) 

            while isinwall:
                s_start = rnd.rand(2,)
                isinwall = (s_start[0]>0.48 and s_start[0]<0.52) and (s_start[0]<0.7)


            episode_latent, episode_obs, total_reward = self.sample_episode_obs(Nsteps=500, s=s_start, SF_mtx=SF_obs, gamma=gamma,
                                                                                 w_rew=w_rew, w_obs=w_obs, P=W_oa,  beta=beta)


            T_obs = self.update_T_online(T_obs, self.Psi_fun(episode_obs.T), eps=1.e-4*(total_reward>1))

            SF_obs = self.get_SF_matrix(T_obs, gamma)


            rewards.append(total_reward)

        return episode_latent, episode_obs, total_reward, T_obs, SF_obs


    ### with inference

    def sample_episode(self, Nsteps, s, T, gamma, w_rew, w_mu, P,  beta=None):
        ''' Chooses actions according to current policy/SF and reward vector w
            inferring latent state and using actions to update the true latent state
        '''

        SF_mtx = self.get_SF_matrix(T, gamma)

        episode_latent = np.zeros((Nsteps,2))
        episode_latent[0,:] = s.squeeze()

        episode_inferred = np.zeros((Nsteps,self.K))

        # get observable state o
        o = s +  self.sig_obs *  rnd.randn(2,)
        #episode_obs[0,:] = o.squeeze()

        # init inferred state to observed feature
        mu = self.Psi_fun(o.reshape(-1,1))
        episode_inferred[0,:] = mu.squeeze()

        total_reward = 0

        for it in range(1,Nsteps):

            # observe rewards if any (according to true reward function)
            r = w_rew.T.dot(self.Psi_fun(s.reshape(-1,1)))
            total_reward += r


            # choose acton based on inferred state mu
            a = self.choose_action(mu.reshape(-1,1), r, P, SF_mtx, gamma, w_mu, beta=beta)

            # update latent state using action a 
            s, r_walls = self.update_state(s, a)

            # get observable state o
            o = s +  self.sig_obs *  rnd.randn(2,)


            #update inferred state -- use fixed self.T or use the one evolving with the policy??
            mu_bias = np.concatenate([mu, np.ones((1,1))], axis=0)
            o_bias =  np.concatenate([self.Psi_fun(o.reshape(-1,1)), np.ones((1,1))], axis=0)
            mu = self.W_filt.dot(np.concatenate([self.T.dot(mu_bias), o_bias], axis=0))
            #mu = (0.7*T.dot(mu_bias))[:100] + self.W_filt[:,-101:].dot(o_bias)

            
            # save latent and observed states
            episode_latent[it,:] = s.squeeze()
            episode_inferred[it,:] = mu.squeeze()


        return episode_latent, episode_inferred, total_reward


    def policy_iteration(self, N, T_mu, gamma, w_rew, w_mu, W_mua,alpha, beta=None):


        rewards = []
        for it in tqdm(range(N)):
            
            s_start = rnd.rand(2,)
            # Checks if starting point falls in the wall
            isinwall = (s_start[0]>0.48 and s_start[0]<0.52) and (s_start[0]<0.7) 

            while isinwall:
                s_start = rnd.rand(2,)
                isinwall = (s_start[0]>0.48 and s_start[0]<0.52) and (s_start[0]<0.7)


            episode_latent, episode_inferred, total_reward = self.sample_episode(Nsteps=500, s=s_start, T=T_mu, gamma=gamma,
                                                                                 w_rew=w_rew, w_mu=w_mu, P=W_mua,  beta=beta)


            T_mu = self.update_T_online(T_mu, episode_inferred.T, eps=1.e-4*((total_reward>1)))


            rewards.append(total_reward)

            '''
            if it%10==0:
                
                rews_mu, rews_obs, rew_latent = self.policy_evaluation(Nepisode=100, T_mu=T_mu, SF_obs=SF_policy_obs, SF_latent=SF_policy, 
                 gamma=0.99, w_rew=w_rew, w_mu=w_mu, w_obs=w_obs, W_mua=W_mua, W_oa=W_oa, W_sa=W_sa, beta=5 )
            '''


        SF_mu = self.get_SF_matrix(T_mu, gamma)

        return episode_latent, episode_inferred, rewards, T_mu, SF_mu


    def policy_evaluation(self, Nepisode, T_mu, SF_obs, SF_latent, gamma, w_rew, w_mu, w_obs, W_mua, W_oa, W_sa, beta ):
       
        ''' Sample N episodes from different initial states, compare avg collected rewards '''

        rews_mu = np.zeros((Nepisode,))
        rews_obs = np.zeros((Nepisode,))
        rews_latent = np.zeros((Nepisode,))

        for it in tqdm(range(Nepisode)):
            
            s_start = rnd.rand(2,)
            # Checks if starting point falls in the wall
            isinwall = (s_start[0]>0.48 and s_start[0]<0.52) and (s_start[0]<0.7) 

            while isinwall:
                s_start = rnd.rand(2,)
                isinwall = (s_start[0]>0.48 and s_start[0]<0.52) and (s_start[0]<0.7)



            # with inferred states
            _, _,  r_mu = self.sample_episode(Nsteps=500, s=s_start, T=T_mu, gamma=gamma,
                                            w_rew=w_rew, w_mu=w_mu, P=W_mua,  beta=beta)

            # with observations
            _, _, r_obs = self.sample_episode_obs(Nsteps=500, s=s_start, SF_mtx=SF_obs, gamma=gamma,
                                                  w_rew=w_rew, w_obs=w_obs, P=W_oa,  beta=beta)

            # with latent states
            _, r_latent = self.sample_episode_latent(Nsteps=500, s=s_start, SF_mtx=SF_latent,
                                                gamma=gamma, w_rew=w_rew, P=W_sa,  beta=beta)


            rews_mu[it] = r_mu
            rews_obs[it] = r_obs
            rews_latent[it] = r_latent

        return rews_mu, rews_obs, rews_latent
