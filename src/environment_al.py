import gym
import numpy as np
from gym import spaces
from surprise import SVD
from surprise import Reader
from surprise.accuracy import rmse
from surprise.model_selection import train_test_split
from surprise import Dataset


class rec_env_AL(gym.Env):
    """" Recommendation system OpenAI gym environment"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df, pop_items, gini_items, popgini_items, ent_items, popent_items, error_items, poperror_items, variance_items, popvar_items, cold_users, threshold_item_df, recs=5):
        '''
        df: The dataframe with the userId, itemId and the interaction
        pop_items: the popularity items array
        gini_items: the gini items array
        popgini_items: the popgini items array
        ent_items: the entropy items array
        popent_items: the popent items array
        error_items: the error items array
        poperror_items: the poperror items array
        variance_items: the variance items array
        popvar_items: the popvar items array
        cold_users: the cold users array
        threshold_item_df: the dataframe of with the applied threshold
        recs: the number of shown items
        '''
        super(rec_env_AL, self).__init__()

        self.reader = Reader(rating_scale=(0, 1))

        self.df = df

        self.pop_items = pop_items
        self.gini_items = gini_items
        self.popgini_items = popgini_items
        self.ent_items = ent_items
        self.popent_items = popent_items
        self.error_items = error_items
        self.poperror_items = poperror_items
        self.variance_items = variance_items
        self.popvar_items = popvar_items

        self.cold_users = cold_users
        self.item_df = threshold_item_df

        # Maximum recommendations
        self.max_steps = recs

        # Number of AL methods
        num_of_methods = 9

        # Number of shown items
        num_of_recs = recs

        # Actions are the items
        self.action_space = spaces.Discrete(num_of_methods)

        # Observation space
        self.observation_space = spaces.Box(low=0, high=np.max(self.item_df), shape=(num_of_recs,), dtype=np.int32)
        self.rmse_list = np.array([], dtype=float)
        self.rmse_ind = -1

    def step(self, action):
        """Run one timestep of the environment's dynamics.
        Accepts an action and returns a tuple (observation, reward, done, info).
        # Arguments
            action (object): An action provided by the environment.
        # Returns
            observation (object): Agent's observation of the current environment.
            reward (float) : Amount of reward returned after previous action.
            done (boolean): Whether the episode has ended, in which case further step() calls will return undefined results.
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """

        if self.done:
            return self.ob, self.reward, self.done, {}
        else:

        ### ACTION 0 = PopGini strategy
            if action == 0:

                np.put(self.ob, [self.array_ind], self.popgini_items[self.popgini_ind])
                self.popgini_ind += 1
                self.curr_step += 1
                self.array_ind += 1

        ### ACTION 1 = Entropy strategy
            elif action == 1:
                np.put(self.ob, [self.array_ind], self.ent_items[self.ent_ind])
                self.ent_ind += 1
                self.curr_step += 1
                self.array_ind += 1

        ### Action 2 = PopEnt strategy
            elif action == 2:
                np.put(self.ob, [self.array_ind], self.popent_items[self.popent_ind])
                self.popent_ind += 1
                self.curr_step += 1
                self.array_ind += 1

        ### Action 3 = Popularity strategy
            elif action == 3:
                np.put(self.ob, [self.array_ind], self.pop_items[self.pop_ind])
                self.pop_ind += 1
                self.curr_step += 1
                self.array_ind += 1

        ### ACTION 4 = PopError strategy
            elif action == 4:
                np.put(self.ob, [self.array_ind], self.poperror_items[self.poperror_ind])
                self.poperror_ind += 1
                self.curr_step += 1
                self.array_ind += 1

        ### ACTION 5 = Variance strategy
            elif action == 5:
                np.put(self.ob, [self.array_ind], self.variance_items[self.variance_ind])
                self.variance_ind += 1
                self.curr_step += 1
                self.array_ind += 1

        ### ACTION 6 = PopVar strategy
            elif action == 6:
                np.put(self.ob, [self.array_ind], self.popvar_items[self.popvar_ind])
                self.popvar_ind += 1
                self.curr_step += 1
                self.array_ind += 1

        ### ACTION 7 = Gini strategy
            elif action == 7:
                np.put(self.ob, [self.array_ind], self.gini_items[self.gini_ind])
                self.gini_ind += 1
                self.curr_step += 1
                self.array_ind += 1

        ### ACTION 8 = Error strategy
            elif action == 8:
                np.put(self.ob, [self.array_ind], self.error_items[self.error_ind])
                self.error_ind += 1
                self.curr_step += 1
                self.array_ind += 1

        self.done = self.curr_step == self.max_steps
        if self.done:
            self.helper()
        return self.ob, self.reward, self.done, {}

    def helper(self):
        algo = SVD(n_factors=10, n_epochs=100, lr_all=0.001, reg_all=0.01)
        train_df = self.df[(~self.df.userId.isin(self.cold_users)) | (self.df.itemId.isin(self.ob))]

        cold_users_interacted = np.array(self.df[(self.df.userId.isin(self.cold_users)) & (self.df.itemId.isin(self.ob))]['userId'])
        test_df = self.df[(self.df.userId.isin(cold_users_interacted)) & (~self.df.itemId.isin(self.ob))]

        if test_df.empty:
            self.total_rmse = 1
        else:
            try:
                trainset = Dataset.load_from_df(train_df, self.reader)
                trainset = trainset.build_full_trainset()
                testset = Dataset.load_from_df(test_df, self.reader)

                _, testset = train_test_split(testset, test_size=1.0)

                predictions = algo.fit(trainset).test(testset)
                self.total_rmse = np.round(rmse(predictions, verbose=False), 3)
            except ValueError:
                self.total_rmse = 1

            self.reward = np.round(1/self.total_rmse, 3)

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        # Returns
            observation (object): The initial observation of the space. Initial reward is assumed to be 0.
        """

        self.curr_step = 0
        self.done = False

        self.ob = np.zeros(self.max_steps, dtype=int)

        # indices
        self.gini_ind = 0
        self.popgini_ind = 0
        self.ent_ind = 0
        self.popent_ind = 0
        self.pop_ind = 0
        self.error_ind = 0
        self.poperror_ind = 0
        self.variance_ind = 0
        self.popvar_ind = 0

        self.array_ind = 0

        # reward
        self.reward = 0.0

        self.total_rmse = 0

        return self.ob

    def render(self, mode='human', close=False):
        """Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.)
        # Arguments
            mode (str): The mode to render with.
            close (bool): Close all open renderings.
        """
        pass
