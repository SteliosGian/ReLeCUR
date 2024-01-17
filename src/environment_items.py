import gym
import numpy as np
from gym import spaces
from surprise import SVD
from surprise import Reader
from surprise.accuracy import rmse
from surprise.model_selection import train_test_split
from surprise import Dataset


class rec_env_items(gym.Env):
    """" Recommendation system OpenAI gym environment"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df, items, cold_users, threshold_item_df, recs=5):
        '''
        df: The dataframe with the users, ratings and interactions
        items: The popularity items array
        cold_users: the cold users array
        threshold_item_df: The dataframe with the applied threshold
        recs: The number of shown items
        '''
        super(rec_env_items, self).__init__()

        self.reader = Reader(rating_scale=(0, 1))

        self.df = df
        self.items = items
        self.cold_users = cold_users
        self.item_df = threshold_item_df

        # Maximum recommendations
        self.max_steps = recs

        # Number of shown items
        num_of_recs = recs

        # Actions are the items
        self.action_space = spaces.Discrete(len(self.items))

        # Observation space
        self.observation_space = spaces.Box(low=0, high=np.max(self.items), shape=(num_of_recs,), dtype=np.int32)

        self.rmse_list = np.array([])
        self.rmse_ind = 0

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
            if (self.array_ind > 0) & (self.items[action] == self.ob[self.array_ind - 1]):

                np.put(self.ob, [self.array_ind], self.items[action])
            else:
                np.put(self.ob, [self.array_ind], self.items[action])

            self.curr_step += 1
            self.array_ind += 1
            self.done = self.curr_step == self.max_steps
            if self.done:
                self.helper()

        return self.ob, self.reward, self.done, {'RMSE': self.total_rmse}

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
                self.total_rmse = np.round(rmse(predictions, verbose=False), decimals=3)
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
