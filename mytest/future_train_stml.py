# -*- coding: utf-8 -*-

import click
import gym_trading  #必须引入才自动注册
import gym
import numpy as np
import pandas as pd
import random
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop, Adam
import logging

from collections import deque
from keras import regularizers

from keras import losses
from keras.layers.normalization import BatchNormalization


log = logging.getLogger(__name__)
logging.basicConfig()
log.setLevel(logging.INFO)
log.info('%s logger started.',__name__)

PLOT_AFTER_ROUND=1

import keras.backend as T
from keras import losses
def constrainedCrossEntropy(ytrue, ypred):
  ypred = T.clip(ypred, 0.001, 0.9999)
  return losses.categorical_crossentropy(ytrue, ypred)


class DQN:
    def __init__(self, env):
        self.env = env
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95 #FIX
        self.epsilon = 1.0
        self.epsilon_min = 0.01 #FIX
        self.epsilon_decay = 0.995 #FIX
        self.learning_rate = 0.0001
        self.tau = .125
        self.batch_size = 128 # 64 TODO
        self.state_size = self.env.observation_space.shape[0]

        self.model = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        model = Sequential()
        model.add(LSTM(64,
                       input_shape=(1, self.state_size),
                       return_sequences=True,
                       stateful=False))
        model.add(Dropout(0.5))
        model.add(LSTM(64,
                       input_shape=(1, self.state_size),
                       return_sequences=False,
                       stateful=False))
        model.add(Dropout(0.5))
        #model.add(Dense(self.env.action_space.n,use_bias=True,
        #                kernel_initializer='lecun_uniform',
        #                kernel_regularizer=regularizers.l2(0.00001),
        #                bias_regularizer=regularizers.l2(0.00001)))
        model.add(Dense(self.env.action_space.n, kernel_initializer='lecun_uniform'))
        model.add(BatchNormalization())
        model.add(Activation('linear'))

        #adam = Adam(lr=self.learning_rate,clipnorm=1., clipvalue=0.5)
        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=adam)
        return model


    def act(self, state):
        """Acting Policy of the DQNAgent
        """
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        qval = self.model.predict(state, batch_size=1,verbose=0)
        assert not np.any(np.isnan(qval))
        return np.argmax(qval)

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        """Memory Management and training of the agent
        """
        if len(self.memory) < self.batch_size:
            return

        state, action, reward, next_state, done = self._get_batches()
        reward += (self.gamma
                   * np.logical_not(done)
                   * np.amax(self.model.predict(next_state.reshape(self.batch_size,1,self.state_size),verbose=False),axis=1))
        q_target = self.target_model.predict(state.reshape(self.batch_size,1,self.state_size),verbose=False)

        _ = pd.Series(action)
        one_hot = pd.get_dummies(_).as_matrix()
        action_batch = np.where(one_hot == 1)
        # print "batch:",action_batch
        q_target[action_batch] = reward
        return self.model.fit(state.reshape(self.batch_size,1,self.state_size), q_target,
                              batch_size=self.batch_size,
                              epochs=1,
                              shuffle=False, #FIX IT
                              verbose=False)


    def _get_batches(self):
        batch = np.array(random.sample(self.memory, self.batch_size))
        state_batch = np.concatenate(batch[:, 0]) \
            .reshape(self.batch_size, self.state_size)
        action_batch = batch[:, 1]
        reward_batch = batch[:, 2]
        next_state_batch = np.concatenate(batch[:, 3]) \
            .reshape(self.batch_size, self.state_size)
        done_batch = batch[:, 4]
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch


    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        self.model.save(fn)


@click.command()
@click.option(
    '-s',
    '--symbol',
    default='al',
    show_default=True,
    help='given future code ',
)
@click.option(
    '-b',
    '--begin',
    default='2017-09-01',
    show_default=True,
    help='The begin date of the train.',
)

@click.option(
    '-e',
    '--end',
    default='2017-09-05',
    show_default=True,
    help='The end date of the train.',
)

@click.option(
    '-d',
    '--days',
    type=int,
    default=252,
    help='train days',
)

@click.option(
    '-t',
    '--train_round',
    type=int,
    default=100,
    help='train round',
)


@click.option(
     '--plot/--no-plot',
     #default=os.name != "nt",
     is_flag = True,
     default=False,
     help="render when training"
)

@click.option(
    '-m',
    '--model_path',
    default='.',
    show_default=True,
    help='trained model save path.',
)

def execute(symbol,begin,end,days,train_round,plot,model_path):
    print ("---------------------------------------1 ")
    env = gym.make('trading-v0').env
    env.initialise(symbol=symbol, start=begin, end=end)

    trials = train_round
    trial_len = env.src.get_count()
    dqn_agent = DQN(env=env)
    simrors = np.zeros(trials)
    mktrors = np.zeros(trials)
    victory = False
    i = 0
    for trial in range(trials):
        if victory == True:
            break;
        cur_state = env.reset().reshape(1,1,dqn_agent.state_size) #FIX IT
        for step in range(trial_len):
            action = dqn_agent.act(cur_state)
            new_state, reward, done, _ = env.step(action)
            new_state = new_state.reshape(1,1,dqn_agent.state_size)
            dqn_agent.remember(cur_state, action, reward, new_state, done)

            dqn_agent.replay()  # internally iterates default (prediction) model
            dqn_agent.target_train()  # iterates target model

            cur_state = new_state
            i += 1
            if trial >= PLOT_AFTER_ROUND:
            #####################################################################################
                if plot:
                    env.render()
            ####################################################################################
            if done:
                log.info("trail %d has done - total train step %d" ,trial,i)
                df = env.sim.to_df()
                simrors[trial] = df.bod_nav.values[-1] - 1  # compound returns
                mktrors[trial] = df.mkt_nav.values[-1] - 1
                log.info('total step:%6d, sim ret: %8.4f, mkt ret: %8.4f, net: %8.4f', i,
                        simrors[trial], mktrors[trial], simrors[trial] - mktrors[trial])
                if trial > 10:
                    vict = pd.DataFrame({'sim': simrors[trial - 10:trial],
                                         'mkt': mktrors[trial - 10:trial]})
                    vict['net'] = vict.sim - vict.mkt
                    log.info('vict:%f',vict.net.mean())

                    if vict.net.mean() > 10.2:
                        victory = True
                        log.info('Congratulations, Warren Buffet!  You won the trading game.')
                break
    import os
    log.info("Completed in %d trials , save it as %s",trial,os.path.join(model_path,dqn_agent.env.src.symbol + ".model"))
    dqn_agent.save_model(os.path.join(model_path,dqn_agent.env.src.symbol + ".model"))


if __name__ == "__main__":
    #import os
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    execute()
