# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 23:16:37 2022

@author: 38688
"""

import numpy as np
from MATD3_models import MATD3
from MADDPG_models import MADDPG
from MAPPO_models import MAPPO
from make_case import case_2
#import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':
    case = case_2()

    agents = MATD3(case)
    #agents = MADDPG(case)
    #agents = MAPPO(case)

    noise_std = 0.4 # the std of Gaussian noise for exploration
    noise_std_min = 0.0005
    max_train_steps = 20000  # Maximum number of training steps
    no_noise_steps = 20000
    lr_delay_steps = 20000
    random_steps = 2000  # Take the random actions in the beginning for the better exploration
    total_steps = 0  # Record the total steps during the training
    begin_learn_num = 128
    lr_frequency = 8
    
    # mean action final
    a_G = agents.choose_action_random_G()
    #if MAPPO, use this:
    #a_G, a_logprob = agents.choose_action_random_G(np.random.rand(case.state_dim_G))

    r, s, s2, income, p1, p2, p3, p4 = case.step_forward(a_G)
    
    capacity_record = np.empty(shape=(case.producer_num,case.thermal_types+case.wind_types,0))
    income_record = np.empty(shape=(case.producer_num,0))
    option_fee_record = np.empty(shape=(0))
    carbon_price_record = np.empty(shape=(0))
    GC_price_record = np.empty(shape=(0))
    E_price_record = np.empty(shape=(0))
    capacity_record = np.append(capacity_record,
                                case.exist_capacity.reshape((case.producer_num,case.thermal_types+case.wind_types,1)),axis = 2)
    income_record = np.append(income_record,income.reshape((case.producer_num,1)),axis = 1)
    option_fee_record = np.append(option_fee_record, p1)
    carbon_price_record = np.append(carbon_price_record, p2)
    GC_price_record = np.append(GC_price_record, p3)
    E_price_record = np.append(E_price_record, p4)
    
    while total_steps < max_train_steps:

        if total_steps <= random_steps:  # Take random actions in the beginning for the better exploration
            a_G = agents.choose_action_random_G()
        else:
            if total_steps <= no_noise_steps:
                # Add Gaussian noise to action for exploration
                a_G = agents.choose_action_with_noise_G(s,noise_std)
            else:
                a_G = agents.choose_action_G(s)
        r, s_, s2_, income, p1, p2, p3, p4 = case.step_forward(a_G)

        # if MAPPO, use this:
        '''
        if total_steps <= random_steps:  # Take random actions in the beginning for the better exploration
            a_G, a_logprob = agents.choose_action_random_G(s)
        else:
            if total_steps <= no_noise_steps:
                # Add Gaussian noise to action for exploration
                a_G, a_logprob = agents.choose_action_with_noise_G(s, noise_std)
            else:
                a_G, a_logprob = agents.choose_action_G(s)
        r, s_, s2_, income, p1, p2, p3, p4 = case.step_forward(a_G)
        '''
        capacity_record = np.append(capacity_record,
                                    case.exist_capacity.reshape((case.producer_num,
                                                                 case.thermal_types+case.wind_types,1)),axis = 2)
        income_record = np.append(income_record,income.reshape((case.producer_num,1)),axis = 1)
        option_fee_record = np.append(option_fee_record, p1)
        carbon_price_record = np.append(carbon_price_record, p2)
        GC_price_record = np.append(GC_price_record, p3)
        E_price_record = np.append(E_price_record, p4)
        
        agents.replay_buffer.store(s, a_G.flatten(), r, s_, s2, s2_)
        # if MAPPO, use this:
        #agents.replay_buffer.store(s, a_G.flatten(), r, s_, s2, s2_, a_logprob.flatten())
        s = s_
        s2 = s2_
        total_steps += 1
        
        if  total_steps % lr_frequency == 0:
            print('total_steps:',total_steps)
            agents.learn()
            print('wind',case.exist_capacity[:,1])
            if total_steps >= random_steps:
                noise_std = np.max([noise_std*0.997,noise_std_min])
    
    
    record_all = capacity_record[:,0,:].T
    record_all = np.append(record_all, capacity_record[:,1,:].T,axis = 1)
    record_all = np.append(record_all, income_record.T,axis = 1)
    record_all = np.append(record_all, option_fee_record.reshape((-1,1)),axis = 1)
    record_all = np.append(record_all, carbon_price_record.reshape((-1,1)),axis = 1)
    record_all = np.append(record_all, GC_price_record.reshape((-1,1)),axis = 1)
    record_all = np.append(record_all, E_price_record.reshape((-1,1)),axis = 1)
    data = pd.DataFrame(record_all)
    writer = pd.ExcelWriter('record.xlsx')
    data.to_excel(writer, 'page_1')
    writer.save()
    writer.close()