# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 14:33:11 2022

@author: 38688
"""

from scipy.stats import beta
from pyDOE import lhs
import numpy as np
import pandas as pd

class case_2:
    def __init__(self):
        self.strike_price = 70
        self.option_fee = 0
        self.voll = 150
        #self.producer_num = 6
        #self.strategy_id = [0,1,2,3,4,5]
        self.producer_num = 3
        self.strategy_id = [0,1,2]
        #比发电商多1
        #self.risk_appetite = np.array([0.3,0,-0.3,0.1,0,-0.1,0.2])
        self.risk_appetite = np.array([0.3,0,-0.3,0.2])
        self.confidence_level = 0.05
        self.scene_num = 100
        self.CVar_calculate_location = int(self.scene_num*self.confidence_level)
        self.T_num = 8760
        self.bid_factor = 1
        #self.carbon_price_curve = np.array([0, 0])
        #self.GC_price_curve = np.array([0, 0])
        self.carbon_price_curve = np.array([4.59, 0.72e-06])
        self.GC_price_curve = np.array([28.68, -0.29e-06])
        self.carbon_price = 0
        self.GC_price = 0
        
        self.thermal_types = 1
        #b invest_cost(norm) pmax(norm) pmin(norm)
        self.thermal_parameters = np.array([[43.02,30000,0.1]])

        #beta

        self.wind_a = np.array([[20,10,30,20]])
        self.wind_b = np.array([[20,30,10,20]])
        #正态
        #self.wind_a = np.array([[1,1,1,1]])
        #self.wind_b = np.array([[0.1,0.1,0.1,0.1]])
        self.wind_types = 1
        self.wind_periods = np.array([2190,2190,2190,2190])
        self.wind_curves = np.ones((self.T_num,self.scene_num,self.wind_types))
        # cost invest_cost(norm)
        self.wind_parameters = np.array([[20,87000]])
        
        #self.init_capacity = np.array([[4000,400],[2000,200],[400,40],[3000,300],[2000,200],[600,60]]).astype(np.float64)
        #self.exist_capacity = np.array([[4000,400],[2000,200],[400,40],[3000,300],[2000,200],[600,60]]).astype(np.float64)
        self.init_capacity = np.array([[7000,700],[4000,400],[1000,100]]).astype(np.float64)
        self.exist_capacity = np.array([[7000,700],[4000,400],[1000,100]]).astype(np.float64)
        
        self.RO_sales = np.append(0*np.ones((self.producer_num,self.thermal_types)),
                                  0*np.ones((self.producer_num,self.wind_types)),1)
        
        self.demand = np.array(pd.read_excel('load_data.xlsx', sheet_name=0), 
                               dtype=float).reshape((-1,1))[:self.T_num]@np.ones((1,self.scene_num))
        
        self.state_dim_G = 4
        self.action_dim_G = self.wind_types
        self.max_gen = 8000*np.ones((self.producer_num,self.wind_types))
        self.min_gen = 0*np.ones((self.producer_num,self.wind_types))
        '''
        self.max_gen = np.append(1000*np.ones((self.producer_num,self.thermal_types)),
                                 8000*np.ones((self.producer_num,self.wind_types)),axis = 1)
        self.min_gen = np.append(0*np.ones((self.producer_num,self.thermal_types)),
                                 0*np.ones((self.producer_num,self.wind_types)),axis = 1)
        self.max_action = self.max_gen[self.strategy_id,:]-self.exist_capacity[self.strategy_id,:]
        self.min_action = -1.0*self.exist_capacity[self.strategy_id,:]
        self.max_action[:,self.thermal_types:self.thermal_types+self.wind_types] = 0.4
        self.min_action[:,self.thermal_types:self.thermal_types+self.wind_types] = 0.0
        '''
        
        self.state_dim2 = 4
        
        self.update_max_min_actions()
        self.make_cost_order()
        
        self.make_wind_curve()
        self.make_adjust()
        
        self.last_round_income = np.zeros(self.producer_num)
        
    def make_cost_order(self):
        cost = []
        bid = []
        invest = []
        for i in range(self.thermal_types):
            cost.append(self.thermal_parameters[i,0])
            bid.append(self.thermal_parameters[i,0]*self.bid_factor)
            invest.append(self.thermal_parameters[i,1])
        for i in range(self.wind_types):
            cost.append(self.wind_parameters[i,0])
            bid.append(self.wind_parameters[i,0]*self.bid_factor)
            invest.append(self.wind_parameters[i,1])
        self.cost_list = np.array(cost)
        self.bid_list = np.sort(np.array(bid))
        self.invest_list = np.array(invest)
        self.bid_gap_list = []
        for i in range(len(self.bid_list)):
            self.bid_gap_list.append(self.bid_list[i] - sum(self.bid_list[:i]))
        self.bid_order = np.argsort(np.array(bid)).astype(np.int32)
        
    def make_wind_curve(self):
        '''
        for w in range(self.wind_types):
            lhd = lhs(self.wind_periods.shape[0], samples=self.scene_num)
            T_id = 0
            for i in range(self.wind_periods.shape[0]):
                self.wind_curves[T_id:T_id+self.wind_periods[i],:,w] = \
                    np.ones((self.wind_periods[i],self.scene_num))* \
                        beta.ppf(lhd[:,i],self.wind_a[w,i],self.wind_b[w,i])
                T_id += self.wind_periods[i]

        '''
        for w in range(self.wind_types):
            T_id = 0
            for i in range(self.wind_periods.shape[0]):
                self.wind_curves[T_id:T_id+self.wind_periods[i],:,w] = \
                    np.ones((self.wind_periods[i],self.scene_num))* \
                        np.random.beta(self.wind_a[w,i],self.wind_b[w,i],self.scene_num)
                T_id += self.wind_periods[i]


        
    def make_adjust(self):
        self.thermal_each = self.exist_capacity[:,0:self.thermal_types]
        self.thermal_RO = self.thermal_each*self.RO_sales[:,0:self.thermal_types]
        self.wind_each = self.exist_capacity[:,self.thermal_types:self.thermal_types+self.wind_types]
        self.wind_RO = self.wind_each*self.RO_sales[:,self.thermal_types:self.thermal_types+self.wind_types]
        self.gen_total = np.append(np.ones((self.T_num,self.scene_num,self.thermal_types))
                                   *np.sum(self.thermal_each,axis=0),self.wind_curves*np.sum(self.wind_each,axis=0),axis=2)
        self.RO_total = np.append(self.thermal_RO,self.wind_RO,axis=1)
        self.gen_proportion = np.append(self.thermal_each,self.wind_each,axis=1)/ \
            (np.sum(np.append(self.thermal_each,self.wind_each,axis=1),axis=0)+0.01)
        
    def market_clear(self):
        
        price = np.zeros((self.T_num,self.scene_num))
        output = np.zeros((self.T_num,self.scene_num,self.thermal_types+self.wind_types))
        rest_demand = 1.0*self.demand
        for i in range(self.thermal_types+self.wind_types):
            d1 = 1.0*rest_demand
            price = price + (rest_demand>0).astype(np.int32)*self.bid_gap_list[i]
            rest_demand -= self.gen_total[:,:,self.bid_order[i]]
            d2 = 1.0*rest_demand
            output[:,:,self.bid_order[i]] = np.maximum(0,d1) - np.maximum(0,d2)
        price = price + (rest_demand>0).astype(np.int32)*(self.voll-self.bid_list[-1])
        
        return price, output

    
    def update_max_min_actions(self):
        self.max_action = self.max_gen[self.strategy_id,:]
        self.min_action = self.min_gen[self.strategy_id,:]
        
    def action_real(self, actions):
        #real_actions = (np.around(actions*2.999-0.5,0)/2)*(self.max_action-self.min_action)+self.min_action
        real_actions = actions*(self.max_action-self.min_action)+self.min_action
        return real_actions
    
    def step_forward(self,actions):
        actions = self.action_real(actions)
        self.exist_capacity[self.strategy_id,self.thermal_types:self.thermal_types+self.wind_types] = actions
        self.make_wind_curve()
        self.make_adjust()
        price, output = self.market_clear()
        energy_price_mean = np.mean(np.sum(np.sum(output,axis=2)*price,0)/np.sum(np.sum(output,axis=2),0))
        RO_D_auction = np.sum((price > (self.voll-0.1)).astype(np.int32),axis=0)*(self.voll-self.strike_price)
        CVar_D = np.mean(np.sort(RO_D_auction)[-self.CVar_calculate_location:])
        self.option_fee = (np.mean(RO_D_auction) + self.risk_appetite[-1]*CVar_D)/(1+self.risk_appetite[-1])
        carbon_amount = np.sum(output[:,:,0:self.thermal_types],axis=0)*self.thermal_parameters[:,2]
        GC_amount = np.sum(output[:,:,self.thermal_types:self.thermal_types+self.wind_types],axis=0)
        self.carbon_price = self.carbon_price_curve[0] + \
            self.carbon_price_curve[1]*np.sum(carbon_amount,axis=1)
        self.GC_price = self.GC_price_curve[0] + \
            self.GC_price_curve[1]*np.sum(GC_amount,axis=1)
        state = np.array([np.mean(price)/self.voll,self.option_fee/(self.T_num*(self.voll-self.strike_price))
                         ,np.mean(self.carbon_price)/15,np.mean(self.GC_price)/30])
        
        price_matrix = np.repeat(np.expand_dims(price, 2), self.thermal_types+self.wind_types, 2)
        
        reward_all_scences = (np.sum(output*price_matrix,0) + 
                              np.sum(-output,0)*self.cost_list + 
                              np.append(-self.carbon_price.reshape((self.scene_num,self.thermal_types))*carbon_amount,
                                        self.GC_price.reshape((self.scene_num,self.wind_types))*GC_amount,axis=1))@ \
            self.gen_proportion[self.strategy_id,:].T - \
                np.sum(np.maximum(0,(self.exist_capacity[self.strategy_id,:]-
                                     self.init_capacity[self.strategy_id,:]))*self.invest_list,1)
            
        
        for i in range(len(self.strategy_id)):
            reward_all_scences[:,i] += self.option_fee* \
                np.sum(self.RO_total[self.strategy_id[i],:])

            RO_loss = np.sum(np.sum(np.maximum(0,price_matrix-self.strike_price)*
                                    np.minimum(np.ones((self.T_num,self.scene_num,self.thermal_types+self.wind_types))*
                                               self.RO_total[self.strategy_id[i],:],
                                               output*self.gen_proportion[self.strategy_id[i],:]),axis=0),axis=1) + \
                np.sum(np.sum((price_matrix-self.strike_price>0).astype(np.int32)*
                              (self.voll-self.strike_price)*
                              np.maximum(np.ones((self.T_num,self.scene_num,self.thermal_types+self.wind_types))*
                                         self.RO_total[self.strategy_id[i],:]-
                                         output*self.gen_proportion[self.strategy_id[i],:],0),axis=0),axis=1)
            reward_all_scences[:,i] -= RO_loss
        CVar = np.mean(np.sort(reward_all_scences,axis=0)
                       [0:self.CVar_calculate_location,:],axis=0)
        reward = ((np.mean(reward_all_scences,axis=0) + self.risk_appetite[self.strategy_id]*CVar))/ \
            (1+self.risk_appetite[self.strategy_id])/(0.5*10**9)
        
        self.last_round_income = reward
        
        self.update_max_min_actions()
        
        state2 = (self.exist_capacity[self.strategy_id,self.thermal_types:self.thermal_types+self.wind_types]/
                  self.max_gen[self.strategy_id,:]).flatten()
        
        return reward, state, state, reward*0.5*10**9, self.option_fee, \
            np.mean(self.carbon_price), np.mean(self.GC_price), energy_price_mean

    def get_final_information(self):
        self.make_wind_curve()
        self.make_adjust()
        price, output = self.market_clear()
        
        RO_D_auction = np.sum((price > (self.voll-0.1)).astype(np.int32),axis=0)*(self.voll-self.strike_price)
        CVar_D = np.mean(np.sort(RO_D_auction)[-self.CVar_calculate_location:])
        self.option_fee = (np.mean(RO_D_auction) + self.risk_appetite[-1]*CVar_D)/((1+self.risk_appetite[-1]))
        carbon_amount = np.sum(output[:,:,0:self.thermal_types],axis=0)*self.thermal_parameters[:,2]
        GC_amount = np.sum(output[:,:,self.thermal_types:self.thermal_types+self.wind_types],axis=0)
        self.carbon_price = self.carbon_price_curve[0] + \
            self.carbon_price_curve[1]*np.sum(carbon_amount,axis=1)
        self.GC_price = self.GC_price_curve[0] + \
            self.GC_price_curve[1]*np.sum(GC_amount,axis=1)
        energy_price_mean = np.sum(np.sum(output,axis=2)*price,0)/np.sum(np.sum(output,axis=2),0)
        wind_percent = np.sum(output[:,:,1],0)/np.sum(np.sum(output,2),0)
        price_matrix = np.repeat(np.expand_dims(price, 2), self.thermal_types+self.wind_types, 2)
        income_RO = self.option_fee*np.sum(self.RO_total,axis=1)
        cost_RO_refund = np.zeros(self.scene_num)
        cost_RO_penalty = np.zeros(self.scene_num)
        
        
        for i in range(self.producer_num):
            cost_RO_refund += np.sum(np.sum(np.maximum(0,price_matrix-self.strike_price)*
                                            np.minimum(np.ones((self.T_num,self.scene_num,self.thermal_types+self.wind_types))*
                                                       self.RO_total[i,:],
                                                       output*self.gen_proportion[i,:]),axis=0),axis=1)
            cost_RO_penalty += np.sum(np.sum((price_matrix-self.strike_price>0).astype(np.int32)*
                                             (self.voll-self.strike_price)*
                                             np.maximum(np.ones((self.T_num,self.scene_num,self.thermal_types+self.wind_types))*
                                                        self.RO_total[i,:]-
                                                        output*self.gen_proportion[i,:],0),axis=0),axis=1)
        
        D_energy_cost = np.sum(np.sum(output,axis=2)*price,0)
        D_RO_cost = np.sum(income_RO)
        D_RO_income = cost_RO_refund + cost_RO_penalty
        D_RO = D_RO_cost - D_RO_income
        D = D_energy_cost + D_RO

        return self.option_fee, self.carbon_price, self.GC_price, energy_price_mean, \
            D_energy_cost, D_RO_cost, D_RO_income, D_RO, D, wind_percent