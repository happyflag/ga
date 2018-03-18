#!/usr/bin/python
#-*-coding:utf-8 -*-

import random
import math
from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt


CXPB, MUTPB, NGEN, popsize = 0.8, 0.3, 50, 100  # control parameters

up = [64, 64, 64, 64, 64, 64, 64, 64, 64, 64]  # upper range for variables
low = [-64, -64, -64, -64, -64, -64, -64, -64, -64, -64]  # lower range for variables
parameter = [CXPB, MUTPB, NGEN, popsize, low, up]
maxTime = 500


origin_list = [0,1,2,3,4,5,6,7,8,9,10,11,12]
origin_list_limit = [0,1,2,3,4,5,6,7,8,9]
origin_dsm = np.array([[20, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0], [0, 27, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [1, 1, 35, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                          [0, 1, 1, 34, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 1, 0, 45, 1, 1, 0, 0, 0, 0, 0, 0],
                          [1, 0, 1, 0, 1, 27, 1, 0, 0, 0, 0, 0, 0],
                          [1, 0, 1, 0, 1, 1, 30, 0, 0, 0, 0, 0, 0], [0, 1, 0, 1, 0, 0, 0, 40, 1, 1, 0, 1, 1],
                          [0, 0, 1, 0, 0, 0, 0, 0, 21, 0, 0, 1, 0],
                          [0, 0, 1, 0, 0, 0, 0, 0, 0, 35, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 26, 0, 1],
                          [0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 36, 1],
                          [0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 30]])
origin_rp = np.array([[20,0,0.1,0,0.2,0.2,0,0,0,0,0,0,0],
                      [0,27,0.1,0.2,0,0,0,0,0,0,0,0,0],
                      [0.4,0.2,35,0.2,0,0,0,0,0.1,0.2,0,0,0],
                      [0,0.3,0.1,34,0,0,0,0,0,0,0,0,0],
                      [0.2,0,0.1,0,45,0,0,0,0,0,0,0,0],
                      [0.2,0,0.2,0,0.1,27,0,0,0,0,0,0,0],
                      [0.2,0,0.2,0,0.2,0.1,30,0,0,0,0,0,0],
                      [0,0.2,0,0.1,0,0,0,40,0.2,0.2,0,0.8,0.2],
                      [0,0,0.2,0,0,0,0,0,21,0,0,0.3,0],
                      [0,0,0.1,0,0,0,0,0,0,35,0,0,0],
                      [0,0,0,0,0,0,0,0,0.2,0.3,26,0,0.1],
                      [0,0.2,0,0.1,0,0,0,0.1,0.4,0.2,0,36,0.2],
                      [0,0.3,0,0.2,0,0,0,0,0.2,0.1,0,0.3,30]])
origin_ri = np.array([[20,0,0.1,0,0.5,0.4,0,0,0,0,0,0,0],
                      [0,27,0.1,0.6,0,0,0,0,0,0,0.5,0.4,0],
                      [0.5,0.2,35,0.6,0,0,0,0,0.5,0.2,0,0,0],
                      [0,0.3,0.1,34,0,0,0,0,0,0,0,0,0],
                      [0.6,0,0.1,0,45,0,0,0,0,0,0,0,0],
                      [0.5,0,0.2,0,0.4,27,0,0,0,0,0,0,0],
                      [0.5,0,0.2,0,0.5,0.5,30,0,0,0,0,0,0],
                      [0,0.2,0,0.4,0,0,0,40,0.6,0.2,0,0.5,0.6],
                      [0,0,0.2,0,0,0,0,0,21,0,0,0.4,0],
                      [0,0,0.1,0,0,0,0,0,0,35,0,0,0],
                      [0,0,0,0,0,0,0,0,0.6,0.3,26,0,0.6],
                      [0,0.2,0,0.6,0,0,0,0.5,0.4,0.4,0,36,0.5],
                      [0,0.3,0,0.5,0,0,0,0,0.5,0.1,0,0.5,30]])


def apend(origin_limit):
    geneinforesult = []
    for gene in origin_limit:
        if gene < 4:
            geneinforesult.append(gene)
        elif gene == 4:
            geneinforesult.append(4)
            geneinforesult.append(5)
            geneinforesult.append(6)
        elif gene > 4 and gene < 9:
            geneinforesult.append(gene + 2)
        else:
            geneinforesult.append(11)
            geneinforesult.append(12)
    return geneinforesult

def shrinkle(geneinfo_param):
    shrinkleGeng = []
    for gene in geneinfo_param:
        if gene < 4:
            shrinkleGeng.append(gene)
        elif gene == 4:
            shrinkleGeng.append(4)
        elif gene == 5:
            continue
        elif gene == 6:
            continue
        elif gene > 6 and gene < 11:
            shrinkleGeng.append(gene - 2)
        elif gene == 11:
            shrinkleGeng.append(9)
        else:
            continue
    return shrinkleGeng

# 产生初始种群
def generateInitalPop():
    pop = []
    for i in range(parameter[3]):
        geneinfo = [0,1,2,3,4,5,6,7,8,9]
        randpos1 = random.sample(origin_list_limit, 4)
        tmp = geneinfo[randpos1[0]]
        geneinfo[randpos1[0]] = geneinfo[randpos1[1]]
        geneinfo[randpos1[1]] = tmp
        tmp = geneinfo[randpos1[2]]
        geneinfo[randpos1[2]] = geneinfo[randpos1[3]]
        geneinfo[randpos1[3]] = tmp

        pop.append(apend(geneinfo))
    return pop


# 计算适应度函数
def _cal_fitness(geneinfo, w1, w_nc):
    NC = 0
    NC_first = 0
    NC_second = 0
    RCT = 0
    # caculate NC
    for i in origin_list:
        internal_NC_first = 0
        internal_NC_second = 0
        for j in range(i+1, 12):
            internal_NC_first = internal_NC_first + origin_dsm[geneinfo[i], geneinfo[j]]
        for j in range(i+1,12):
            internal_NC_second = internal_NC_second + origin_dsm[geneinfo[i], geneinfo[j]] * (j - i)
        NC_first = NC_first + internal_NC_first
        NC_second = NC_second + internal_NC_second
    NC = w1 * NC_first + (1-w1) * NC_second

    # caculate RCT

    for i in origin_list:
        internal_RCT_first = 0
        internal_RCT_sencond = 0
        for j in range(i+1, 12):
            internal_RCT_first = internal_RCT_first + (origin_rp[geneinfo[i], geneinfo[j]] * origin_ri[geneinfo[i],geneinfo[j]])
        for k in range(0,12):
            internal_RCT_sencond = internal_RCT_sencond + (origin_dsm[geneinfo[k], geneinfo[k]] * origin_rp[geneinfo[k], geneinfo[i]] * origin_ri[geneinfo[k], geneinfo[i]])
        RCT = RCT + internal_RCT_first * internal_RCT_sencond

    return w_nc * NC + (1-w_nc) * RCT, NC, RCT

def _variation_(pop):
    # 变异
    result_pop = []
    for gene in pop:
        result_pop.append(gene)
    for i in range(25):
        one_pop = pop[i]
        rand_pos = random.sample(origin_list_limit, 2)
        tmp = one_pop[rand_pos[0]]
        one_pop[rand_pos[0]] = one_pop[rand_pos[1]]
        one_pop[rand_pos[1]] = tmp
        result_pop.append(one_pop)
    return result_pop

# 交叉
def _cross_(fitpop):
    result_pop = []
    for i in range(25):
        if i < 24:
            one_pop = fitpop[i]
            second_pop = fitpop[i + 1]
        else:
            one_pop = fitpop[i]
            second_pop = fitpop[0]

        rm_tail = one_pop[5:9]
        rp_tail = []
        for pop in second_pop:
            if pop in rm_tail:
                rp_tail.append(pop)
        one_pop[5] = rp_tail[0]
        one_pop[6] = rp_tail[1]
        one_pop[7] = rp_tail[2]
        one_pop[8] = rp_tail[3]
        result_pop.append(one_pop)

    return result_pop

# 产生子代种群
def _generateNextPop(fitPop):
    shringle_pop = []
    for pop in fitPop:
        shringle_pop.append(shrinkle(pop))

    # 交叉 变异
    tmp_pop = _cross_(shringle_pop)
    result_pop = _variation_(tmp_pop)
    result = []
    for pop in result_pop:
        result.append(apend(pop))
    return result

def _find_min_(fitness, m):
    temp = []
    for i in range(m):
        idx = fitness.index(min(fitness))
        temp.append(idx)
        fitness[idx] = max(fitness)
    return temp



# 计算中间计算过程数据，便于画图

if __name__ == '__main__':

    minFit = 100
    minPop = origin_list

    population = generateInitalPop()
    ori_fitness = []
    param_fit = []
    for geneinfo in population:
        single_fitness, single_NC, single_RCT = _cal_fitness(geneinfo, 0.5, 0.999)
        ori_fitness.append(single_fitness)
        param_fit.append(single_fitness)

    fit_index = _find_min_(param_fit, 25)
    global_best_fitness = []
    global_best_pop = []
    for idx in fit_index:
        global_best_fitness.append(ori_fitness[idx])
        global_best_pop.append(population[idx])

    plot_NC = []
    plot_RCT = []
    plot_fitness = []
    plot_best_fit = []
    plt_global_best_pop =[]
    for k in range(maxTime):
        population = _generateNextPop(global_best_pop)

        fitness = []
        NC = []
        RCT = []
        parm_fit2 = []
        parm_NC = []
        parm_RCT = []
        i=0
        for geneinfo in population:
            single_fitness, single_NC, single_RCT = _cal_fitness(geneinfo, 0.5, 0.999)
            if single_fitness < minFit:
                minFit = single_fitness
                minPop = geneinfo
            # fitness[i] = single_fitness
            # parm_fit2[i] = single_fitness
            # NC[i] = single_NC
            # parm_NC[i] = single_NC
            # RCT[i] = single_RCT
            # parm_RCT[i] = single_RCT

            fitness.append(single_fitness)
            parm_fit2.append(single_fitness)
            NC.append(single_NC)
            parm_NC.append(single_NC)
            parm_RCT.append(single_RCT)
            RCT.append(single_RCT)

        k_fit_index = _find_min_(parm_fit2, 25)
        k_fit_NC = _find_min_(parm_NC, 25)
        k_fit_RCT = _find_min_(parm_NC, 25)

        plot_fitness.append(min(fitness))
        plot_NC.append(min(NC))
        plot_RCT.append(min(RCT))

        for m in range(25):
            if fitness[k_fit_index[m]] < np.max(global_best_fitness):
                mx_idx = global_best_fitness.index(max(global_best_fitness))
                global_best_fitness[mx_idx] = fitness[k_fit_index[m]]
                global_best_pop[mx_idx] = population[k_fit_index[m]]
                # print population[k_fit_index[m]]
        plot_best_fit.append(min(global_best_fitness))
        plt_global_best_pop.append(global_best_pop[global_best_fitness.index(min(global_best_fitness))])


    # plt.figure(1)
    # plt.plot(np.arange(0,maxTime,1), plot_NC)
    # plt.title("best nc")
    # plt.show()
    # plt.figure(2)
    # plt.plot(np.arange(0,25,1),global_best_fitness)
    # plt.title("global final fitness")
    # plt.show()
    # plt.figure(3)
    # plt.plot(np.arange(0,maxTime,1),plot_RCT)
    # plt.title("best rct")
    # plt.show()
    # plt.figure(4)
    # plt.plot(np.arange(0,maxTime,1),plot_fitness)
    # plt.title(" fitness")
    # plt.show()
    plt.figure(5)
    plt.plot(np.arange(0,maxTime,1),plot_best_fit)
    plt.title("best fitness")
    plt.show()

    idx_1530 = global_best_fitness.index(min(global_best_fitness))
    pop_1530 = global_best_pop[idx_1530]
    print minFit
    print minPop
    print global_best_fitness
    print global_best_pop
    print plot_RCT
    print plot_fitness
    print plot_NC
    print plt_global_best_pop


