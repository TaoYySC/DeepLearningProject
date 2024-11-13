# from pulp import LpMinimize, LpProblem, LpVariable
#
# # 区间消防车行驶最长时间（表1数据）
# travel_time = [
#     [4, 10, 16, 28, 27, 20],
#     [10, 5, 24, 32, 17, 10],
#     [16, 24, 4, 12, 27, 21],
#     [28, 32, 12, 5, 15, 25],
#     [27, 17, 27, 15, 3, 14],
#     [20, 10, 21, 25, 14, 6]
# ]
#
# # 各区“同时发生”的火警次数Xi的概率分布（表2数据）
# probability_distribution = [
#     [0.9, 0.8, 0.85, 0.79, 0.76, 0.81],
#     [0.05, 0.1, 0.06, 0.06, 0.09, 0.05],
#     [0.03, 0.05, 0.05, 0.09, 0.08, 0.04],
#     [0.01, 0.03, 0.02, 0.04, 0.03, 0.06],
#     [0.01, 0.02, 0.02, 0.02, 0.04, 0.04]
# ]
#
# # 各区发生火警占全市的比例（表3数据）
# fire_alarm_proportion = [0.18, 0.19, 0.14, 0.21, 0.16, 0.12]
#
# # 创建整数规划问题实例，目标是最小化成本
# prob = LpProblem("FireStation_Problem", LpMinimize)
#
# # 决策变量：x表示是否建消防站（0-1变量），y表示每个区配备的消防车数量（整数变量）
# x = [LpVariable(f"x_{i}", cat='Binary') for i in range(6)]
# y = [LpVariable(f"y_{i}", lowBound=0, cat='Integer') for i in range(6)]
#
# # 目标函数：最小化建站成本和消防车购买成本
# prob += 200 * sum(x[i] for i in range(6)) + 30 * sum(y[i] for i in range(6))
#
# # 约束条件1：消防车能在15分钟内赶到现场
# for i in range(6):
#     for j in range(6):
#         if travel_time[i][j] > 15:
#             continue
#         prob += x[i] >= 1 - x[j]  # 如果i区能在15分钟内到j区，那么若j区不建消防站，i区必须建
#
# # 约束条件2：根据预期火警次数确定每个区至少配备的消防车数量下限
# for i in range(6):
#     expected_fire_count = 0
#     for k in range(5):
#         expected_fire_count += (k + 1) * (probability_distribution[k][i] * fire_alarm_proportion[i])
#     prob += y[i] >= 2 if expected_fire_count >= 1 else 0  # 如果预期火警次数大于等于1，则至少配备2辆消防车
#
# # 解决整数规划问题
# prob.solve()
#
# # 输出结果
# print('建站方案（1表示建，0表示不建）：')
# for i in range(6):
#     print(f"区 {i + 1}: {int(x[i].value())}")
#
# print('各站消防车配备数量：')
# for i in range(6):
#     if y[i] is not None:
#         print(f"区 {i + 1}: {int(y[i].value())}")
#     else:
#         print(f"区 {i + 1}: 无数据")
#
# print(f'总成本: {prob.objective.value()} 万元')

import pulp
import numpy as np

# 定义区域和距离矩阵
areas = [1, 2, 3, 4, 5, 6]
distance_matrix = np.array([
    [0, 12, 28, 16, 15, 20],
    [12, 0, 25, 15, 10, 18],
    [28, 25, 0, 14, 30, 26],
    [16, 15, 14, 0, 15, 20],
    [15, 10, 30, 15, 0, 14],
    [20, 18, 26, 20, 14, 0]
])

# 设定时间限制（分钟）
time_limit = 15

# 定义火灾发生的概率分布（表2）
fire_prob = [0.18, 0.19, 0.13, 0.17, 0.21, 0.12]

# 定义问题
prob = pulp.LpProblem("FireStationOptimization", pulp.LpMinimize)

# 定义决策变量
y = pulp.LpVariable.dicts("FireStation", areas, cat='Binary')  # 是否建立消防站
n = pulp.LpVariable.dicts("FireTrucks", areas, lowBound=0, cat='Integer')  # 消防车数量

# 目标函数：最小化消防站建设成本和消防车购置成本
prob += pulp.lpSum([200 * y[i] for i in areas]) + pulp.lpSum([30 * n[i] for i in areas]), "Total Cost"

# 约束1：确保每个区域在15分钟内至少有一个消防站可以派车到达
for j in areas:
    prob += pulp.lpSum([y[i] for i in areas if distance_matrix[i-1][j-1] <= time_limit]) >= 1, f"Coverage_Constraint_Area_{j}"

# 约束2：每次火警至少派出两辆车
for i in areas:
    prob += n[i] >= 2 * y[i], f"Min_Trucks_Constraint_Area_{i}"

# 约束3：确保消防车数量能够满足多起火警的情况
for i in areas:
    prob += n[i] >= fire_prob[i-1] * 2, f"Fire_Probability_Constraint_Area_{i}"

# 求解问题
prob.solve()

# 输出结果
print("Status:", pulp.LpStatus[prob.status])
for i in areas:
    print(f"Area {i}: Build Fire Station =", pulp.value(y[i]), ", Number of Fire Trucks =", pulp.value(n[i]))

# 输出总成本
print("Total Cost:", pulp.value(prob.objective))

