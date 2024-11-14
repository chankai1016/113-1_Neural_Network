import numpy as np

def backpropagation(input, target, bias, weight):
    print("\n#### 1. Forward Pass")
    print("\n##### A. Hidden Unit")
    print("\n| i |  net(i)  |   a(i)   |")
    print("| :--: | ---: | ---: |")
    net = [0] * 10
    a = [0] * 10
    for i in range(len(input)):
        a[i] = input[i]
    for i in range(2, 5 + 1):  # 2, 3, 4, 5 : hidden num
        for j in range(0, 1 + 1):  # 0, 1 : input num
            # net_i = a_0*a_0i + a_1*a_1i + b_i
            net[i] += input[j] * weight[j][i]
        net[i] += bias[i]
        a[i] = (1 + np.exp(-1 * net[i])) ** -1
        print("| **{}** | {:.6f} | {:.6f} |".format(i, net[i], a[i]))
    print("\n##### B. Output Unit")
    print("\n| i |  net(i)  |   a(i)   |")
    print("| :--: | ---: | ---: |")
    for i in range(6, 9 + 1):  # 6, 7, 8, 9 : output num
        for j in range(2, 5 + 1):  # 2, 3, 4, 5 : hidden num
            # net_i = a_2*a_2i + a_3*a_3i + a_4*a_4i + a_5*a_5i + b_i
            net[i] += a[j] * weight[j][i]
        net[i] += bias[i]
        a[i] = (1 + np.exp(-1 * net[i])) ** -1
        print("| **{}** | {:.6f} | {:.6f} |".format(i, net[i], a[i]))

    print("\n#### 2. Backward Pass")
    delta = [0] * 10
    print("\n##### A. Output Unit")
    print("\n| i |   δ(i)   |")
    print("| :--: | ---: |")
    for i in range(6, 9 + 1):  # 6, 7, 8, 9 : output num
        delta[i] = (target[i - 6] - a[i]) * ( a[i] * (1 - a[i] ))  # 1-6 = 0, 1, 2, 3 : target num
        print("| **{}** | {:.6f} |".format(i, delta[i]))
    print("\n##### B. Hidden Unit")
    print("\n| i |   δ(i)   |")
    print("| :--: | ---: |")
    for i in range(2, 5 + 1):  # 2, 3, 4, 5 : hidden num
        for j in range(6, 9 + 1):  # 6, 7, 8, 9 : output num
            delta[i] += delta[j] * weight[i][j]
        delta[i] *= a[i] * (1 - a[i])
        print("| **{}** | {:.6f} |".format(i, delta[i]))

    print("\n#### 3. Change of Weights and Biases")
    print("\n##### A. Weights")
    print("\n| i - j |   ΔW(i)   |  W^new(i) | i - j |   ΔW(i)   |  W^new(i) | i - j |   ΔW(i)   |  W^new(i) | i - j |   ΔW(i)   |  W^new(i) |")
    print("| :--: | ---: | ---: | :--: | ---: | ---: | :--: | ---: | ---: | :--: | ---: | ---: |")
    d_weight = [[0] * 10] * 6
    weight_new = [[0] * 10] * 6
    for i in range(0, 1+1): # 0, 1 : input num
        for j in range(2, 5+1): # 2, 3, 4, 5 : hidden num
            d_weight[i][j] = 0.20 * delta[j] * a[i]
            weight_new[i][j] = weight[i][j] + d_weight[i][j]
            print("| **{} - {}** | {:.6f} | {:.6f} ".format(i, j, d_weight[i][j], weight_new[i][j]), end = "")
        print("|")
    for i in range(2, 5+1): # 2, 3, 4, 5 : hidden num
        for j in range(6, 9+1): # 6, 7, 8, 9 : output num
            d_weight[i][j] = 0.20 * delta[j] * a[i]
            weight_new[i][j] = weight[i][j] + d_weight[i][j]
            print("| **{} - {}** | {:.6f} | {:.6f} ".format(i, j, d_weight[i][j], weight_new[i][j]), end = "")
        print("|")
    print("\n##### B. Biases")
    print("\n| i |   Δb(i)   |  b^new(i) | i |   Δb(i)   |  b^new(i) |")
    print("| :--: | ---: | ---: | :--: | ---: | ---: |")
    d_bias = [0] * 10
    bias_new = [0] * 10
    for i in range(2, 9+1): # 2 , ..., 9 : biases num
        d_bias[i] = 0.20 * delta[i]
        bias_new[i] = bias[i] + d_bias[i]
    # for output table
    for i in range(2, 5+1):
        print("| **{}** | {:.6f} | {:.6f} | **{}** | {:.6f} | {:.6f} |".format(i, d_bias[i], bias_new[i], i+4, d_bias[i+4], bias_new[i+4]))
    return 0