def heb_net(x, target, alpha, init):
    i = 0
    w = list()

    # first time
    print("#{}".format(1), end=", ")
    print("P_1 = {}, w = {}".format(x[0], init), end=", ")
    net = wx(init, x[0])  # net value
    print("net = {}".format(net), end=", ")
    print("f(net) = {}, target = {}".format(f_net(net), target[0]))
    if f_net(net) == target[0]:  # y = t ?
        w.append(list(init))
        print("Result: do nothing.   {}".format(w[0]))
    else:
        w_new_list = list(w_new(init, alpha, target[0], f_net(net), x[0]))
        w.append(list(w_new_list))
        print("Result: weight change {}".format(w[0]))

    i += 1

    # second time -
    while True:
        print("#{}".format(i + 1), end=", ")
        print("P_{} = {}, w = {}".format((i % len(x)) + 1,x[i % len(x)], w[i - 1]), end=", ")
        net = wx(w[i - 1], x[i % len(x)])  # net value
        print("net = {}".format(net), end=", ")
        print("f(net) = {}, target = {}".format(f_net(net), target[i % len(x)]))
        if f_net(net) == target[i % len(x)]:  # y = t ?
            w.append(list(w[i - 1]))
            print("Result: do nothing.   {}".format(w[i]))
        else:
            w_new_list = list(
                w_new(w[i - 1], alpha, target[i % len(x)], f_net(net), x[i % len(x)])
            )
            w.append(w_new_list)
            print("Result: weight change {}".format(w[i]))
        if i >= len(x):
            check = 0
            for j in range(len(x)):
                if w[i] == w[i - j]:
                    check += 1
            if check >= len(x):
                print("##")
                print("倒數{}次計算結果等值，即weight = {}".format(len(x),w[i]))
                # for j in range(check):
                #     if j == check - 1:
                #         print()
                #     else:
                #         print(w[i - j], end=", ")
                print()
                break
            if i >= 100 - 1:
                print("## 執行100次")
                break
        i += 1

def w_new(w_old, alpha, t, y, x):
    # w_new = w_old + alpha * (t - y) * x
    x_cal = [alpha * (t - y) * 1]
    result = []
    for i in x:
        x_cal.append(alpha * (t - y) * i)
    for i in range(len(x_cal)):
        result.append(w_old[i] + x_cal[i])
    return result


def f_net(wx):
    if wx > 0:
        return 1
    else:
        return 0


def wx(w, x):
    result = w[0]
    for i in range(len(x)):
        result += w[i + 1] * x[i]
    return result


if __name__ == "__main__":
    x = [[1, 2], [1, 1], [-1, 0], [3, 3], [3, 2], [4, 3]]
    target = [1, 1, 1, 0, 0, 0]
    alpha = 1
    init = [1, -1, -1]  # initial value
    heb_net(x, target, alpha, init)
