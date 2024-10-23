from main import heb_net

def main():
    x = [[1, 2], [1, 1], [-1, 0], [3, 3], [3, 2], [4, 3]]
    target = [1, 1, 1, 0, 0, 0]
    alpha = 1
    init = [1, -1, -1]  # initial value
    heb_net(x, target, alpha, init)


if __name__ == "__main__":
    main()
