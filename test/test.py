import tensorflow as tf

def f(*a):
    print(len(a))
    for e in a:
        print(e)

if __name__ == '__main__':
    f(1, 2, 3)