import random
import math


def lcm(x, y):
    return x * y // math.gcd(x, y)

def _egcd(a, b):
    """扩展欧几里得算法：返回 (g, x, y)，满足 a*x + b*y = g。"""
    if b == 0:
        return a, 1, 0
    g, x1, y1 = _egcd(b, a % b)
    x = y1
    y = x1 - (a // b) * y1
    return g, x, y

def _modinv(a, m):
    """模逆（兼容 Python 3.7），返回 a 在模 m 下的乘法逆元。"""
    a = a % m
    g, x, _ = _egcd(a, m)
    if g != 1:
        raise ValueError("modular inverse does not exist")
    return x % m


class PublicKey:

    def __init__(self, n, g):
        self.n = n
        self.g = g
        self.n_square = n * n


class PrivateKey:

    def __init__(self, lam, mu):
        self.lam = lam
        self.mu = mu


class Paillier:

    def keygen(self, p=17, q=19):

        n = p * q
        g = n + 1

        lam = lcm(p - 1, q - 1)

        n_square = n * n

        l_val = self.L(pow(g, lam, n_square), n)
        mu = _modinv(l_val, n)

        pub = PublicKey(n, g)
        priv = PrivateKey(lam, mu)

        return pub, priv


    def L(self, x, n):
        return (x - 1) // n


    def encrypt(self, pub, m):
        # Paillier 要求 r 属于 Z*_n（与 n 互素），否则会导致解密结果异常
        r = random.randint(1, pub.n - 1)
        while math.gcd(r, pub.n) != 1:
            r = random.randint(1, pub.n - 1)

        c = (pow(pub.g, m, pub.n_square) *
             pow(r, pub.n, pub.n_square)) % pub.n_square

        return c


    def decrypt(self, pub, priv, c):

        x = pow(c, priv.lam, pub.n_square)

        l = self.L(x, pub.n)

        m = (l * priv.mu) % pub.n

        return m
