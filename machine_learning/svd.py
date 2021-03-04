import numpy as np

# 本着做一遍才算理解的学习思路，用原生的np.linalg.eig实现了一个简单的svd函数
# 做的过程中也碰到了一些问题，如下做法是错的:
#     sigma, V = np.linalg.eig(A.T @ A)
#     sigma, U = np.linalg.eig(A @ A.T) // 错误
# 因为特征值正负号是随机的，所以U和V的特征向量正负号是有某种关联的，
# 所以正确的做法是通过U推导VT或者通过VT推导U: A = U @ sigma @ VT

def svd(A):
    m, n = A.shape
    if m > n:
        sigma, V = np.linalg.eig(A.T @ A)
        # 将sigma 和V 按照特征值从大到小排列
        arg_sort = np.argsort(sigma)[::-1]
        sigma = np.sort(sigma)[::-1]
        V = V[:, arg_sort]

        # 对sigma进行平方根处理
        sigma_matrix = np.diag(np.sqrt(sigma))

        sigma_inv = np.linalg.inv(sigma_matrix)

        U = A @ V.T @ sigma_inv
        U = np.pad(U, pad_width=((0, 0), (0, m - n)))
        sigma_matrix = np.pad(sigma_matrix, pad_width=((0, m - n), (0, 0)))
        return (U, sigma_matrix, V)
    else:
        # 同m>n 只不过换成从U开始计算
        sigma, U = np.linalg.eig(A @ A.T)
        arg_sort = np.argsort(sigma)[::-1]
        sigma = np.sort(sigma)[::-1]
        U = U[:, arg_sort]

        sigma_matrix = np.diag(np.sqrt(sigma))
        sigma_inv = np.linalg.inv(sigma_matrix)
        V = sigma_inv @ U.T @ A
        V = np.pad(V, pad_width=((0, n - m), (0, 0)))

        sigma_matrix = np.pad(sigma_matrix, pad_width=((0, 0), (0, n - m)))
        return (U, sigma_matrix, V)


if __name__ == "__main__":
    a = np.array([[12, -5, 7, 62333, 255], [-2, 1, -10, 4, 4], [3, 6, 7, 5, -2]])
    U, sigma, V = svd(a)
    print(U @ sigma @ V)
