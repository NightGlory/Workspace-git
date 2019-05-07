import numpy as np


lst = [[1,3,5], [2,4,6]]
print(type(lst))                        # <class 'list'>
np_lst = np.array(lst)
print(type(np_lst))                     # <class 'numpy.ndarray'>
np_lst = np.array(lst, dtype=np.float)  
print(np_lst)
print(np_lst.shape)                     # 返回形状(2, 3)
print(np_lst.ndim)                      # 返回纬度2
print(np_lst.dtype)                     # 返回数据类型float64
print(np_lst.itemsize)                  # 返回数组中每个元素的字节单位长度8
print(np_lst.size)                      # 返回数组大小6


# 常用array
print(np.zeros([2,4]))                  # 返回全零数组
print(np.ones([3,5]))                   # 返回全1数组
print('Rand: ', np.random.rand(2,4))    # 返回2行4列0～1的浮点随机数
print(np.random.rand())                 # 默认参数1
print('RandInt: ', np.random.randint(1,10,3))       # 返回3个从1到10(1<=x<10)的随机整数
print('Randn: ', np.random.randn(2, 4))             # 返回2行4列的标准正态分布随机数
print('Choice: ', np.random.choice([10, 20, 30]))   # 返回给出的三个数中的任何随机的一个
print('Distribute: ', np.random.beta(1,10,5))      # 返回5个1到10的beta分布数


# numpy的操作
print(np.arange(1,11))                  # 返回1<=x<11的数组
print(np.arange(1,11).reshape([2,-1]))     
lst = np.arange(1,11).reshape([2,-1])
print(np.exp(lst))
print(np.exp2(lst))
print(np.sqrt(lst))
print(np.log(lst))
print(np.sin(lst))

print(np.sum(lst))                      # 求和
print(lst.sum())                        # 求和
print(lst.sum(axis=0))                  # 维度求和(列)
print(lst.sum(axis=1))                  # 维度求和(行)
print(lst.max(axis=0))                  # 维度求最大(列)

lst1 = np.array([1,2,3,4,5])
lst2 = np.array([6,7,8,9,0])
print('Add: ', lst1+lst2)
print('Sub: ', lst1-lst2)
print('Square: ', lst1**2)

print('DotMul: ', np.dot(lst1.T,lst2))  # 点乘
print('DotMul: ', np.dot(lst1,lst2))
print('Concatenate: ', np.concatenate((lst1, lst2), axis=0))    # 合并成一行
print(np.vstack((lst1, lst2)))                                  # 合并成两行
print(np.hstack((lst1, lst2)))                                  # 合并成一行

print(np.split(lst1, 5))                                        # 分成5份
print(np.copy(lst2))                                            # 拷贝

# 线性方程组
from numpy.linalg import *

print(np.eye(3))
lst = np.array([[1., 2.], [3., 4.]])
print('Inv: ',inv(lst))                 # 逆矩阵
print('T:', lst.transpose())            # 转置矩阵
print('T:', lst.T)                      # 转置矩阵
print('Det: ', det(lst))                # 求一个方阵的行列式
print('Eig: ', eig(lst))                # 求一个方阵的特征值+特征向量

y = np.array([[5.], [7.]])
print('Solve:', solve(lst, y))          # 求一个方程的解solve([x与y的系数], [方程等式右边的值])


# numpy的其他应用
print('FFT: ', np.fft.fft(np.array([1,1,1,1,1,1,1,1])))         # 信号处理中的快速傅立叶变换
print('Coef: ', np.corrcoef([1,0,1], [0,2,1]))                  # 皮尔逊积矩相关系数--相关系数运算
print('Poly: ', np.poly1d([2,1,3]))                             # 多项式求值2x**2+1x+3
print('Poly: ', np.poly1d([2,1,3], True))                       # 多项式求值(x-2)(x-1)(x-3)

 