import numpy as np

# print(np.__version__)
#
# a = np.array([1, 2, 3])
# b = np.array([2, 3, 4])
# c = np.stack((a, b),axis=0) #默认axis=0
# print(c)
# print(c.shape)

# gaussin_blur_3x3 = np.divide([
#     [1., 2., 1.],
#     [2., 4., 2.],
#     [1., 2., 1.],
# ], 16.) # (3, 3)  除法操作
# gaussin_blur_3x3 = np.stack((gaussin_blur_3x3, gaussin_blur_3x3), axis=-1) # (3, 3, 2)
# print(gaussin_blur_3x3)
#
# print('///////////////////////////////////////////////////////////////////////////////')
# gaussin_blur_3x3 = np.stack((gaussin_blur_3x3, gaussin_blur_3x3), axis=-1) # (3, 3, 2, 2)
# print(gaussin_blur_3x3)
#
# arrays = [np.random.randn(3, 4) for _ in range(10)]
# print(np.stack(arrays, axis=-1).shape)

x1 = np.arange(9).reshape((3,3))
x2 = np.arange(10,19,1).reshape((3,3))

print(x1)
print(x2)

y2 = np.stack((x1, x2), axis=-1)
print(y2.shape)
print(y2)
