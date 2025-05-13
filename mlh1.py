# Հաշվիր 2 զանգվածների Էվկլիդյան հեռավորությունը։
# 1
# a=[1,2]
# b=[1,3]

# try:
#     sum=((a[0]-b[0]**2)+(a[1]-b[1])**2)**0.5
# except:
#     print('DimensionError')
# print(sum)    


# def dist(a,b):
#     if len(a)!=len(b):
#         return "DimensionError"
#     else:
#         sum=0
#         for i in range(len(a)):
#             sum+=(a[i]-b[i])**2
#         return sum **0.5   
     
# print(dist([1,2],[1,3]))


# 2 with numpy
# import numpy as np

# def dist(a,b):
#     a=np.array(a)
#     b=np.array(b)

#     return np.sqrt(np.sum((a - b)**2)) if len(a)==len(b) else "DimensionError"
    
     
# print(dist([1,2],[1,3]))



# Նորմավորիր զանգվածը այնպես, որ արժեքները լինեն 0-ից 1 միջակայքում։

# 1
# def normalize(arr):
#     min_val = min(arr)
#     max_val = max(arr)
#     if max_val == min_val:
#         return [0 for _ in arr]
#     return [(x - min_val) / (max_val - min_val) for x in arr]



# a=[1,2,3,4]
# print(normalize(a))
    
# 2 with numpy

# import numpy as np

# def normalize(arr):
#     arr = np.array(arr)
#     min_val = arr.min()
#     max_val = arr.max()
#     if max_val == min_val:
#         return np.zeros_like(arr)  
#     return (arr - min_val) / (max_val - min_val)

# a=[1,2,3,4]
# print(normalize(a))


# 3 problem 

# def convert(a):
#     b=sorted(set(a))
#     return b,[[int(x == val) for val in b] for x in a ]

# a = [1, 1, 2, 3, 2, 4]
# b , result = convert(a)

# print(" b = " , b)
# for row in result:
#     print(row)


# with numpy

# import numpy as np

# a = np.array([1, 1, 2, 3, 2, 4])
# b = np.unique(a)

# n=len(a)
# m=len(b)

# result=np.zeros((n , m) , dtype = int)

# for i in range(n):
#     j = np.where(b == a[i])[0][0]  
#     result[i, j] = 1


# print(" b = " , b)
# print("result =\n", result)


# 4 problem

# def padding(a, n1, m1):
#     m = len(a[0])
#     row = [0] * (m + 2 * m1)
#     return (
#         [row.copy() for _ in range(n1)] +
#         [[0]*m1 + r + [0]*m1 for r in a] +
#         [row.copy() for _ in range(n1)]
#     )

# a = [[1, 1], [1, 1]]
# p=padding(a, 1 , 2)

# for padding in p:
#     print(padding) 


# with numpy

# import numpy as np

# def padding(a, n1, m1):
#     return np.pad(a, ((n1, n1), (m1, m1)), mode='constant', constant_values=0)

# a = np.array([[1, 1], [1, 1]])
# result=padding( a , 1 , 2)
# print(result)



# 5 problem

# def conv(a, b, s, p, f):
#     n, m, k = len(a), len(a[0]), len(b)
#     a = [[0]*(m+2*p) for _ in range(p)] + [[0]*p + r + [0]*p for r in a] + [[0]*(m+2*p) for _ in range(p)]
#     out = []
#     for i in range(0, len(a)-k+1, s):
#         row = []
#         for j in range(0, len(a[0])-k+1, s):
#             val = sum(a[i+x][j+y]*b[x][y] for x in range(k) for y in range(k))
#             row.append(f(val))
#         out.append(row)
#     return out

# a = [[1,1,2],
#      [0,1,3],
#      [1,3,0],
#      [4,5,2]]

# b = [[1,0],
#      [0,1]]



# stride = 1
# padding = 0
# f = lambda x: x**2

# print(conv(a, b, stride, padding, f))





# with numpy

# import numpy as np

# def conv(a, b, stride, padding, f):
    
#     a = np.pad(a, ((padding, padding), (padding, padding)), mode='constant')
#     k = b.shape[0]
#     out_h = (a.shape[0] - k) // stride + 1
#     out_w = (a.shape[1] - k) // stride + 1
#     res = np.zeros((out_h, out_w), dtype=int)

    
#     for i in range(0, out_h):
#         for j in range(0, out_w):
#             region = a[i*stride:i*stride+k, j*stride:j*stride+k]
#             res[i, j] = f(np.sum(region * b))
#     return res

# a = np.array([[1, 1, 2],
#               [0, 1, 3],
#               [1, 3, 0],
#               [4, 5, 2]])

# b = np.array([[1, 0],
#               [0, 1]])

# stride = 1
# padding = 0
# f = lambda x: x**2

# print(conv(a, b, stride, padding, f))




# 6 problem

# def predict_label(a, labels, b):
#     return labels[min(range(len(a)), key=lambda i: sum((x - y) ** 2 for x, y in zip(a[i], b)))]

# a = [[1, -2],
#      [2, 5],
#      [-3, -10],
#      [3, 2],
#      [3, 2],
#      [0, 1]]

# labels = [0, 1, 0, 1, 1, 0]
# b = [10, 10]

# print(predict_label(a, labels, b))  # >> 1


# with numpy

# import numpy as np

# def predict_label(a, labels, b):
#     dists = np.sum((a - b) ** 2, axis=1)  
#     idx = np.argmin(dists)               
#     return labels[idx]                   

# a = np.array([[1, -2],
#               [2, 5],
#               [-3, -10],
#               [3, 2],
#               [3, 2],
#               [0, 1]])

# labels = np.array([0,1,0,1,1,0])

# b = np.array([10, 10])

# print(predict_label(a, labels, b))  # >> 1



# 7 problem

# def fill(a, mode):
#     for j in range(len(a[0])):
#         col = [row[j] for row in a if row[j] is not None]
#         val = {"mean": sum(col)/len(col), "min": min(col), "max": max(col)}[mode]
#         for row in a:
#             if row[j] is None: row[j] = val
#     return a

# a = [[None, 200, 10],
#      [2, 110, None],
#      [0, 120, 11],
#      [0, 400, None],
#      [1, None, 9]]

# mode = "mean"
# res = fill(a, mode)

# for row in res:
#     print(row)


# with numpy

# import numpy as np

# def fill(a, mode):
#     a = a.copy()
#     for i in range(a.shape[1]):
#         col = a[:, i]
#         valid = col[~np.isnan(col)]
#         if mode == "mean":
#             val = np.mean(valid)
#         elif mode == "min":
#             val = np.min(valid)
#         elif mode == "max":
#             val = np.max(valid)
#         else:
#             raise ValueError("Invalid mode")
#         col[np.isnan(col)] = val
#         a[:, i] = col
#     return a

# a = np.array([[np.nan, 200, 10],
#               [2, 110, np.nan],
#               [0, 120, 11],
#               [0, 400, np.nan],
#               [1, np.nan, 9]])

# mode = "mean"
# print(fill(a, mode))




# 8 problem-



# 9 problem

# import random

# def rand_rows_cols(a, k, q):
#     m, n = len(a), len(a[0])
#     row_count = int(k * m)
#     col_count = int(q * n)
    
#     rows = [random.choice(a) for _ in range(row_count)]
#     cols = random.sample(range(n), col_count)
    
#     return [[row[i] for i in cols] for row in rows]

# a = [[1, 1, 2],
#      [0, 108, 3],
#      [1, 3, 65],
#      [50, 35, 5],
#      [5, 83, 110],
#      [98, 99, 10],
#      [8, 9, 103],
#      [9, 23, 15]]

# k = 0.25
# q = 0.6

# print(rand_rows_cols(a, k, q))



# with numpy

# import numpy as np

# def rand_rows_cols(a, k, q):
#     m, n = a.shape
#     row_count = int(round(k * m))
#     col_count = int(round(q * n))
    
#     rows = np.random.choice(m, row_count, replace=True)
#     cols = np.random.choice(n, col_count, replace=False)
    
#     return a[rows][:, cols]

# a = np.array([[1, 1, 2],
#               [0, 108, 3],
#               [1, 3, 65],
#               [50, 35, 5],
#               [5, 83, 110],
#               [98, 99, 10],
#               [8, 9, 103],
#               [9, 23, 15]])

# k = 0.25
# q = 0.6

# print(rand_rows_cols(a, k, q))




# 10 peoblem

# def between_100_200(a):
#     return [row for row in a if 100 <= sum(row) < 200]

# a = [[1, 1, 2],
#      [0, 108, 3],
#      [1, 3, 65],
#      [50, 35, 5],
#      [5, 83, 110],
#      [98, 99, 10],
#      [8, 9, 103],
#      [9, 23, 15]]

# print(between_100_200(a))


# with numpy
# import numpy as np

# def between_100_200(a):
#     row_sums = np.sum(a, axis=1)
#     return a[(row_sums >= 100) & (row_sums < 200)]


# a = np.array([[1, 1, 2],
#               [0, 108, 3],
#               [1, 3, 65],
#               [50, 35, 5],
#               [5, 83, 110],
#               [98, 99, 10],
#               [8, 9, 103],
#               [9, 23, 15]])

# print(between_100_200(a))



# 11 problem


# import math

# def Cholesky(x):
#     n = len(x)
#     L = [[0] * n for _ in range(n)]  
#     for i in range(n):
#         for j in range(i+1):
#             if i == j:
#                 L[i][j] = math.sqrt(x[i][i] - sum(L[i][k] ** 2 for k in range(j)))
#             else:
#                 L[i][j] = (x[i][j] - sum(L[i][k] * L[j][k] for k in range(j))) / L[j][j]
#     return L

# x = [[4, 12, -16], [12, 37, -43], [-16, -43, 98]]

# L = Cholesky(x)

# for row in L:
#     print(row)



# with numpy

# import numpy as np

# def Cholesky(x):
#     return np.linalg.cholesky(x)

# x = np.array([[4, 12, -16], 
#               [12, 37, -43], 
#               [-16, -43, 98]], dtype=np.int32)

# L = Cholesky(x)

# print(L)


# 12 problem

# def zero_one(n):
#     return [[1 if i == 0 or i == n-1 or j == 0 or j == n-1 else 0 for j in range(n)] for i in range(n)]
# n = 5
# arr = zero_one(n)
# for row in arr:
#     print(row)


# with numpy
import numpy as np

def zero_one(n):
    arr = np.zeros((n, n), dtype=int)  
    arr[0, :] = 1 
    arr[-1, :] = 1  
    arr[:, 0] = 1 
    arr[:, -1] = 1  
    return arr

n = 5
arr = zero_one(n)
print(arr)
