import numpy as np
import pybtas
A = np.zeros((2, 2, 7))
# for i in range(7):
#   X = np.random.randn(4).reshape((2, 2))
#   X = X + X.T
#   A[:, :, i] = X
cnt = 0
for i in range(2):
  for j in range(2):
    for k in range(7):
      A[i, j, k] = cnt
      cnt += 1

A= np.ascontiguousarray(A)
A = A + A.transpose(1, 0, 2)
for i in range(2):
  for j in range(2):
    for k in range(7):
      print((i, j, k), A[i, j, k])

print()
assert np.allclose(A, A.transpose(1, 0, 2))

beta, gamma, scale = pybtas.cp3(A, 7)

print("beta")
for i in range(beta.shape[0]):
  for j in range(beta.shape[1]):
    print((i, j), beta[i, j])

print("scale")
print(scale)
Atest = np.einsum("ar,br,xr,r", beta, beta, gamma, scale.ravel())
for i in range(2):
  for j in range(2):
    for k in range(7):
      print((i, j, k), A[i, j, k] - Atest[i, j, k])

assert np.allclose(Atest, A)
