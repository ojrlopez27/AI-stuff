import torch
import numpy as np

print('===== 1 ======')
a = torch.tensor([1, 2, 3, 4, 5])
print(a.dtype)

print('===== 2 ======')
a = torch.tensor([1, 0.2, 3, 4.0, 5])
b = torch.tensor([1, 0.2, 3, 4.0, 5], dtype=torch.int32)
print(a.dtype, b.type)
print(a.type(), b.type())

print('===== 3 ======')
a = torch.FloatTensor([0, 1, 2, 3, 4])
print(a.dtype)
print(a.type())
print(a)

print('===== 4 ======')
a = torch.tensor([1, 2, 3, 4, 5])
print(a)
a = a.type(torch.FloatTensor)
print(a.dtype)
print(a.type())
print(a)

print('===== 5 (reshaping) ======')
a = torch.tensor([1, 2, 3, 4, 5])
print(a)
print(a.view(5, 1))
print(a.view(-1, 1))  # use -1 if you don't know the dimension

print('===== 6 (conversation to numpy) ======')
numpy_array = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
print(numpy_array)
torch_tensor = torch.from_numpy(numpy_array)
print(torch_tensor)
back_to_numpy = torch_tensor.numpy()
print(back_to_numpy)

print('===== 7 (slicing and assignment) ======')
c = torch.tensor([1, 2, 3, 4, 5, 6])
d = c[2:4]
print(d)
c[2:4] = torch.tensor([100, 200])
print(c)

print('===== 8 (operations) ======')
u = torch.tensor([3, 2])
v = torch.tensor([5, 6])
z = u + v
print(z)
a = u * 4
print(a)
b = u + 2
print(b)
c = u * v
print(c)
d = torch.dot(u, v)
print(d)


print('===== 9 (plotting) ======')
array = torch.linspace(-2, 2, steps=9)  # 9 is the number of times the interval between -2 and 2 is split
print(array)
x = torch.linspace(0, 2 * np.pi, 100)
y = torch.sin(x)
import matplotlib.pyplot as plt
plt.plot(x.numpy(), y.numpy())