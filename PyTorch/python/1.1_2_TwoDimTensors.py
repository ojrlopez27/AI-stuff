import torch
import pandas as pd

print('============ 1 Basics')
a = [[11, 12, 13], [21, 22, 23], [31, 32, 33]]
t = torch.tensor(a)
print(t.ndimension())
print(t.shape)
print(t.size())
print(t.numel())


print('============ 2 Slicing')
print(t[1, 2]) # 23
print(t[1:3, 2]) # [23, 33]
print(t[1, 1:3]) # [22, 23]


print('============ 3 Operations')
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
y = torch.tensor([[0, 2, 4], [6, 8, 0]])
z = torch.tensor([[0, 2], [4, 6], [8, 0]])
print('Addition: ', x + y)
print('Scalar addition: ', x + 5)
print('Scalar multiplication: ', x * 3)
print('Tensor product: ', x * y)
print('Matrix multiplication: ', torch.mm(x, z))

print('============ 4 Pandas')
df = pd.DataFrame({'a':[11,21,31],'b':[12,22,312]})
print("Pandas Dataframe to numpy: ", df.values)
print("Type BEFORE converting: ", df.values.dtype)
new_tensor = torch.from_numpy(df.values)
print("Tensor AFTER converting: ", new_tensor)
print("Type AFTER converting: ", new_tensor.dtype)