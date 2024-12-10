# source:  https://www.youtube.com/watch?v=krEEwRH-VLo&ab_channel=JustData


import torch

torch.manual_seed(42)


x = torch.arange(start=0, end=12)
print(x)

print(x.shape)
new_x = x.reshape(3, 4)
print(new_x)

# -1 calcula automaticamente la otra dimension, es decir 4
new_x = x.reshape(3, -1)
print(new_x)

#  Gives error because 5 is not a multiplier of 12
# new_x = x.reshape(5, -1)
# print(new_x)

zeros = torch.zeros((3, 2, 3))
ones = torch.ones((3, 2, 3))

# crear tensores con numeros aleatorios
rnd = torch.randn(3, 3, 2, 2)
rnd1 = torch.randint(-20, 20, (4, 5) )  # (4, 5) es el tamaño del tensor

# acceder filas y columnas especificas
# -1 indica hasta el final
X = rnd1[1:3, 1:-1]  # 1:3 => filas 1 y 2 (3 no incluido), 1:-1 => columnas 1, 2 (la última col que es -1 no se incluye)
print(X)


# acceder las dos ultimas columnas de un tensor de 3 dimensiones
rnd2 = torch.randint(-20, 20, (3, 4, 5)) # 3 matrices con 4 filas y 5 columnas, con numeros aleatorios de -20 a 20
X = rnd2[:, :, -2:]
print(X)

# suma las componentes por columnas
s1 = rnd2.float().sum(dim=0)
# suma las componentes por filas
s2 = rnd2.sum(dim=1)
print('s1: ', s1, 's2: ', s2)