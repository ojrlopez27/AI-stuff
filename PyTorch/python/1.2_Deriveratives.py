import torch
import matplotlib.pylab as plt


# Create a tensor x
# set the parameter requires_grad to true because you are going to take the derivative of the tensor.
x = torch.tensor(2.0, requires_grad = True)
print("The tensor x: ", x)

# Create a tensor y according to y = x^2
y = x ** 2
print("The result of y = x^2: ", y)

# Take the derivative. Try to print out the derivative at the value x = 2
y.backward()
print("The dervative at x = 2: ", x.grad)

print("="*50)
# Determine the derivative of  𝑦=2𝑥^3+𝑥, at  𝑥=1
x = torch.tensor(1.0, requires_grad=True)
y = 2 * x ** 3 + x
y.backward()
print("The derivative result: ", x.grad)


print("="*50)
# Partial deriveratives
# Calculate f(u, v) = v * u + u^2 at u = 1, v = 2
u = torch.tensor(1.0,requires_grad=True)
v = torch.tensor(2.0,requires_grad=True)
f = u * v + u ** 2
print("The result of v * u + u^2: ", f)
# Calculate the derivative with respect to u
f.backward()
print("The partial derivative with respect to u: ", u.grad)
# Calculate the derivative with respect to v
print("The partial derivative with respect to u: ", v.grad)

