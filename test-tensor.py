import torch

### vector;
### 1.0 2.0
### 3.0 3.0
a = torch.tensor([[1., 2.], [3., 4.]])  # shape (2, 2), dtype float32
print(a)

print("shape:   {shape}".format(shape = a.shape))
print("dtype:   {dtype}".format(dtype = a.dtype))
print("device:  {device}".format(device = a.device) )

b = a.reshape(4)      # 展平
print(b)

c = a * 2 + 1         # 逐元素运算
print(c)

print(a.T)
d = a @ a.T           # 矩阵乘法

## dim=0 按列相加
## dim=1 按行相加
e = a.sum(dim=0)      # 按维度归约 -> tensor([4., 6.])
print(e)

x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x

### 对y进行求导，backward: y = 2x + 3; x.grad = 2 * 2.0 + 3 = 7;
y.backward()

### x 每增加一份，y 增加几份
print(x.grad)