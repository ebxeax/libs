##### 单层感知机
![slpmodel](https://gitee.com/Carrawayang/markdown-picture-res/raw/master/singleLayerPerceptron.png)
$$
\begin{aligned}
& y = XW + b \\
& y = \sum x_i*w_i+b\\
\end{aligned}
$$

##### Derivative
$$
\begin{aligned}
&E=\frac{1}{2}(O^1_0-t)^2\\
&\frac{\delta E}{\delta W_{j0}}=(O_0-t)\frac{\delta O_0}{\delta w_{j0}}\\
&=(O_0-t)\frac{\delta O_0}{\delta w_{j0}}\\
&=(O_0-t)\delta(x_0)(1-\delta(x_0))\frac{\delta x_0^1}{\delta w_j^0}\\
&=(O_0-t)O_0(1-O_0)\frac{\delta x_0^1}{\delta w_j^0}\\
&=(O_0-t)O_0(1-O_0)x_j^0
\end{aligned}
$$


```python
import torch,torch.nn.functional as F
```


```python
x = torch.randn(1, 10)
w = torch.randn(1, 10, requires_grad=True)
o = torch.sigmoid(x@w.t())
o.shape
```




    torch.Size([1, 1])




```python
loss = F.mse_loss(torch.ones(1, 1), o)
loss.shape
```




    torch.Size([])




```python
loss.backward()
```


```python
w.grad
```




    tensor([[-0.1801,  0.1923,  0.2480, -0.0919,  0.1487,  0.0196, -0.1588, -0.1652,
              0.3811, -0.2290]])



##### Multi-output Perceptron
![mop](https://gitee.com/Carrawayang/markdown-picture-res/raw/master/Multi-outputPerceptron.png)
##### Derivative
$$
\begin{aligned}
&E=\frac{1}{2}(O^1_i-t)^2\\
&\frac{\delta E}{\delta W_{jk}}=(O_k-t_k)\frac{\delta O_k}{\delta w_{jk}}\\
&=(O_k-t)\frac{\delta O_0}{\delta w_{j0}}\\
&=(O_k-t)\delta(x_0)(1-\delta(x_0))\frac{\delta x_0^1}{\delta w_j^0}\\
&=(O_k-t)O_0(1-O_0)\frac{\delta x_0^1}{\delta w_j^0}\\
&=(O_k-t)O_0(1-O_0)x_j^0
\end{aligned}
$$
