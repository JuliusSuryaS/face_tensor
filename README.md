# face_tensor
###### FaceWarehouse tensor reduction and obtaining single face vertices.
###### Simple HOOSVD operation on Matlab and C++ matrix operation to obtain vertices.
---
Prerequisite
* Matlab
* OpenGL (visualization)
* Eigen C++

The Simple HOOSVD operation is done with matlab (not optimized).\
Obtaining and visualizing the vertices is done on C++ with Eigen library and OpenGL. 
---

##### Algorithm Overview
Tensor decomposition properties
Tensor is defined as mode multiplication of its core tensor with each corresponding mode, \
![Tensor](/img/tensor_eq.png)

Core tensor is defined as the transpose mode multiplication of its full tensor, \
![Tensor](/img/core_eq.png)

###### HOOSVD
Operations on tensor is done by unfolding the tensor into matrix. \
The tensor can be folded into each mode, \
![Tensor](/img/unfold_tensor.png)

U_1, U_2, U_3 is obtained by performing SVD operations on each mode.\
The mode multiplication is equivalent to kronecker product.


The main goal of this work to obtain the vertices (mode 1), therefore the mode 1 can be set fixed. \
![Tensor](/img/unfold_tensor_reduced.png)

The other modes (mode 2 and 3) correspond to Identity and Expression can be reduced (Similar to SVD reduction).\
U can be reduced by keeping the high variance on the left only. \
The original tensor estimation can be obtained by keeping the U_1 fixed with reduced U_2 and U_3, \
![Tensor](/img/reduced_tensor_u1.png)

To obtain single face vertices, the reduced U_2 and U_3 can be further reduced by using only the row values.
![Tensor](/img/single_vert.png)

Output of reduced face tensor approximation. (Reduce the very large core tensor (original ~1GB binary file) to smaller size)
![Face](/img/output.png)
