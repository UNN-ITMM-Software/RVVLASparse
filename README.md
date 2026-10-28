# RVVLASparse

**RVVLASparse** is an open-source, high-performance C++ library designed for efficient sparse matrix operations on **RISC-V** processors. 
The library utilizes **RVV 1.0** intrinsics and **OpenMP** threading. 

*This repository accompanies the research paper: ["Rethinking Sparse Formats for RISC-V: A Hierarchical Approach to High-Performance SpMV"](https://arxiv.org/abs/2609.11352).*

### Supported Operations:
* Sparse matrix-dense vector multiplication 

### Supported Formats for SpMV:
* **HCSR** (Hierarchical CSR — our modification of CSR)
* CSR 
* SELL-C-σ
* CSR5
* LAV
* VNEC
* VHCC
* CVR

### Performance
The library has been evaluated on **SpacemiT K1 and K3** RISC-V boards. Benchmarks on 8 cores demonstrate that the HCSR format provides a significant speedup over the scalar CSR baseline:
* **Double Precision:** An average speedup of **19%** on K1 and **13%** on K3.
* **Single Precision:** An average speedup of **36%** on K1 and **34%** on K3.

