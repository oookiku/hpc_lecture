# High Performance Scientific Computing Final Report
## 20M10293 Daisuke Kikuta  
  
I selected :  
- Rewrite the 2-D N-S code in C++ (10pts)  
- CUDA (20pts)  
Total pts: 10 + 20 = 30 pts  
  
The codes I implemented are:  
1. main.cu  
   - including parameter settings, memory allocation, data output, etc...  
2. ns.hpp  
3. ns.cu  
   - including solvers for N-S eq and boundary conditions for cavity flow  
  
The answers I got from these codes are shown in "final_report.ipynb" and  
I confirmed the correctness comparing with the sample (10_cavity.ipynb).
