# Accelerating-MATLAB-with-GPU-Computing-Example-Codes
Modified code examples for book *Suh, Kim: Accelerating MATLAB with GPU Computing: A Primer with Examples*.

In the official implementations (https://booksite.elsevier.com/9780124080805/), codes are written and compiled in a pipeline as below:
- `nvcc *.h *.cu` -> `*.obj`
- `mex *.cpp *.obj -lcudart -L"..."` -> '*.mex'
- call the mex function through `*.m`

However, in this repo, all cuda codes are integrated into one `*.cu` file and can be straight compiled via
```
mexcuda *.cu
```
