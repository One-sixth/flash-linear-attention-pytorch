set PATH=%PATH%;"Z:\Software\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.37.32822\bin\Hostx64\x64"

nvcc -c .\la_cuda.cu -I"Z:\Software\Python\Python311\Lib\site-packages\torch\include" --expt-relaxed-constexpr

pause