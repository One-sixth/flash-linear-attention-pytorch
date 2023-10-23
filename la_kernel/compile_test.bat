set PATH=%PATH%;"Z:\Software\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.37.32822\bin\Hostx64\x64"

@REM nvcc -c .\linear_attention_kernel.cu -I"Z:\Software\Python\Python311\Lib\site-packages\torch\include" --expt-relaxed-constexpr -w -std=c++17
nvcc -c .\linear_attention_kernel.cu -I"Z:\Software\Python\Python311\Lib\site-packages\torch\include\torch\csrc\api\include" -I"Z:\Software\Python\Python311\Lib\site-packages\torch\include" --expt-relaxed-constexpr -w -std=c++17

pause