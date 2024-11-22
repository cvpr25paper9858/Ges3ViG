from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='imputer',
    ext_modules=[
        CUDAExtension('imputer', [
            'src/imputer.cpp',
            'src/imputer_cuda.cu',
        ], extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']})
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
