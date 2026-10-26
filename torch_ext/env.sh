# Build environment on cosmopc27: torch 2.13+cu130 needs a CUDA 13 nvcc; the
# system toolkit is 12.6, so nvcc 13.0 comes from pip wheels in pydeps/
# (nvidia-cuda-nvcc/crt/nvvm/cccl/runtime 13.0.x, ninja).
B=$HOME/bench/attn-op
export CUDA_HOME=$B/cuda13
if [ ! -e $CUDA_HOME/bin/nvcc ]; then
  C=$B/pydeps/nvidia/cu13
  mkdir -p $CUDA_HOME
  for x in bin include nvvm; do ln -sfn $C/$x $CUDA_HOME/$x; done
  mkdir -p $CUDA_HOME/lib64
  for f in $C/lib/*; do ln -sfn $f $CUDA_HOME/lib64/; done
  ln -sfn libcudart.so.13 $CUDA_HOME/lib64/libcudart.so
fi
export PATH=$CUDA_HOME/bin:$B/pydeps/bin:$PATH
export PYTHONPATH=$B/pydeps_py:$B/src/torch_ext:$PYTHONPATH
export TORCH_CUDA_ARCH_LIST=8.9
export MLIPCPP_TORCH_BUILD=$B/build
PY=$HOME/bench/venv/bin/python
