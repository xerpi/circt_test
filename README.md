```bash
cmake .. \
  -DLLVM_DIR=$PWD/../../circt/llvm/build/lib/cmake/llvm \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCIRCT_DIR=$PWD/../../circt/build/lib/cmake/circt/ \
  -DMLIR_DIR=$PWD/../../circt/llvm/build/lib/cmake/mlir/
```
