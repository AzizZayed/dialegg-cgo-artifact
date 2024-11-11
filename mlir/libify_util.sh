mkdir -p mlir/run

mlir-opt --convert-elementwise-to-linalg --convert-tensor-to-linalg \
         --convert-linalg-to-loops --one-shot-bufferize=bufferize-function-boundaries=true \
         --convert-linalg-to-loops --expand-strided-metadata --lower-affine --convert-index-to-llvm \
         --convert-math-to-llvm --convert-scf-to-cf --convert-cf-to-llvm --convert-arith-to-llvm \
         --convert-func-to-llvm --finalizing-bufferize --finalize-memref-to-llvm --reconcile-unrealized-casts \
         mlir/util.mlir -o mlir/run/util.ll.mlir

mlir-translate --mlir-to-llvmir mlir/run/util.ll.mlir -o mlir/run/util.ll

clang -dynamiclib mlir/run/util.ll -o mlir/run/libutil.dylib \
    -L/Users/aziz/dev/lib/llvm/build-release/lib -lmlir_c_runner_utils \
    -Wl,-rpath,/Users/aziz/dev/lib/llvm/build-release/lib