mlir-opt --convert-elementwise-to-linalg --convert-tensor-to-linalg \
         --convert-linalg-to-loops --one-shot-bufferize=bufferize-function-boundaries=true \
         --convert-linalg-to-loops --expand-strided-metadata --lower-affine --convert-index-to-llvm \
         --convert-math-to-llvm --convert-scf-to-cf --convert-cf-to-llvm --convert-arith-to-llvm \
         --convert-func-to-llvm --finalizing-bufferize --finalize-memref-to-llvm --reconcile-unrealized-casts \
         mlir/util.mlir -o mlir/util.ll.mlir

mlir-translate --mlir-to-llvmir mlir/util.ll.mlir -o mlir/util.ll

if [[ "$OSTYPE" == "darwin"* ]]; then
    clang -dynamiclib mlir/util.ll -o mlir/libutil.dylib \
        -L/llvm/build-release/lib -lmlir_c_runner_utils \
        -Wl,-rpath,/llvm/build-release/lib
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    clang -shared mlir/util.ll -o mlir/libutil.so \
        -L/llvm/build-release/lib -lmlir_c_runner_utils \
        -Wl,-rpath,/llvm/build-release/lib
else
    echo "Unsupported OS"
    exit 1
fi
