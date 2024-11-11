func.func private @printI64(i64)
func.func private @printF64(f64)
func.func private @printComma()
func.func private @printNewline()

// C funcs
func.func private @clock() -> i64
func.func private @putchar(i32) -> i32
llvm.func @printf(!llvm.ptr, ...) -> i32

func.func private @printF64Tensor1D(tensor<?xf64>)
func.func private @printF64Tensor2D(tensor<?x?xf64>)
func.func private @displayTime(i64, i64)

// The arith dialct operations take primitive-like types as arguments.
// So a float-like type is represented as f32, f64 or any collection-type like vector<?xf32> or tensor<?xf32>.
func.func @main() -> i64 {
    %start = func.call @clock() : () -> i64  // Start measuring time

    // tensor constants
    %tensor0 = arith.constant dense<[[1., 2., 3.], [4., 5., 6.]]> : tensor<2x3xf64>
    %tensor1 = arith.constant dense<[[2., 3., 4.], [5., 6., 7.]]> : tensor<2x3xf64>

    // Add two tensors
    %tensor2 = arith.addf %tensor0, %tensor1 : tensor<2x3xf64>
    %tensor_cast = tensor.cast %tensor2 : tensor<2x3xf64> to tensor<?x?xf64>
    func.call @printF64Tensor2D(%tensor_cast) : (tensor<?x?xf64>) -> ()
    func.call @printNewline() : () -> ()

    %end = func.call @clock() : () -> i64  // End measuring time
    func.call @displayTime(%start, %end) : (i64, i64) -> ()

    %zero = arith.constant 0 : i64
    func.return %zero : i64
}