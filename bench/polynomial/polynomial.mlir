func.func private @printNewline()
func.func private @clock() -> i64
func.func private @displayTime(i64, i64)
func.func private @printF64Tensor1D(tensor<?xf64>)

func.func @fillRandomF64Tensor2D(%tensor: tensor<?x?xf64>) -> tensor<?x?xf64> {
    // Create a 2D tensor with random values with the linalg.fill_rng_2d op

    %seed = arith.constant 0 : i32
    %min = arith.constant -10.0 : f64
    %max = arith.constant 10.0 : f64

    %tensor_filled = linalg.fill_rng_2d ins(%min, %max, %seed : f64, f64, i32) 
                                        outs(%tensor : tensor<?x?xf64>) -> tensor<?x?xf64>

    return %tensor_filled : tensor<?x?xf64>
}

// 5 arith, 6 tensor, 2 scf, 1 func
func.func @main() -> i32 {
    // polynomial a + bx + cx^2 + dx^3
    %c1000000 = arith.constant 1000000 : index
    %c4 = arith.constant 4 : index

    %tensor_cast = tensor.empty(%c1000000, %c4) : tensor<?x?xf64>
    %tensor_filled = func.call @fillRandomF64Tensor2D(%tensor_cast) : (tensor<?x?xf64>) -> tensor<?x?xf64>
    %tensor = tensor.cast %tensor_filled : tensor<?x?xf64> to tensor<1000000x4xf64>

    %start = func.call @clock() : () -> i64
    
    %x = arith.constant 5.0 : f64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    %poly_eval_init = tensor.empty() : tensor<1000000xf64>
    %poly_eval = scf.for %i = %c0 to %c1000000 step %c1 iter_args(%current_poly_eval = %poly_eval_init) -> (tensor<1000000xf64>) {
        %a = tensor.extract %tensor[%i, %c0] : tensor<1000000x4xf64>
        %b = tensor.extract %tensor[%i, %c1] : tensor<1000000x4xf64>
        %c = tensor.extract %tensor[%i, %c2] : tensor<1000000x4xf64>
        %d = tensor.extract %tensor[%i, %c3] : tensor<1000000x4xf64>

        %num = func.call @poly_eval_3(%a, %b, %c, %d, %x) : (f64, f64, f64, f64, f64) -> f64
        %updated_poly_eval = tensor.insert %num into %current_poly_eval[%i] : tensor<1000000xf64>
        scf.yield %updated_poly_eval : tensor<1000000xf64>
    }

    %end = func.call @clock() : () -> i64

    %poly_eval_cast = tensor.cast %poly_eval : tensor<1000000xf64> to tensor<?xf64>
    func.call @printF64Tensor1D(%poly_eval_cast) : (tensor<?xf64>) -> ()
    func.call @printNewline() : () -> ()
    func.call @displayTime(%start, %end) : (i64, i64) -> ()

    %c0_i32 = arith.constant 0 : i32
    func.return %c0_i32 : i32
}

// 2 func, 8 arith, 2 math
func.func @poly_eval_3(%a: f64, %b: f64, %c: f64, %d: f64, %x: f64) -> f64 {
    %c2 = arith.constant 2.0 : f64
    %c3 = arith.constant 3.0 : f64

    %x_2 = math.powf %x, %c2 : f64 // x^2
    %x_3 = math.powf %x, %c3 : f64 // x^3

    %t1 = arith.mulf %c, %x  : f64 // cx
    %t2 = arith.mulf %b, %x_2  : f64 // bx^2
    %t3 = arith.mulf %a, %x_3  : f64 // ax^3

    %t4 = arith.addf %t2, %t3 : f64 // ax^3 + bx^2
    %t5 = arith.addf %t1, %t4 : f64 // cx + ax^3 + bx^2
    %t6 = arith.addf %d, %t5 : f64 // d + cx + ax^3 + bx^2

    func.return %t6 : f64
}
