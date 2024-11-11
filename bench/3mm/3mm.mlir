func.func private @printNewline()
func.func private @clock() -> i64
func.func private @displayTime(i64, i64)
func.func private @printI64Tensor2D(tensor<?x?xi64>)

func.func @fillRandomI64Tensor2D(%tensor: tensor<?x?xi64>) -> tensor<?x?xi64> {
    // Create a 2D tensor with random values with the linalg.fill_rng_2d op

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %cols = tensor.dim %tensor, %c1 : tensor<?x?xi64>
    %rows = tensor.dim %tensor, %c0 : tensor<?x?xi64>

    %seed = arith.constant 0 : i32
    %min = arith.constant -10.0 : f64
    %max = arith.constant 10.0 : f64
    %init = tensor.empty(%rows, %cols) : tensor<?x?xf64>

    %init_filled = linalg.fill_rng_2d ins(%min, %max, %seed : f64, f64, i32) 
                                      outs(%init : tensor<?x?xf64>) -> tensor<?x?xf64>

    // Floor each value and cast to i64 with generic op
    %init_floor = linalg.floor ins(%init_filled : tensor<?x?xf64>) outs(%init : tensor<?x?xf64>) -> tensor<?x?xf64>
    %tensor_filled = arith.fptosi %init_floor : tensor<?x?xf64> to tensor<?x?xi64>

    return %tensor_filled : tensor<?x?xi64>
}

func.func @main() -> i32 {
    %c200 = arith.constant 200 : index
    %c175 = arith.constant 175 : index
    %c250 = arith.constant 250 : index
    %c150 = arith.constant 150 : index
    %c10 = arith.constant 10 : index

    %x_cast = tensor.empty(%c200, %c175) : tensor<?x?xi64>
    %y_cast = tensor.empty(%c175, %c250) : tensor<?x?xi64>
    %z_cast = tensor.empty(%c250, %c150) : tensor<?x?xi64>
    %w_cast = tensor.empty(%c150, %c10) : tensor<?x?xi64>

    %x_filled = func.call @fillRandomI64Tensor2D(%x_cast) : (tensor<?x?xi64>) -> tensor<?x?xi64>
    %y_filled = func.call @fillRandomI64Tensor2D(%y_cast) : (tensor<?x?xi64>) -> tensor<?x?xi64>
    %z_filled = func.call @fillRandomI64Tensor2D(%z_cast) : (tensor<?x?xi64>) -> tensor<?x?xi64>
    %w_filled = func.call @fillRandomI64Tensor2D(%w_cast) : (tensor<?x?xi64>) -> tensor<?x?xi64>

    %x = tensor.cast %x_filled : tensor<?x?xi64> to tensor<200x175xi64>
    %y = tensor.cast %y_filled : tensor<?x?xi64> to tensor<175x250xi64>
    %z = tensor.cast %z_filled : tensor<?x?xi64> to tensor<250x150xi64>
    %w = tensor.cast %w_filled : tensor<?x?xi64> to tensor<150x10xi64>

    %start = func.call @clock() : () -> i64  // Start measuring time
    %res = func.call @_3mm(%x, %y, %z, %w) : (tensor<200x175xi64>, tensor<175x250xi64>, tensor<250x150xi64>, tensor<150x10xi64>) -> tensor<200x10xi64>
    %end = func.call @clock() : () -> i64  // End measuring time

    %res_cast = tensor.cast %res : tensor<200x10xi64> to tensor<?x?xi64>
    func.call @printI64Tensor2D(%res_cast) : (tensor<?x?xi64>) -> ()
    func.call @printNewline() : () -> ()
    func.call @displayTime(%start, %end) : (i64, i64) -> ()

    %c0 = arith.constant 0 : i32
    func.return %c0 : i32
}

// 2 func, 3 tensor, 3 linalg
func.func @_3mm(%x: tensor<200x175xi64>, %y: tensor<175x250xi64>, %z: tensor<250x150xi64>, %w: tensor<150x10xi64>) -> tensor<200x10xi64> {
    %xy_init = tensor.empty() : tensor<200x250xi64>
    %xy = linalg.matmul ins(%x, %y : tensor<200x175xi64>, tensor<175x250xi64>) 
                        outs(%xy_init : tensor<200x250xi64>) -> tensor<200x250xi64>
    
    %xy_z_init = tensor.empty() : tensor<200x150xi64>
    %xy_z = linalg.matmul ins(%xy, %z : tensor<200x250xi64>, tensor<250x150xi64>) 
                        outs(%xy_z_init : tensor<200x150xi64>) -> tensor<200x150xi64>

    %xy_z__w_init = tensor.empty() : tensor<200x10xi64>
    %xy_z__w = linalg.matmul ins(%xy_z, %w : tensor<200x150xi64>, tensor<150x10xi64>)
                            outs(%xy_z__w_init : tensor<200x10xi64>) -> tensor<200x10xi64>
    
    func.return %xy_z__w : tensor<200x10xi64>
}