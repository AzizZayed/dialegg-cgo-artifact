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
    %c100 = arith.constant 100 : index
    %c10 = arith.constant 10 : index
    %c150 = arith.constant 150 : index
    %c8 = arith.constant 8 : index

    %x_cast = tensor.empty(%c100, %c10) : tensor<?x?xi64>
    %y_cast = tensor.empty(%c10, %c150) : tensor<?x?xi64>
    %z_cast = tensor.empty(%c150, %c8) : tensor<?x?xi64>

    %x_filled = func.call @fillRandomI64Tensor2D(%x_cast) : (tensor<?x?xi64>) -> tensor<?x?xi64>
    %y_filled = func.call @fillRandomI64Tensor2D(%y_cast) : (tensor<?x?xi64>) -> tensor<?x?xi64>
    %z_filled = func.call @fillRandomI64Tensor2D(%z_cast) : (tensor<?x?xi64>) -> tensor<?x?xi64>

    %x = tensor.cast %x_filled : tensor<?x?xi64> to tensor<100x10xi64>
    %y = tensor.cast %y_filled : tensor<?x?xi64> to tensor<10x150xi64>
    %z = tensor.cast %z_filled : tensor<?x?xi64> to tensor<150x8xi64>

    %start = func.call @clock() : () -> i64  // Start measuring time
    %res = func.call @_2mm(%x, %y, %z) : (tensor<100x10xi64>, tensor<10x150xi64>, tensor<150x8xi64>) -> tensor<100x8xi64>
    %end = func.call @clock() : () -> i64  // End measuring time

    %res_cast = tensor.cast %res : tensor<100x8xi64> to tensor<?x?xi64>
    func.call @printI64Tensor2D(%res_cast) : (tensor<?x?xi64>) -> ()
    func.call @printNewline() : () -> ()
    func.call @displayTime(%start, %end) : (i64, i64) -> ()

    %c0 = arith.constant 0 : i32
    func.return %c0 : i32
}

// 2 func, 2 tensor, 2 linalg
func.func @_2mm(%x: tensor<100x10xi64>, %y: tensor<10x150xi64>, %z: tensor<150x8xi64>) -> tensor<100x8xi64> {
    // xy_z cost ac(b+d) = 100*150*(10+8) = 2,700,000
    // x_yz cost bd(c+a) = 10*8*(150+100) = 2,000
    %xy_init = tensor.empty() : tensor<100x150xi64>
    %xy = linalg.matmul ins(%x, %y : tensor<100x10xi64>, tensor<10x150xi64>) 
                        outs(%xy_init : tensor<100x150xi64>) -> tensor<100x150xi64>
    
    %xy_z_init = tensor.empty() : tensor<100x8xi64>
    %xy_z = linalg.matmul ins(%xy, %z : tensor<100x150xi64>, tensor<150x8xi64>) 
                          outs(%xy_z_init : tensor<100x8xi64>) -> tensor<100x8xi64>
    
    func.return %xy_z : tensor<100x8xi64>
}