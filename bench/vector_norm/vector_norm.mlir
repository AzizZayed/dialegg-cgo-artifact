func.func private @printNewline()
func.func private @clock() -> i64
func.func private @displayTime(i64, i64)
func.func private @printF32Tensor2D(tensor<?x?xf32>)

func.func @fillRandomF64Tensor2D(%tensor: tensor<?x?xf64>) -> tensor<?x?xf64> {
    // Create a 2D tensor with random values with the linalg.fill_rng_2d op

    %seed = arith.constant 0 : i32
    %min = arith.constant -10.0 : f64
    %max = arith.constant 10.0 : f64

    %tensor_filled = linalg.fill_rng_2d ins(%min, %max, %seed : f64, f64, i32) 
                                        outs(%tensor : tensor<?x?xf64>) -> tensor<?x?xf64>

    return %tensor_filled : tensor<?x?xf64>
}

// 3 func, 4 arith, 7 tensor, 2 scf
func.func @normalize_distance_vectors(%vectors: tensor<1000000x3xf32>) -> tensor<1000000x3xf32> {
    // Distance between multiple 3D points and normalizing the resulting vectors. This is a common operation in computer graphics and physics simulations.

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c1000000 = arith.constant 1000000 : index

    %norm_vectors_init = tensor.empty() : tensor<1000000x3xf32>
    %norm_vectors = scf.for %i = %c0 to %c1000000 step %c1 iter_args(%current_vector = %norm_vectors_init) -> (tensor<1000000x3xf32>) {
        %x = tensor.extract %vectors[%i, %c0] : tensor<1000000x3xf32>
        %y = tensor.extract %vectors[%i, %c1] : tensor<1000000x3xf32>
        %z = tensor.extract %vectors[%i, %c2] : tensor<1000000x3xf32>

        %nx, %ny, %nz = func.call @normalize_vector(%x, %y, %z) : (f32, f32, f32) -> (f32, f32, f32)

        %normalized_vector1 = tensor.insert %nx into %current_vector[%i, %c0] : tensor<1000000x3xf32>
        %normalized_vector2 = tensor.insert %ny into %normalized_vector1[%i, %c1] : tensor<1000000x3xf32>
        %normalized_vector3 = tensor.insert %nz into %normalized_vector2[%i, %c2] : tensor<1000000x3xf32>

        scf.yield %normalized_vector3 : tensor<1000000x3xf32>
    }

    func.return %norm_vectors : tensor<1000000x3xf32>
}

func.func @main() -> i32 {
    %c1000000 = arith.constant 1000000 : index
    %c3 = arith.constant 3 : index

    %points_cast = tensor.empty(%c1000000, %c3) : tensor<?x?xf64>
    %points_filled = func.call @fillRandomF64Tensor2D(%points_cast) : (tensor<?x?xf64>) -> tensor<?x?xf64>
    %points_trunc = arith.truncf %points_filled : tensor<?x?xf64> to tensor<?x?xf32>
    %points = tensor.cast %points_trunc : tensor<?x?xf32> to tensor<1000000x3xf32>

    %start = func.call @clock() : () -> i64
    %vectors_normalized = func.call @normalize_distance_vectors(%points) : (tensor<1000000x3xf32>) -> tensor<1000000x3xf32>
    %end = func.call @clock() : () -> i64

    %vectors_normalized_cast = tensor.cast %vectors_normalized : tensor<1000000x3xf32> to tensor<?x?xf32>
    func.call @printF32Tensor2D(%vectors_normalized_cast) : (tensor<?x?xf32>) -> ()
    func.call @printNewline() : () -> ()
    func.call @displayTime(%start, %end) : (i64, i64) -> ()

    %c0 = arith.constant 0 : i32
    func.return %c0 : i32
}

// 2 func, 13 arith
func.func @fast_inv_sqrt(%x: f32) -> f32 {
    // C code from https://en.wikipedia.org/wiki/Fast_inverse_square_root
    // float Q_rsqrt(float number) {
    //     long i;
    //     float x2, y;
    //     const float threehalfs = 1.5F;

    //     x2 = number * 0.5F;
    //     y  = number;
    //     i  = * ( long * ) &y;                       // evil floating point bit level hacking
    //     i  = 0x5f3759df - ( i >> 1 );               // what the fuck?
    //     y  = * ( float * ) &i;
    //     y  = y * ( threehalfs - ( x2 * y * y ) );   // 1st iteration
    //     // y  = y * ( threehalfs - ( x2 * y * y ) );   // 2nd iteration, this can be removed

    //     return y;
    // }

    %c1 = arith.constant 1 : i32
    %half = arith.constant 0.5 : f32
    %three_halfs = arith.constant 1.5 : f32
    %magic_number = arith.constant 0x5f3759df : i32

    %x2 = arith.mulf %x, %half : f32 // x2 = number * 0.5F
    %i = arith.bitcast %x : f32 to i32 // i = * ( long * ) &y
    %i_shifted = arith.shrsi %i, %c1 : i32 // i >> 1
    %i_subbed = arith.subi %magic_number, %i_shifted : i32 // i  = 0x5f3759df - ( i >> 1 );
    %y = arith.bitcast %i_subbed : i32 to f32 // y = * ( float * ) &i

    %y2 = arith.mulf %y, %y : f32 // y2 = y * y
    %x2y2 = arith.mulf %x2, %y2 : f32 // x2y2 = x2 * y * y
    %sub = arith.subf %three_halfs, %x2y2 : f32 // sub = threehalfs - (x2 * y * y)
    %y_it = arith.mulf %y, %sub : f32 // y_times_sub = y * (threehalfs - (x2 * y * y))

    func.return %y_it : f32
}

// 2 func, 10 arith, 1 math
func.func @normalize_vector(%x: f32, %y: f32, %z: f32) -> (f32, f32, f32) {
    %c1_f32 = arith.constant 1.0 : f32

    %x_squared = arith.mulf %x, %x : f32
    %y_squared = arith.mulf %y, %y : f32
    %z_squared = arith.mulf %z, %z : f32

    %distance_squared_1 = arith.addf %x_squared, %y_squared : f32
    %distance_squared = arith.addf %distance_squared_1, %z_squared : f32 // distance_squared = x^2 + y^2 + z^2

    %distance = math.sqrt %distance_squared fastmath<fast> : f32 // distance = sqrt(x^2 + y^2 + z^2)
    %inv_distance = arith.divf %c1_f32, %distance fastmath<fast> : f32 // inv_distance = 1 / sqrt(x^2 + y^2 + z^2)

    %x_normalized = arith.mulf %x, %inv_distance : f32
    %y_normalized = arith.mulf %y, %inv_distance : f32
    %z_normalized = arith.mulf %z, %inv_distance : f32

    func.return %x_normalized, %y_normalized, %z_normalized : f32, f32, f32
}