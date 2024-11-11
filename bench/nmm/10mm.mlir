

module {
  func.func @_10mm(%arg0: tensor<73x77xi32>, %arg1: tensor<77x99xi32>, %arg2: tensor<99x39xi32>, %arg3: tensor<39x20xi32>, %arg4: tensor<20x76xi32>, %arg5: tensor<76x70xi32>, %arg6: tensor<70x75xi32>, %arg7: tensor<75x41xi32>, %arg8: tensor<41x84xi32>, %arg9: tensor<84x93xi32>, %arg10: tensor<93x79xi32>) -> tensor<73x79xi32> {
    %0 = tensor.empty() : tensor<73x99xi32>
    %1 = linalg.matmul ins(%arg0, %arg1 : tensor<73x77xi32>, tensor<77x99xi32>) outs(%0 : tensor<73x99xi32>) -> tensor<73x99xi32>
    %2 = tensor.empty() : tensor<73x39xi32>
    %3 = linalg.matmul ins(%1, %arg2 : tensor<73x99xi32>, tensor<99x39xi32>) outs(%2 : tensor<73x39xi32>) -> tensor<73x39xi32>
    %4 = tensor.empty() : tensor<73x20xi32>
    %5 = linalg.matmul ins(%3, %arg3 : tensor<73x39xi32>, tensor<39x20xi32>) outs(%4 : tensor<73x20xi32>) -> tensor<73x20xi32>
    %6 = tensor.empty() : tensor<73x76xi32>
    %7 = linalg.matmul ins(%5, %arg4 : tensor<73x20xi32>, tensor<20x76xi32>) outs(%6 : tensor<73x76xi32>) -> tensor<73x76xi32>
    %8 = tensor.empty() : tensor<73x70xi32>
    %9 = linalg.matmul ins(%7, %arg5 : tensor<73x76xi32>, tensor<76x70xi32>) outs(%8 : tensor<73x70xi32>) -> tensor<73x70xi32>
    %10 = tensor.empty() : tensor<73x75xi32>
    %11 = linalg.matmul ins(%9, %arg6 : tensor<73x70xi32>, tensor<70x75xi32>) outs(%10 : tensor<73x75xi32>) -> tensor<73x75xi32>
    %12 = tensor.empty() : tensor<73x41xi32>
    %13 = linalg.matmul ins(%11, %arg7 : tensor<73x75xi32>, tensor<75x41xi32>) outs(%12 : tensor<73x41xi32>) -> tensor<73x41xi32>
    %14 = tensor.empty() : tensor<73x84xi32>
    %15 = linalg.matmul ins(%13, %arg8 : tensor<73x41xi32>, tensor<41x84xi32>) outs(%14 : tensor<73x84xi32>) -> tensor<73x84xi32>
    %16 = tensor.empty() : tensor<73x93xi32>
    %17 = linalg.matmul ins(%15, %arg9 : tensor<73x84xi32>, tensor<84x93xi32>) outs(%16 : tensor<73x93xi32>) -> tensor<73x93xi32>
    %18 = tensor.empty() : tensor<73x79xi32>
    %19 = linalg.matmul ins(%17, %arg10 : tensor<73x93xi32>, tensor<93x79xi32>) outs(%18 : tensor<73x79xi32>) -> tensor<73x79xi32>
    return %19 : tensor<73x79xi32>
  }
}
