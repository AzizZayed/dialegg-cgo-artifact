module {
  func.func @_4mm(%arg0: tensor<73x77xi32>, %arg1: tensor<77x99xi32>, %arg2: tensor<99x39xi32>, %arg3: tensor<39x20xi32>, %arg4: tensor<20x76xi32>) -> tensor<73x76xi32> {
    %0 = tensor.empty() : tensor<73x99xi32>
    %1 = linalg.matmul ins(%arg0, %arg1 : tensor<73x77xi32>, tensor<77x99xi32>) outs(%0 : tensor<73x99xi32>) -> tensor<73x99xi32>
    %2 = tensor.empty() : tensor<73x39xi32>
    %3 = linalg.matmul ins(%1, %arg2 : tensor<73x99xi32>, tensor<99x39xi32>) outs(%2 : tensor<73x39xi32>) -> tensor<73x39xi32>
    %4 = tensor.empty() : tensor<73x20xi32>
    %5 = linalg.matmul ins(%3, %arg3 : tensor<73x39xi32>, tensor<39x20xi32>) outs(%4 : tensor<73x20xi32>) -> tensor<73x20xi32>
    %6 = tensor.empty() : tensor<73x76xi32>
    %7 = linalg.matmul ins(%5, %arg4 : tensor<73x20xi32>, tensor<20x76xi32>) outs(%6 : tensor<73x76xi32>) -> tensor<73x76xi32>
    return %7 : tensor<73x76xi32>
  }
}
