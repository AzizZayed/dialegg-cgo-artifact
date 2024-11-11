module {
  func.func @_20mm(%arg0: tensor<73x77xi32>, %arg1: tensor<77x99xi32>, %arg2: tensor<99x39xi32>, %arg3: tensor<39x20xi32>, %arg4: tensor<20x76xi32>, %arg5: tensor<76x70xi32>, %arg6: tensor<70x75xi32>, %arg7: tensor<75x41xi32>, %arg8: tensor<41x84xi32>, %arg9: tensor<84x93xi32>, %arg10: tensor<93x79xi32>, %arg11: tensor<79x61xi32>, %arg12: tensor<61x82xi32>, %arg13: tensor<82x84xi32>, %arg14: tensor<84x46xi32>, %arg15: tensor<46x36xi32>, %arg16: tensor<36x78xi32>, %arg17: tensor<78x44xi32>, %arg18: tensor<44x42xi32>, %arg19: tensor<42x12xi32>, %arg20: tensor<12x77xi32>) -> tensor<73x77xi32> {
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
    %20 = tensor.empty() : tensor<73x61xi32>
    %21 = linalg.matmul ins(%19, %arg11 : tensor<73x79xi32>, tensor<79x61xi32>) outs(%20 : tensor<73x61xi32>) -> tensor<73x61xi32>
    %22 = tensor.empty() : tensor<73x82xi32>
    %23 = linalg.matmul ins(%21, %arg12 : tensor<73x61xi32>, tensor<61x82xi32>) outs(%22 : tensor<73x82xi32>) -> tensor<73x82xi32>
    %24 = tensor.empty() : tensor<73x84xi32>
    %25 = linalg.matmul ins(%23, %arg13 : tensor<73x82xi32>, tensor<82x84xi32>) outs(%24 : tensor<73x84xi32>) -> tensor<73x84xi32>
    %26 = tensor.empty() : tensor<73x46xi32>
    %27 = linalg.matmul ins(%25, %arg14 : tensor<73x84xi32>, tensor<84x46xi32>) outs(%26 : tensor<73x46xi32>) -> tensor<73x46xi32>
    %28 = tensor.empty() : tensor<73x36xi32>
    %29 = linalg.matmul ins(%27, %arg15 : tensor<73x46xi32>, tensor<46x36xi32>) outs(%28 : tensor<73x36xi32>) -> tensor<73x36xi32>
    %30 = tensor.empty() : tensor<73x78xi32>
    %31 = linalg.matmul ins(%29, %arg16 : tensor<73x36xi32>, tensor<36x78xi32>) outs(%30 : tensor<73x78xi32>) -> tensor<73x78xi32>
    %32 = tensor.empty() : tensor<73x44xi32>
    %33 = linalg.matmul ins(%31, %arg17 : tensor<73x78xi32>, tensor<78x44xi32>) outs(%32 : tensor<73x44xi32>) -> tensor<73x44xi32>
    %34 = tensor.empty() : tensor<73x42xi32>
    %35 = linalg.matmul ins(%33, %arg18 : tensor<73x44xi32>, tensor<44x42xi32>) outs(%34 : tensor<73x42xi32>) -> tensor<73x42xi32>
    %36 = tensor.empty() : tensor<73x12xi32>
    %37 = linalg.matmul ins(%35, %arg19 : tensor<73x42xi32>, tensor<42x12xi32>) outs(%36 : tensor<73x12xi32>) -> tensor<73x12xi32>
    %38 = tensor.empty() : tensor<73x77xi32>
    %39 = linalg.matmul ins(%37, %arg20 : tensor<73x12xi32>, tensor<12x77xi32>) outs(%38 : tensor<73x77xi32>) -> tensor<73x77xi32>
    return %39 : tensor<73x77xi32>
  }
}
