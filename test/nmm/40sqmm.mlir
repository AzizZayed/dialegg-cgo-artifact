module {
  func.func @_40mm(%arg0: tensor<100x100xi32>, %arg1: tensor<100x100xi32>, %arg2: tensor<100x100xi32>, %arg3: tensor<100x100xi32>, %arg4: tensor<100x100xi32>, %arg5: tensor<100x100xi32>, %arg6: tensor<100x100xi32>, %arg7: tensor<100x100xi32>, %arg8: tensor<100x100xi32>, %arg9: tensor<100x100xi32>, %arg10: tensor<100x100xi32>, %arg11: tensor<100x100xi32>, %arg12: tensor<100x100xi32>, %arg13: tensor<100x100xi32>, %arg14: tensor<100x100xi32>, %arg15: tensor<100x100xi32>, %arg16: tensor<100x100xi32>, %arg17: tensor<100x100xi32>, %arg18: tensor<100x100xi32>, %arg19: tensor<100x100xi32>, %arg20: tensor<100x100xi32>, %arg21: tensor<100x100xi32>, %arg22: tensor<100x100xi32>, %arg23: tensor<100x100xi32>, %arg24: tensor<100x100xi32>, %arg25: tensor<100x100xi32>, %arg26: tensor<100x100xi32>, %arg27: tensor<100x100xi32>, %arg28: tensor<100x100xi32>, %arg29: tensor<100x100xi32>, %arg30: tensor<100x100xi32>, %arg31: tensor<100x100xi32>, %arg32: tensor<100x100xi32>, %arg33: tensor<100x100xi32>, %arg34: tensor<100x100xi32>, %arg35: tensor<100x100xi32>, %arg36: tensor<100x100xi32>, %arg37: tensor<100x100xi32>, %arg38: tensor<100x100xi32>, %arg39: tensor<100x100xi32>, %arg40: tensor<100x100xi32>) -> tensor<100x100xi32> {
    %0 = tensor.empty() : tensor<100x100xi32>
    %1 = linalg.matmul ins(%arg0, %arg1 : tensor<100x100xi32>, tensor<100x100xi32>) outs(%0 : tensor<100x100xi32>) -> tensor<100x100xi32>
    %2 = tensor.empty() : tensor<100x100xi32>
    %3 = linalg.matmul ins(%1, %arg2 : tensor<100x100xi32>, tensor<100x100xi32>) outs(%2 : tensor<100x100xi32>) -> tensor<100x100xi32>
    %4 = tensor.empty() : tensor<100x100xi32>
    %5 = linalg.matmul ins(%3, %arg3 : tensor<100x100xi32>, tensor<100x100xi32>) outs(%4 : tensor<100x100xi32>) -> tensor<100x100xi32>
    %6 = tensor.empty() : tensor<100x100xi32>
    %7 = linalg.matmul ins(%5, %arg4 : tensor<100x100xi32>, tensor<100x100xi32>) outs(%6 : tensor<100x100xi32>) -> tensor<100x100xi32>
    %8 = tensor.empty() : tensor<100x100xi32>
    %9 = linalg.matmul ins(%7, %arg5 : tensor<100x100xi32>, tensor<100x100xi32>) outs(%8 : tensor<100x100xi32>) -> tensor<100x100xi32>
    %10 = tensor.empty() : tensor<100x100xi32>
    %11 = linalg.matmul ins(%9, %arg6 : tensor<100x100xi32>, tensor<100x100xi32>) outs(%10 : tensor<100x100xi32>) -> tensor<100x100xi32>
    %12 = tensor.empty() : tensor<100x100xi32>
    %13 = linalg.matmul ins(%11, %arg7 : tensor<100x100xi32>, tensor<100x100xi32>) outs(%12 : tensor<100x100xi32>) -> tensor<100x100xi32>
    %14 = tensor.empty() : tensor<100x100xi32>
    %15 = linalg.matmul ins(%13, %arg8 : tensor<100x100xi32>, tensor<100x100xi32>) outs(%14 : tensor<100x100xi32>) -> tensor<100x100xi32>
    %16 = tensor.empty() : tensor<100x100xi32>
    %17 = linalg.matmul ins(%15, %arg9 : tensor<100x100xi32>, tensor<100x100xi32>) outs(%16 : tensor<100x100xi32>) -> tensor<100x100xi32>
    %18 = tensor.empty() : tensor<100x100xi32>
    %19 = linalg.matmul ins(%17, %arg10 : tensor<100x100xi32>, tensor<100x100xi32>) outs(%18 : tensor<100x100xi32>) -> tensor<100x100xi32>
    %20 = tensor.empty() : tensor<100x100xi32>
    %21 = linalg.matmul ins(%19, %arg11 : tensor<100x100xi32>, tensor<100x100xi32>) outs(%20 : tensor<100x100xi32>) -> tensor<100x100xi32>
    %22 = tensor.empty() : tensor<100x100xi32>
    %23 = linalg.matmul ins(%21, %arg12 : tensor<100x100xi32>, tensor<100x100xi32>) outs(%22 : tensor<100x100xi32>) -> tensor<100x100xi32>
    %24 = tensor.empty() : tensor<100x100xi32>
    %25 = linalg.matmul ins(%23, %arg13 : tensor<100x100xi32>, tensor<100x100xi32>) outs(%24 : tensor<100x100xi32>) -> tensor<100x100xi32>
    %26 = tensor.empty() : tensor<100x100xi32>
    %27 = linalg.matmul ins(%25, %arg14 : tensor<100x100xi32>, tensor<100x100xi32>) outs(%26 : tensor<100x100xi32>) -> tensor<100x100xi32>
    %28 = tensor.empty() : tensor<100x100xi32>
    %29 = linalg.matmul ins(%27, %arg15 : tensor<100x100xi32>, tensor<100x100xi32>) outs(%28 : tensor<100x100xi32>) -> tensor<100x100xi32>
    %30 = tensor.empty() : tensor<100x100xi32>
    %31 = linalg.matmul ins(%29, %arg16 : tensor<100x100xi32>, tensor<100x100xi32>) outs(%30 : tensor<100x100xi32>) -> tensor<100x100xi32>
    %32 = tensor.empty() : tensor<100x100xi32>
    %33 = linalg.matmul ins(%31, %arg17 : tensor<100x100xi32>, tensor<100x100xi32>) outs(%32 : tensor<100x100xi32>) -> tensor<100x100xi32>
    %34 = tensor.empty() : tensor<100x100xi32>
    %35 = linalg.matmul ins(%33, %arg18 : tensor<100x100xi32>, tensor<100x100xi32>) outs(%34 : tensor<100x100xi32>) -> tensor<100x100xi32>
    %36 = tensor.empty() : tensor<100x100xi32>
    %37 = linalg.matmul ins(%35, %arg19 : tensor<100x100xi32>, tensor<100x100xi32>) outs(%36 : tensor<100x100xi32>) -> tensor<100x100xi32>
    %38 = tensor.empty() : tensor<100x100xi32>
    %39 = linalg.matmul ins(%37, %arg20 : tensor<100x100xi32>, tensor<100x100xi32>) outs(%38 : tensor<100x100xi32>) -> tensor<100x100xi32>
    %40 = tensor.empty() : tensor<100x100xi32>
    %41 = linalg.matmul ins(%39, %arg21 : tensor<100x100xi32>, tensor<100x100xi32>) outs(%40 : tensor<100x100xi32>) -> tensor<100x100xi32>
    %42 = tensor.empty() : tensor<100x100xi32>
    %43 = linalg.matmul ins(%41, %arg22 : tensor<100x100xi32>, tensor<100x100xi32>) outs(%42 : tensor<100x100xi32>) -> tensor<100x100xi32>
    %44 = tensor.empty() : tensor<100x100xi32>
    %45 = linalg.matmul ins(%43, %arg23 : tensor<100x100xi32>, tensor<100x100xi32>) outs(%44 : tensor<100x100xi32>) -> tensor<100x100xi32>
    %46 = tensor.empty() : tensor<100x100xi32>
    %47 = linalg.matmul ins(%45, %arg24 : tensor<100x100xi32>, tensor<100x100xi32>) outs(%46 : tensor<100x100xi32>) -> tensor<100x100xi32>
    %48 = tensor.empty() : tensor<100x100xi32>
    %49 = linalg.matmul ins(%47, %arg25 : tensor<100x100xi32>, tensor<100x100xi32>) outs(%48 : tensor<100x100xi32>) -> tensor<100x100xi32>
    %50 = tensor.empty() : tensor<100x100xi32>
    %51 = linalg.matmul ins(%49, %arg26 : tensor<100x100xi32>, tensor<100x100xi32>) outs(%50 : tensor<100x100xi32>) -> tensor<100x100xi32>
    %52 = tensor.empty() : tensor<100x100xi32>
    %53 = linalg.matmul ins(%51, %arg27 : tensor<100x100xi32>, tensor<100x100xi32>) outs(%52 : tensor<100x100xi32>) -> tensor<100x100xi32>
    %54 = tensor.empty() : tensor<100x100xi32>
    %55 = linalg.matmul ins(%53, %arg28 : tensor<100x100xi32>, tensor<100x100xi32>) outs(%54 : tensor<100x100xi32>) -> tensor<100x100xi32>
    %56 = tensor.empty() : tensor<100x100xi32>
    %57 = linalg.matmul ins(%55, %arg29 : tensor<100x100xi32>, tensor<100x100xi32>) outs(%56 : tensor<100x100xi32>) -> tensor<100x100xi32>
    %58 = tensor.empty() : tensor<100x100xi32>
    %59 = linalg.matmul ins(%57, %arg30 : tensor<100x100xi32>, tensor<100x100xi32>) outs(%58 : tensor<100x100xi32>) -> tensor<100x100xi32>
    %60 = tensor.empty() : tensor<100x100xi32>
    %61 = linalg.matmul ins(%59, %arg31 : tensor<100x100xi32>, tensor<100x100xi32>) outs(%60 : tensor<100x100xi32>) -> tensor<100x100xi32>
    %62 = tensor.empty() : tensor<100x100xi32>
    %63 = linalg.matmul ins(%61, %arg32 : tensor<100x100xi32>, tensor<100x100xi32>) outs(%62 : tensor<100x100xi32>) -> tensor<100x100xi32>
    %64 = tensor.empty() : tensor<100x100xi32>
    %65 = linalg.matmul ins(%63, %arg33 : tensor<100x100xi32>, tensor<100x100xi32>) outs(%64 : tensor<100x100xi32>) -> tensor<100x100xi32>
    %66 = tensor.empty() : tensor<100x100xi32>
    %67 = linalg.matmul ins(%65, %arg34 : tensor<100x100xi32>, tensor<100x100xi32>) outs(%66 : tensor<100x100xi32>) -> tensor<100x100xi32>
    %68 = tensor.empty() : tensor<100x100xi32>
    %69 = linalg.matmul ins(%67, %arg35 : tensor<100x100xi32>, tensor<100x100xi32>) outs(%68 : tensor<100x100xi32>) -> tensor<100x100xi32>
    %70 = tensor.empty() : tensor<100x100xi32>
    %71 = linalg.matmul ins(%69, %arg36 : tensor<100x100xi32>, tensor<100x100xi32>) outs(%70 : tensor<100x100xi32>) -> tensor<100x100xi32>
    %72 = tensor.empty() : tensor<100x100xi32>
    %73 = linalg.matmul ins(%71, %arg37 : tensor<100x100xi32>, tensor<100x100xi32>) outs(%72 : tensor<100x100xi32>) -> tensor<100x100xi32>
    %74 = tensor.empty() : tensor<100x100xi32>
    %75 = linalg.matmul ins(%73, %arg38 : tensor<100x100xi32>, tensor<100x100xi32>) outs(%74 : tensor<100x100xi32>) -> tensor<100x100xi32>
    %76 = tensor.empty() : tensor<100x100xi32>
    %77 = linalg.matmul ins(%75, %arg39 : tensor<100x100xi32>, tensor<100x100xi32>) outs(%76 : tensor<100x100xi32>) -> tensor<100x100xi32>
    %78 = tensor.empty() : tensor<100x100xi32>
    %79 = linalg.matmul ins(%77, %arg40 : tensor<100x100xi32>, tensor<100x100xi32>) outs(%78 : tensor<100x100xi32>) -> tensor<100x100xi32>
    return %79 : tensor<100x100xi32>
  }
}