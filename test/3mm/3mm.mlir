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