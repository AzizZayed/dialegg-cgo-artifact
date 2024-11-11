func.func @image_conversion(%image : tensor<3840x2160x3xi64>) -> tensor<3840x2160xi64> { // convert 4K RGB image to grayscale
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index

    %rows = arith.constant 3840 : index
    %cols = arith.constant 2160 : index
    %image_gray_init = tensor.empty() : tensor<3840x2160xi64>

    %image_gray = scf.for %i = %c0 to %rows step %c1 iter_args(%current_image = %image_gray_init) -> (tensor<3840x2160xi64>) {
        %row_result = scf.for %j = %c0 to %cols step %c1 iter_args(%current_row = %current_image) -> (tensor<3840x2160xi64>) {
            %r = tensor.extract %image[%i, %j, %c0] : tensor<3840x2160x3xi64>
            %g = tensor.extract %image[%i, %j, %c1] : tensor<3840x2160x3xi64>
            %b = tensor.extract %image[%i, %j, %c2] : tensor<3840x2160x3xi64>

            %gray = func.call @rgb_to_grayscale(%r, %g, %b) : (i64, i64, i64) -> i64

            %updated_row = tensor.insert %gray into %current_row[%i, %j] : tensor<3840x2160xi64>
            scf.yield %updated_row : tensor<3840x2160xi64>
        }
        scf.yield %row_result : tensor<3840x2160xi64>
    }

    func.return %image_gray : tensor<3840x2160xi64>
}

func.func @rgb_to_grayscale(%r: i64, %g: i64, %b: i64) -> i64 {
	%c256 = arith.constant 256 : i64
	
    // nums from https://www.baeldung.com/cs/convert-rgb-to-grayscale
	%w_r = arith.constant 77 : i64
	%w_g = arith.constant 150 : i64
	%w_b = arith.constant 29 : i64
	
	%r_s = arith.muli %r, %w_r : i64
	%g_s = arith.muli %g, %w_g : i64
	%b_s = arith.muli %b, %w_b : i64

	%r_f = arith.divsi %r_s, %c256 : i64
	%g_f = arith.divsi %g_s, %c256 : i64
	%b_f = arith.divsi %b_s, %c256 : i64

    // change above divisions to bitshifts
    // %c8 = arith.constant 8 : i64
    // %r_f = arith.shrsi %r_s, %c8 : i64
    // %g_f = arith.shrsi %g_s, %c8 : i64
    // %b_f = arith.shrsi %b_s, %c8 : i64
	
	%sum = arith.addi %r_f, %g_f : i64
	%gray = arith.addi %sum, %b_f : i64

	func.return %gray : i64
}