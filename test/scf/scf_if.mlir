func.func @conditional_sqrt(%x: f32) -> f32 {
    %zero = arith.constant 0.0 : f32
    %cond = arith.cmpf oge, %x, %zero : f32
    %sqrt = scf.if %cond -> (f32) {
        %sqrt = math.sqrt %x : f32
        scf.yield %sqrt : f32
    } else {
        %neg = arith.negf %x : f32
        %sqrt = math.sqrt %neg : f32
        scf.yield %sqrt : f32
    }
    func.return %sqrt : f32
}
