func.func @constants() -> i32 {
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32
    %sum = arith.addi %c2, %c3 : i32

    func.return %sum : i32
}