func.func @classic(%a: i32) -> i32 {
  %c2 = arith.constant 2 : i32
  %mul = arith.muli %a, %c2 : i32
  %div = arith.divsi %mul, %c2 : i32
  
  func.return %div : i32
}