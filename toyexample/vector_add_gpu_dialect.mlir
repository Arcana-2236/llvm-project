module attributes {torch.debug_module_name = "Vector_Add"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: memref<1024xf64>, %arg1: memref<1024xf64>) -> memref<1024xf64> {
    %alloc = memref.alloc() : memref<1024xf64>
    memref.dealloc %alloc : memref<1024xf64>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1024xf64>
    %c0 = arith.constant 0 : index
    %c1024 = arith.constant 1024 : index
    %0 = arith.subi %c1024, %c0 : index
    %c1 = arith.constant 1 : index
    %c1_1 = arith.constant 1 : index
    gpu.launch blocks(%arg2, %arg3, %arg4) in (%arg8 = %0, %arg9 = %c1_1, %arg10 = %c1_1) threads(%arg5, %arg6, %arg7) in (%arg11 = %c1_1, %arg12 = %c1_1, %arg13 = %c1_1) {
      %1 = arith.addi %c0, %arg2 : index
      %2 = memref.load %arg0[%1] : memref<1024xf64>
      %3 = memref.load %arg1[%1] : memref<1024xf64>
      %4 = arith.addf %2, %3 : f64
      memref.store %4, %alloc_0[%1] : memref<1024xf64>
      gpu.terminator
    }
    return %alloc_0 : memref<1024xf64>
  }
}
