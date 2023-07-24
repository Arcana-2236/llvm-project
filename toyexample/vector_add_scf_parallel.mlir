module attributes {torch.debug_module_name = "Vector_Add"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: memref<1024xf64>, %arg1: memref<1024xf64>) -> memref<1024xf64> {
    %c0 = arith.constant 0 : index
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %alloc = memref.alloc() : memref<1024xf64>
    memref.dealloc %alloc : memref<1024xf64>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1024xf64>
    scf.parallel (%arg2) = (%c0) to (%c1024) step (%c1) {
      %0 = memref.load %arg0[%arg2] : memref<1024xf64>
      %1 = memref.load %arg1[%arg2] : memref<1024xf64>
      %2 = arith.addf %0, %1 : f64
      memref.store %2, %alloc_0[%arg2] : memref<1024xf64>
      scf.yield
    } 
    // { mapping = [#gpu.loop_dim_map<processor = sequential, map = (d0) -> (d0), bound = (d0) -> (d0)>, #gpu.loop_dim_map<processor = block_x, map = (d0) -> (d0), bound = (d0) -> (d0)>] }
    return %alloc_0 : memref<1024xf64>
  }
}
