gpu.module @forward_kernel {
    gpu.func @forward_kernel(%arg0: index, %arg1: index, %arg2: memref<1024xf64>, %arg3: memref<1024xf64>, %arg4: memref<1024xf64>, %arg5: index, %arg6: index) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>} {
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %2 = gpu.block_id  z
      %3 = gpu.thread_id  x
      %4 = gpu.thread_id  y
      %5 = gpu.thread_id  z
      %6 = gpu.grid_dim  x
      %7 = gpu.grid_dim  y
      %8 = gpu.grid_dim  z
      %9 = gpu.block_dim  x
      %10 = gpu.block_dim  y
      %11 = gpu.block_dim  z
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      // %12 = affine.apply #map1(%1)[%arg0, %arg1]
      %12 = emitc.call "map1"(%1) : (index) -> index
      scf.for %arg7 = %arg5 to %arg0 step %arg6 {
        // %13 = arith.addi %arg7, %12 : index
        %13 = emitc.call "addi"(%arg7, %12) : (index, index) -> index
        %14 = memref.load %arg2[%13] : memref<1024xf64>
        %15 = memref.load %arg3[%13] : memref<1024xf64>
        %16 = arith.addf %14, %15 : f64
        memref.store %16, %arg4[%13] : memref<1024xf64>
      }
      gpu.return
    }
  }
