add_clang_library(MLIRCIROffloadOpt
  KernelBindingTable.cpp
  DeadKernelElimination.cpp
  KernelArgConstantPropagation.cpp

  DEPENDS
  MLIRCIROpsIncGen
  MLIRCIRPassIncGen

  LINK_LIBS PUBLIC
  MLIRCIR
  MLIRIR
  MLIRPass
  MLIRSupport
  )
