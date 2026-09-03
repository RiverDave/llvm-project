add_clang_library(MLIRCIROffloadOpt
  KernelBindingTable.cpp
  KernelCloning.cpp
  DeadKernelElimination.cpp
  KernelArgConstantPropagation.cpp

  DEPENDS
  MLIRCIROpsIncGen
  MLIRCIRPassIncGen

  LINK_LIBS PUBLIC
  MLIRLLVMDialect
  MLIRCIR
  MLIRIR
  MLIRPass
  MLIRSupport
  )
