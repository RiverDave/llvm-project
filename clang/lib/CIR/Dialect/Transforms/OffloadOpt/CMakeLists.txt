add_clang_library(MLIRCIROffloadOpt
  KernelBindingTable.cpp
  DeadKernelElimination.cpp
  KernelArgConstantPropagation.cpp
  OffloadLaunchNoalias.cpp

  DEPENDS
  MLIRCIROpsIncGen
  MLIRCIRPassIncGen

  LINK_LIBS PUBLIC
  MLIRCIR
  MLIRCIRAnalysis
  MLIRIR
  MLIRPass
  MLIRSupport
  )
