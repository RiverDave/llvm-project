add_clang_library(MLIRCIROffloadOpt
  KernelBindingTable.cpp

  DEPENDS
  MLIRCIROpsIncGen

  LINK_LIBS PUBLIC
  MLIRCIR
  MLIRIR
  MLIRSupport
  )
