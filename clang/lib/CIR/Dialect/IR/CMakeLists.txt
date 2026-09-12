add_clang_library(MLIRCIR
  CIRAttrs.cpp
  CIRDialect.cpp
  CIRInlinerInterface.cpp
  CIRMemorySlot.cpp
  CIRTypes.cpp
  CIRDataLayout.cpp
  CIROpenCLAttrs.cpp

  DEPENDS
  MLIRCIROpsIncGen
  MLIRCIRTypeConstraintsIncGen
  MLIRCIREnumsGen
  MLIRCIROpInterfacesIncGen
  MLIRCIRLoopOpInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRCIRInterfaces
  MLIRDLTIDialect
  MLIRDataLayoutInterfaces
  MLIRFuncDialect
  MLIRFuncInlinerExtension
  MLIRLLVMIRTransforms
  MLIRLoopLikeInterface
  MLIRCIRInterfaces
  clangAST
  )
