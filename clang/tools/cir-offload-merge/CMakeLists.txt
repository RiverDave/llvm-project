get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

include_directories(${LLVM_MAIN_SRC_DIR}/../mlir/include)
include_directories(${CMAKE_BINARY_DIR}/tools/mlir/include)

add_clang_tool(cir-offload-merge
  cir-offload-merge.cpp
)

clang_target_link_libraries(cir-offload-merge
  PRIVATE
  clangBasic
  clangDriver
  CIROpenMPSupport
  MLIRCIR
  MLIRCIROffloadOpt
)

target_link_libraries(cir-offload-merge
  PRIVATE
  ${dialect_libs}
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
)
