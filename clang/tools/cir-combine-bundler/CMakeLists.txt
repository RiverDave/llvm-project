get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

include_directories(${LLVM_MAIN_SRC_DIR}/../mlir/include)
include_directories(${CMAKE_BINARY_DIR}/tools/mlir/include)

add_clang_tool(cir-combine-bundler
  cir-combine-bundler.cpp
)

clang_target_link_libraries(cir-combine-bundler
  PRIVATE
  clangBasic
  clangDriver
  CIROpenMPSupport
  MLIRCIR
)

target_link_libraries(cir-combine-bundler
  PRIVATE
  ${dialect_libs}
  MLIRIR
  MLIRParser
  MLIRSupport
)
