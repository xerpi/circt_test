cmake_minimum_required(VERSION 3.14)

project(circt_test)

find_package(CIRCT REQUIRED)

add_executable(circt_test
  main.cpp
)

target_compile_features(circt_test
  PUBLIC
  cxx_std_17
)

target_include_directories(circt_test
  PRIVATE
  ${CIRCT_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
)

target_link_libraries(circt_test
  PRIVATE
  CIRCTComb
  CIRCTExportVerilog
  CIRCTHW
  CIRCTHWArith
  CIRCTHWArithToHW
  CIRCTSV
)
