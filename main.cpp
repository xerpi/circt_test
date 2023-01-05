#include <iostream>
#include <circt/Conversion/ExportVerilog.h>
#include <circt/Conversion/HWArithToHW.h>
#include <circt/Dialect/Comb/CombDialect.h>
#include <circt/Dialect/Comb/CombOps.h>
#include <circt/Dialect/HW/HWDialect.h>
#include <circt/Dialect/HW/HWOps.h>
#include <circt/Dialect/HWArith/HWArithDialect.h>
#include <circt/Dialect/HWArith/HWArithOps.h>
#include <circt/Dialect/SV/SVDialect.h>
#include <llvm/ADT/PostOrderIterator.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/ImplicitLocOpBuilder.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/PassManager.h>

using namespace mlir;
using namespace circt;

int main()
{
  MLIRContext context;
  context.loadDialect<comb::CombDialect>();
  context.loadDialect<hw::HWDialect>();
  context.loadDialect<hwarith::HWArithDialect>();
  context.loadDialect<sv::SVDialect>();
  LocationAttr loc = UnknownLoc::get(&context);
  ModuleOp module = ModuleOp::create(loc, {});
  ImplicitLocOpBuilder builder = ImplicitLocOpBuilder::atBlockEnd(loc, module.getBody());
  IntegerType wirei32 = builder.getIntegerType(32);
  IntegerType wiresi32 = builder.getIntegerType(32, true);
  IntegerType wiresi64 = builder.getIntegerType(64, true);

  // Module ports (inputs and outputs)
  SmallVector<hw::PortInfo> ports;
  ports.push_back(hw::PortInfo{builder.getStringAttr("a"), hw::PortDirection::INPUT, wirei32, 0});
  ports.push_back(hw::PortInfo{builder.getStringAttr("b"), hw::PortDirection::INPUT, wirei32, 1});
  ports.push_back(hw::PortInfo{builder.getStringAttr("c"), hw::PortDirection::INPUT, wirei32, 2});
  ports.push_back(hw::PortInfo{builder.getStringAttr("out"), hw::PortDirection::OUTPUT, wiresi64, 0});

  // Create module top
  hw::HWModuleOp top = builder.create<hw::HWModuleOp>(builder.getStringAttr("Top"), ports);
  builder.setInsertionPointToStart(top.getBodyBlock());

  // Constants
  hw::ConstantOp c0 = builder.create<hw::ConstantOp>(wirei32, 42);
  hw::ConstantOp c1 = builder.create<hw::ConstantOp>(wirei32, 0x11223344);

  // Module implementation
  auto tmp1 = builder.create<comb::AndOp>(top.getArgument(0), c0);
  auto tmp2 = builder.create<comb::OrOp>(top.getArgument(1), c1);
  auto tmp3 = builder.create<comb::XorOp>(top.getArgument(2), tmp1);
  auto tmp2_cast = builder.create<hwarith::CastOp>(wiresi32, top.getArgument(0));
  auto tmp3_cast = builder.create<hwarith::CastOp>(wiresi32, top.getArgument(1));
  auto tmp4 = builder.create<hwarith::MulOp>(ValueRange{tmp2_cast, tmp3_cast});

  // Module output
  auto outputOp = top.getBodyBlock()->getTerminator();
  outputOp->setOperands(ValueRange{tmp4});

  // Print MLIR before running passes
  std::cout << "Original MLIR" << std::endl;
  module.dump();

  // Create and run passes
  PassManager pm(module.getContext());
  pm.addPass(circt::createHWArithToHWPass());
  auto pmRunResult = pm.run(module);

  std::cout << "Run passes result: " << pmRunResult.succeeded() << std::endl;
  std::cout << "Module inputs: " << top.getNumInputs() << ", outputs: " << top.getNumOutputs() << std::endl;

  // Verify module
  auto moduleVerifyResult = mlir::verify(module);
  std::cout << "Module verify result: " << moduleVerifyResult.succeeded() << std::endl;

  // Print MLIR after running passes
  std::cout << "MLIR after running passes" << std::endl;
  module.dump();

  // Exmit Verilog
  std::string str;
  llvm::raw_string_ostream os(str);
  auto exportVerilogResult = exportVerilog(module, os);
  std::cout << "Export Verilog result: " << exportVerilogResult.succeeded() << std::endl;
  std::cout << str << std::endl;

  std::cout << "Done!" << std::endl;
}
