#include <iostream>
#include <circt/Conversion/ExportVerilog.h>
#include <circt/Dialect/Comb/CombDialect.h>
#include <circt/Dialect/Comb/CombOps.h>
#include <circt/Dialect/HW/HWDialect.h>
#include <circt/Dialect/HW/HWInstanceGraph.h>
#include <circt/Dialect/HW/HWOps.h>
#include <circt/Dialect/SV/SVDialect.h>
#include <mlir/IR/ImplicitLocOpBuilder.h>
#include <mlir/IR/Verifier.h>
#include <llvm/ADT/PostOrderIterator.h>
#include <llvm/Support/raw_ostream.h>

using namespace mlir;
using namespace circt;

int main()
{
  MLIRContext context;
  context.loadDialect<comb::CombDialect>();
  context.loadDialect<hw::HWDialect>();
  context.loadDialect<sv::SVDialect>();
  LocationAttr loc = UnknownLoc::get(&context);
  ModuleOp module = ModuleOp::create(loc);
  ImplicitLocOpBuilder builder = ImplicitLocOpBuilder::atBlockEnd(loc, module.getBody());
  IntegerType wire32 = builder.getIntegerType(32);

  // Module ports (inputs and outputs)
  SmallVector<hw::PortInfo> ports;
  ports.push_back(hw::PortInfo{builder.getStringAttr("a"), hw::PortDirection::INPUT, wire32, 0});
  ports.push_back(hw::PortInfo{builder.getStringAttr("b"), hw::PortDirection::INPUT, wire32, 1});
  ports.push_back(hw::PortInfo{builder.getStringAttr("c"), hw::PortDirection::INPUT, wire32, 2});
  ports.push_back(hw::PortInfo{builder.getStringAttr("out"), hw::PortDirection::OUTPUT, wire32, 0});

  // Create module top
  hw::HWModuleOp top = builder.create<hw::HWModuleOp>(builder.getStringAttr("Top"), ports);
  builder.setInsertionPointToStart(top.getBodyBlock());

  // Constants
  hw::ConstantOp c0 = builder.create<hw::ConstantOp>(wire32, 42);
  hw::ConstantOp c1 = builder.create<hw::ConstantOp>(wire32, 0x11223344);

  // Module implementation
  auto tmp1 = builder.create<comb::AndOp>(top.getArgument(0), c0);
  auto tmp2 = builder.create<comb::OrOp>(top.getArgument(1), c1);
  auto tmp3 = builder.create<comb::XorOp>(top.getArgument(2), tmp1);
  auto tmp4 = builder.create<comb::MulOp>(tmp2, tmp3);

  // Module output
  auto outputOp = top.getBodyBlock()->getTerminator();
  outputOp->setOperands(ValueRange{tmp4});

  std::cout << "Module inputs: " << top.getNumInputs() << ", outputs: " << top.getNumOutputs() << std::endl;
  std::cout << "Module verify: " << mlir::verify(module).succeeded() << std::endl;
  module.dump();

  std::string str;
  llvm::raw_string_ostream os(str);
  std::cout << "Export Verilog: " << exportVerilog(module, os).succeeded() << std::endl;
  std::cout << str << std::endl;

  std::cout << "Done!" << std::endl;
}
