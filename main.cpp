#include <iostream>
#include <circt/Conversion/ExportVerilog.h>
#include <circt/Dialect/Comb/CombDialect.h>
#include <circt/Dialect/Comb/CombOps.h>
#include <circt/Dialect/HW/HWDialect.h>
#include <circt/Dialect/HW/HWInstanceGraph.h>
#include <circt/Dialect/HW/HWOps.h>
#include <mlir/IR/ImplicitLocOpBuilder.h>
#include <mlir/IR/Verifier.h>
#include <llvm/ADT/PostOrderIterator.h>
#include <llvm/Support/raw_ostream.h>

using namespace mlir;
using namespace circt;
using namespace comb;
using namespace hw;

int main()
{
  MLIRContext context;
  context.loadDialect<CombDialect>();
  context.loadDialect<HWDialect>();
  LocationAttr loc = UnknownLoc::get(&context);
  ModuleOp module = ModuleOp::create(loc);
  ImplicitLocOpBuilder builder = ImplicitLocOpBuilder::atBlockEnd(loc, module.getBody());
  IntegerType wire32 = builder.getIntegerType(32);

  // Module ports (inputs and outputs)
  SmallVector<PortInfo> ports;
  ports.push_back(PortInfo{builder.getStringAttr("a"), PortDirection::INPUT, wire32, 0});
  ports.push_back(PortInfo{builder.getStringAttr("b"), PortDirection::INPUT, wire32, 1});
  ports.push_back(PortInfo{builder.getStringAttr("out"), PortDirection::OUTPUT, wire32, 0});

  // Create module top
  HWModuleOp top = builder.create<HWModuleOp>(builder.getStringAttr("Top"), ports);
  builder.setInsertionPointToStart(top.getBodyBlock());

  // Constants
  ConstantOp c0 = builder.create<ConstantOp>(wire32, 42);
  ConstantOp c1 = builder.create<ConstantOp>(wire32, 0x11223344);

  // Module implementation
  auto tmp1 = builder.create<comb::AndOp>(top.getArgument(0), c0);
  auto tmp2 = builder.create<comb::OrOp>(top.getArgument(1), c1);
  auto tmp3 = builder.create<comb::XorOp>(tmp1, tmp2);

  // Module output
  auto outputOp = top.getBodyBlock()->getTerminator();
  outputOp->setOperands(ValueRange{tmp3});

  std::cout << "Module inputs: " << top.getNumInputs() << ", outputs: " << top.getNumOutputs() << std::endl;
  std::cout << "Module verify: " << mlir::verify(module).succeeded() << std::endl;
  module.dump();

  std::string str;
  llvm::raw_string_ostream os(str);
  std::cout << "Export Verilog: " << exportVerilog(module, os).succeeded() << std::endl;
  std::cout << str << std::endl;

  std::cout << "Done!" << std::endl;
}
