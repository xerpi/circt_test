#include <iostream>
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

#define EXPECT_EQ(a, b) \
  do { \
    if ((a) != (b)) \
      std::cerr << "Expect equal failed" << std::endl; \
  } while (0)

#define ASSERT_EQ(a, b) \
  assert((a) == (b))

int main()
{
  MLIRContext context;
  context.loadDialect<CombDialect>();
  context.loadDialect<HWDialect>();
  LocationAttr loc = UnknownLoc::get(&context);
  ModuleOp module = ModuleOp::create(loc);
  ImplicitLocOpBuilder builder = ImplicitLocOpBuilder::atBlockEnd(loc, module.getBody());
  IntegerType wireTy = builder.getIntegerType(3);

  // Module ports (inputs and outputs)
  SmallVector<PortInfo> ports;
  ports.push_back(PortInfo{builder.getStringAttr("a"), PortDirection::INPUT, wireTy, 0});
  ports.push_back(PortInfo{builder.getStringAttr("b"), PortDirection::INPUT, wireTy, 1});
  ports.push_back(PortInfo{builder.getStringAttr("out"), PortDirection::OUTPUT, wireTy, 0});

  HWModuleOp top = builder.create<HWModuleOp>(builder.getStringAttr("Top"), ports);
  builder.setInsertionPointToStart(top.getBodyBlock());

  //builder.create<comb::AndOp>(top.getArgument(0), top.getArgument(0), c0);

  /*inputs.emplace_back(builder.getStringAttr("a"), wireA);
  inputs.emplace_back(builder.getStringAttr("b"), wireB);
  inputs.emplace_back(builder.getStringAttr("c"), wireC);
  inputs.emplace_back(builder.getStringAttr("d"), wireD);*/
  //top.insertPorts(inputs, outputs);

  // Constants
  ConstantOp c0 = builder.create<ConstantOp>(wireTy, 0);
  //ConstantOp c1 = builder.create<ConstantOp>(wireTy, 1);

  auto foo = builder.create<comb::AndOp>(top.getArgument(0), c0);
  auto foo2 = builder.create<comb::OrOp>(foo, top.getArgument(1));

  //builder.create<hw::OutputOp>();
  auto output = cast<OutputOp>(top.getBodyBlock()->getTerminator());
  output->insertOperands(0, ValueRange{foo2});
  //builder.create<comb::AndOp>(output.getOutputs(), c1);

  //builder.create<hw::OutputOp>(ValueRange{foo2});

  std::cout << "ins: " << top.getNumInputs() << ", outs: " << top.getNumOutputs() << std::endl;
  std::cout << "results types: " << top.getResultTypes().size() << std::endl;

#if 0
  auto ports = top.getAllPorts();
  ASSERT_EQ(ports.size(), 4u);

  EXPECT_EQ(ports[0].name, builder.getStringAttr("a"));
  EXPECT_EQ(ports[0].direction, PortDirection::OUTPUT);
  EXPECT_EQ(ports[0].type, wireTy);

  EXPECT_EQ(ports[1].name, builder.getStringAttr("b"));
  EXPECT_EQ(ports[1].direction, PortDirection::OUTPUT);
  EXPECT_EQ(ports[1].type, wireTy);

  EXPECT_EQ(ports[2].name, builder.getStringAttr("c"));
  EXPECT_EQ(ports[2].direction, PortDirection::OUTPUT);
  EXPECT_EQ(ports[2].type, wireTy);

  EXPECT_EQ(ports[3].name, builder.getStringAttr("d"));
  EXPECT_EQ(ports[3].direction, PortDirection::OUTPUT);
  EXPECT_EQ(ports[3].type, wireTy);

  auto output = cast<OutputOp>(top.getBodyBlock()->getTerminator());
  ASSERT_EQ(output->getNumOperands(), 4u);

  EXPECT_EQ(output->getOperand(0), wireA.getResult());
  EXPECT_EQ(output->getOperand(1), wireB.getResult());
  EXPECT_EQ(output->getOperand(2), wireC.getResult());
  EXPECT_EQ(output->getOperand(3), wireD.getResult());
#endif

  std::cout << "verify: " << mlir::verify(module).succeeded() << std::endl;
  module.dump();

  std::cout << "Done!" << std::endl;
}
