#include <iostream>

#include "mlir/IR/Types.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

/**
 * Builds a function that performs n matrix multiplications.
 * n = 2: (XY)Z
 * n = 3: ((XY)Z)W
 * n = 4: (((XY)Z)W)V
 */
mlir::func::FuncOp buildNMMFunction(mlir::OpBuilder builder, uint8_t n) {
    int64_t a = 100, b = 100;
    mlir::RankedTensorType tensorType = mlir::RankedTensorType::get({a, b}, builder.getI32Type());
    
    // create function type
    std::vector<mlir::Type> inputTypes;
    for (uint8_t i = 0; i < n + 1; i++) {
        inputTypes.push_back(tensorType);
    }

    // Create the function inside the module.
    std::string funcName = "_" + std::to_string(n) + "mm";
    mlir::func::FuncOp func = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), funcName, builder.getFunctionType(inputTypes, {tensorType}));
    
    mlir::Block* entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock); // Set the insertion point to the function's entry block.

    mlir::BlockArgument x = entryBlock->getArgument(0);
    mlir::BlockArgument y = entryBlock->getArgument(1);

    mlir::Value xy_init = builder.create<mlir::tensor::EmptyOp>(builder.getUnknownLoc(), tensorType, mlir::ValueRange{});
    mlir::linalg::MatmulOp xy = builder.create<mlir::linalg::MatmulOp>(builder.getUnknownLoc(), tensorType, mlir::ValueRange {x, y}, xy_init);

    mlir::linalg::MatmulOp mult = xy;
    for (uint8_t i = 0; i < n - 1; i++) {
        mlir::BlockArgument z = entryBlock->getArgument(i + 2);
        mlir::Value xy_z_init = builder.create<mlir::tensor::EmptyOp>(builder.getUnknownLoc(), tensorType, mlir::ValueRange{});
        mlir::linalg::MatmulOp xy_z = builder.create<mlir::linalg::MatmulOp>(builder.getUnknownLoc(), tensorType, mlir::ValueRange {mult.getResult(0), z}, xy_z_init);

        mult = xy_z;
    }

    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), mult.getResult(0));

    return func;
}

int main() {
    int n; // Number of matrix multiplications.
    std::cin >> n;
    llvm::outs() << n << "MM \n";

    mlir::MLIRContext context; // Initialize an MLIR context.
    context.getOrLoadDialect<mlir::linalg::LinalgDialect>();
    context.getOrLoadDialect<mlir::tensor::TensorDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();

    // Create a module to hold the function.
    mlir::OpBuilder builder(&context);
    auto module = mlir::ModuleOp::create(builder.getUnknownLoc());

    auto NMMFunction = buildNMMFunction(builder, n);
    module.push_back(NMMFunction);

    // Open the file to write the MLIR code to.
    std::error_code EC;
    std::string file = "test/nmm/" + std::to_string(n) + "sqmm.mlir";
    llvm::raw_fd_ostream outputFile(file, EC);

    if (EC) {
        llvm::errs() << "Error opening file: " << EC.message() << "\n";
        return 1;
    }

    // Print the MLIR module to the file.
    module.print(outputFile);

    // Verify the module.
    if (failed(verify(module))) {
        llvm::errs() << "Module verification failed.\n";
        return 1;
    }

    return 0;
}
