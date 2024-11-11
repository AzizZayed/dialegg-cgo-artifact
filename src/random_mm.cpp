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

uint8_t randomInt(uint8_t min, uint8_t max) {
    return min + (rand() % (max - min + 1));
}

int64_t getRows(mlir::Type tensorType) {
    mlir::RankedTensorType rankedTensorType = tensorType.cast<mlir::RankedTensorType>();
    return rankedTensorType.getShape()[0];
}

int64_t getCols(mlir::Type tensorType) {
    mlir::RankedTensorType rankedTensorType = tensorType.cast<mlir::RankedTensorType>();
    return rankedTensorType.getShape()[1];
}

/**
 * Builds a function that performs n matrix multiplications with matrices of random dimensions.
 * n = 2: (XY)Z
 * n = 3: ((XY)Z)W
 * n = 4: (((XY)Z)W)V
 */
mlir::func::FuncOp buildNMMFunction(mlir::OpBuilder builder, uint8_t n) {
    
    // create function type
    uint8_t a = randomInt(10, 100);
    uint8_t b = randomInt(10, 100);

    uint8_t first = a, last = a;
    
    std::vector<mlir::Type> inputTypes;
    for (uint8_t i = 0; i < n + 1; i++) {
        inputTypes.push_back(mlir::RankedTensorType::get({a, b}, builder.getI32Type()));

        a = b;
        last = b;
        
        b = randomInt(10, 100);
    }
    mlir::RankedTensorType outputType = mlir::RankedTensorType::get({first, last}, builder.getI32Type());

    // Create the function inside the module.
    std::string funcName = "_" + std::to_string(n) + "mm";
    mlir::func::FuncOp func = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), funcName, builder.getFunctionType(inputTypes, {outputType}));
    
    mlir::Block* entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock); // Set the insertion point to the function's entry block.

    mlir::BlockArgument x = entryBlock->getArgument(0);
    mlir::BlockArgument y = entryBlock->getArgument(1);

    mlir::Type xy_type = mlir::RankedTensorType::get({getRows(x.getType()), getCols(y.getType())}, builder.getI32Type());
    mlir::Value xy_init = builder.create<mlir::tensor::EmptyOp>(builder.getUnknownLoc(), xy_type, mlir::ValueRange{});
    mlir::linalg::MatmulOp xy = builder.create<mlir::linalg::MatmulOp>(builder.getUnknownLoc(), xy_type, mlir::ValueRange {x, y}, xy_init);

    mlir::linalg::MatmulOp mult = xy;
    for (uint8_t i = 0; i < n - 1; i++) {
        mlir::BlockArgument z = entryBlock->getArgument(i + 2);
        mlir::Type xy_z_type = mlir::RankedTensorType::get({getRows(mult.getResult(0).getType()), getCols(z.getType())}, builder.getI32Type());
        mlir::Value xy_z_init = builder.create<mlir::tensor::EmptyOp>(builder.getUnknownLoc(), xy_z_type, mlir::ValueRange{});
        mlir::linalg::MatmulOp xy_z = builder.create<mlir::linalg::MatmulOp>(builder.getUnknownLoc(), xy_z_type, mlir::ValueRange {mult.getResult(0), z}, xy_z_init);

        mult = xy_z;
    }

    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), mult.getResult(0));

    return func;
}

int main() {
    int n; // Number of matrix multiplications.
    std::cin >> n;
    llvm::outs() << n << "MM \n";

    srand(0);
    
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
    std::string file = "test/nmm/" + std::to_string(n) + "mm.mlir";
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
