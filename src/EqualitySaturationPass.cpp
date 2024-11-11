#include <map>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <string_view>
#include <thread>
#include <chrono>

#include "llvm/ADT/Hashing.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "Utils.h"
#include "EqualitySaturationPass.h"
#include "Egglog.h"

EqualitySaturationPass::EqualitySaturationPass(const std::string& mlirFile, const std::string& eggFile, const EgglogCustomDefs& funcs)
    : mlirFilePath(mlirFile), eggFilePath(eggFile), customFunctions(funcs) {}

void EqualitySaturationPass::runEgglog(const std::vector<EggifiedOp*>& block, const std::string& blockName) {
    std::ifstream eggFile(eggFilePath);
    std::vector<std::string> egglogLines;

    // Read the egglog file
    std::string opsTarget = ";; OPS HERE ;;";
    std::string extractsTarget = ";; EXTRACTS HERE ;;";

    bool insertedOps = false;
    bool insertedExtracts = false;

    auto start = std::chrono::high_resolution_clock::now();

    std::string line;
    while (std::getline(eggFile, line)) {
        egglogLines.push_back(line);

        if (!insertedOps && line == opsTarget) {
            egglogLines.push_back("; " + blockName);
            for (const EggifiedOp* op: block) {  // Insert the operations
                egglogLines.push_back(op->egglogLet());
            }

            insertedOps = true;
        } else if (!insertedExtracts && line == extractsTarget) {
            for (const EggifiedOp* op: block) {  // Extract the results of the egglog run
                if (op->shouldBeExtracted()) {
                    egglogLines.push_back("(extract " + op->getPrintId() + ")");
                }
            }

            insertedExtracts = true;
        }
    }
    eggFile.close();

    // Write the extracted egglog to a new file with the same name and ext .ops.egg
    std::string opsEggFilePath = eggFilePath.substr(0, eggFilePath.find_last_of(".")) + ".ops.egg";
    std::ofstream eggFileOut(opsEggFilePath);
    for (const std::string& line: egglogLines) {
        eggFileOut << line << "\n";
    }
    eggFileOut.close();
    
    auto end = std::chrono::high_resolution_clock::now();
    mlirToEgglogTime += std::chrono::duration<double>(end - start).count();

    // Run egglog and extract the results
    std::string egglogCmd = "egglog " + opsEggFilePath + " > " + egglogExtractedFilename + " 2> " + egglogLogFilename;

    llvm::outs() << "\nRunning egglog: " << egglogCmd << "\n"
                 << "\n";

    start = std::chrono::high_resolution_clock::now();
    std::system(egglogCmd.c_str());
    end = std::chrono::high_resolution_clock::now();

    egglogExecTime += std::chrono::duration<double>(end - start).count();

    // dump output
    printFileContents(egglogLogFilename);
    printFileContents(egglogExtractedFilename);

    llvm::outs() << "\n\nDone running egglog\n"
                 << "\n";
}

void EqualitySaturationPass::runOnBlock(mlir::Block& block, const std::string& blockName) {
    auto start = std::chrono::high_resolution_clock::now();

    // Eggify the block
    mlir::MLIRContext& context = getContext();
    Egglog egglog(context, customFunctions, supportedOps);

    // register all block arguments
    for (mlir::Value value: block.getArguments()) {
        EggifiedOp* eggifiedValue = egglog.eggifyValue(value);
        // eggifiedValue->print(llvm::outs());
    }

    for (mlir::Operation& op: block.getOperations()) {
        EggifiedOp* eggifiedOp = egglog.eggifyOperation(&op);
        // eggifiedOp->print(llvm::outs());
    }

    auto end = std::chrono::high_resolution_clock::now();
    mlirToEgglogTime += std::chrono::duration<double>(end - start).count();

    // dump all ops
    for (const EggifiedOp* eggOp: egglog.eggifiedBlock) {
        eggOp->print(llvm::outs());
    }

    runEgglog(egglog.eggifiedBlock, blockName); // Run egglog on the block

    start = std::chrono::high_resolution_clock::now();

    std::ifstream file(egglogExtractedFilename);

    // Parse the extracted egglog file and replace the MLIR operations
    for (const EggifiedOp* eggOp: egglog.eggifiedBlock) {
        if (!eggOp->shouldBeExtracted()) {
            continue;
        }

        // eggOp.print(llvm::outs());

        std::string line;
        std::getline(file, line);

        // llvm::outs() << eggOp.getPrintId() << " = " << line << "\n";

        mlir::Operation* prevOp = eggOp->mlirOp;
        mlir::OpBuilder builder(prevOp);
        mlir::Operation* newOp = egglog.parseOperation(line, builder);

        if(newOp == nullptr) { // If the operation is an opaque value, replace it with the value
            mlir::Value value = egglog.parseValue(line);
            prevOp->getResult(0).replaceAllUsesWith(value);
            prevOp->erase();
        } else if (newOp != prevOp) { // Check if the whole operation is different, if so, replace it
            // llvm::outs() << "REPLACING: (" << *prevOp << ") WITH (" << *newOp << ")\n\n";
            prevOp->replaceAllUsesWith(newOp);
            prevOp->erase();
        }

        // llvm::outs() << "\n";
    }

    end = std::chrono::high_resolution_clock::now();

    egglogToMlirTime += std::chrono::duration<double>(end - start).count();

    // dump parsed ops cache
    for (const auto& [opStr, op]: egglog.parsedOps) {
        llvm::outs() << opStr << " : " << *op << "\n";
    }

    file.close();
    llvm::outs() << "\n";
}

void EqualitySaturationPass::init() {
    // Make sure both files exist
    if (!llvm::sys::fs::exists(mlirFilePath)) {
        llvm::errs() << "MLIR file does not exist: " << mlirFilePath << "\n";
        exit(1);
    }
    if (!llvm::sys::fs::exists(eggFilePath)) {
        llvm::errs() << "Egg file does not exist: " << eggFilePath << "\n";
        exit(1);
    }
    
    // mlirFilePath without extension
    std::string name = mlirFilePath.substr(0, mlirFilePath.find(".mlir"));
    egglogExtractedFilename = name + "-egglog-extract.log";
    egglogLogFilename = name + "-egglog.log";
    
    std::ifstream opFile(eggFilePath);
    std::string line;

    while (std::getline(opFile, line)) {
        if (EgglogOpDef::isOpFunction(line)) {
            EgglogOpDef parsedOp = EgglogOpDef::parseOpFunction(line);

            supportedOps.emplace(parsedOp.dialect + "." + parsedOp.name + (parsedOp.version.empty() ? "" : "." + parsedOp.version), parsedOp);
            supportedOps.emplace(parsedOp.dialect + "_" + parsedOp.name + (parsedOp.version.empty() ? "" : "_" + parsedOp.version), parsedOp);
            supportedDialects.insert(parsedOp.dialect);
        }
    }

    opFile.close();

    // dump
    llvm::outs() << "Supported ops: ";
    for (const auto& [op, _]: supportedOps) {
        llvm::outs() << op << ", ";
    }

    llvm::outs() << "\nSupported dialects: ";
    for (const std::string& dialect: supportedDialects) {
        llvm::outs() << dialect << ", ";
    }

    llvm::outs() << "\n\n";
}

void EqualitySaturationPass::runOnOperation() {
    init();

    mlir::func::FuncOp rootOp = getOperation();
    llvm::StringRef rootOpName = rootOp.getName();

    llvm::outs() << "Running on function: " << rootOpName << "\n";
    llvm::outs() << "-----------------------------------------\n";

    // Perform equality saturation on all operations of a block.
    for (mlir::Block& block: rootOp.getRegion().getBlocks()) {
        std::string parentOpName = block.getParentOp()->getName().getStringRef().str();
        std::string blockName = rootOpName.str() + "_" + parentOpName;
        runOnBlock(block, blockName);

        // Temporary dead code elimination (until this PR is merged: https://github.com/llvm/llvm-project/pull/99671)
        bool clean = false;
        while (!clean) {
            clean = true;
            
            block.walk([&](mlir::Operation* op) {
                if (mlir::isOpTriviallyDead(op)) {
                    clean = false;
                    op->erase();
                }
            });
        }
    }

    llvm::outs() << "-----------------------------------------\n";
    llvm::outs() << "Done running on function: " << rootOpName << "\n";
    llvm::outs() << "mlirToEgglogTime = " << mlirToEgglogTime << "s\n";
    llvm::outs() << "egglogExecTime = " << egglogExecTime << "s\n";
    llvm::outs() << "egglogToMlirTime = " << egglogToMlirTime << "s\n";
    llvm::outs() << "-----------------------------------------\n";
}

std::unique_ptr<mlir::Pass> createEqualitySaturationPass(const std::string& mlirFile, const std::string& eggFile, const EgglogCustomDefs& funcs) {
    return std::make_unique<EqualitySaturationPass>(mlirFile, eggFile, funcs);
}