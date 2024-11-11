#include "llvm/Support/FileSystem.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/AsmParser/AsmParser.h"

#include "Egglog.h"
#include "EqualitySaturationPass.h"
#include "EggifyPass.h"
#include "EgglogCustomDefs.h"
#include "MatrixMultiplyAssociatePass.h"

std::string getMlirFile(int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
        if (argv[i][0] != '-') {
            return argv[i];
        }
    } 
}

std::string getEggFile(int argc, char** argv, std::string mlirFile) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (i < argc && (arg == "-egg" || arg == "--egg")) {
            return argv[i + 1];
        }
    }

    // Otherwise, use the mlir file name with .egg extension
    std::string name = mlirFile.substr(0, mlirFile.find(".mlir"));
    return name + ".egg";
}

int main(int argc, char** argv) {
    // Register dialects
    mlir::DialectRegistry dialectRegistry;
    mlir::registerAllDialects(dialectRegistry);

    // Register passes
    mlir::registerAllPasses();
    mlir::PassRegistration<EggifyPass>();
    mlir::PassRegistration<MatrixMultiplyAssociatePass>();

    // Equality Saturation Pass
    static llvm::cl::opt<std::string> eggFileOpt(
        "egg",
        llvm::cl::desc("Path to egg file"),
        llvm::cl::value_desc("filename"));

    std::string mlirFile = getMlirFile(argc, argv);
    std::string eggFile = getEggFile(argc, argv, mlirFile);

    llvm::outs() << "mlirFile: " << mlirFile << "\n";
    llvm::outs() << "eggFile: " << eggFile << "\n";
    
    std::map<std::string, AttrStringifyFunction> attrStringifiers = {
            {mlir::arith::FastMathFlagsAttr::name.str(), stringifyFastMathFlagsAttr}};
    std::map<std::string, AttrParseFunction> attrParsers = {
            {"arith_fastmath", parseFastMathFlagsAttr}};

    std::map<std::string, TypeStringifyFunction> typeStringifiers = {
            {mlir::RankedTensorType::name.str(), stringifyRankedTensorType}};
    std::map<std::string, TypeParseFunction> typeParsers = {
            {"RankedTensor", parseRankedTensorType}};

    EgglogCustomDefs funcs = {attrStringifiers, attrParsers, typeStringifiers, typeParsers};
    mlir::PassRegistration<EqualitySaturationPass>([&] { 
        return createEqualitySaturationPass(mlirFile, eggFile, funcs); 
    });

    // Run the main MLIR opt
    mlir::LogicalResult result = mlir::MlirOptMain(argc, argv, "EqSat", dialectRegistry);
    return mlir::asMainReturnCode(result);
}