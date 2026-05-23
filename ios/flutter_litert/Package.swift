// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "flutter_litert",
    platforms: [
        .iOS("12.0")
    ],
    products: [
        .library(name: "flutter-litert", targets: ["flutter_litert"])
    ],
    targets: [
        .binaryTarget(
            name: "TensorFlowLiteC",
            url: "https://github.com/hugocornellier/flutter_litert/releases/download/flex-v1.1.0/TensorFlowLiteC-spm.xcframework.zip",
            checksum: "a2af0eb45c473b5b99bb6be181dbd3206aa18b3b9e0ef230b22257bd595ce375"
        ),
        .binaryTarget(
            name: "TensorFlowLiteCMetal",
            url: "https://github.com/hugocornellier/flutter_litert/releases/download/flex-v1.1.0/TensorFlowLiteCMetal-spm.xcframework.zip",
            checksum: "939a9dbbc88e912083c050476e202ebd783e43dac12c6a07cc5fa22e9b38d4ad"
        ),
        .binaryTarget(
            name: "TensorFlowLiteCCoreML",
            url: "https://github.com/hugocornellier/flutter_litert/releases/download/flex-v1.1.0/TensorFlowLiteCCoreML-spm.xcframework.zip",
            checksum: "11c5e1d4f76fec5cc81c21a0151927d3170e118ba50e6b20f59b9427018e5813"
        ),
        .target(
            name: "flutter_litert",
            dependencies: [
                .target(name: "TensorFlowLiteC"),
                .target(name: "TensorFlowLiteCMetal"),
                .target(name: "TensorFlowLiteCCoreML"),
            ],
            path: "Sources/flutter_litert",
            resources: [
                .process("PrivacyInfo.xcprivacy"),
            ],
            linkerSettings: [
                .linkedFramework("Metal", .when(platforms: [.iOS])),
                .linkedFramework("CoreML", .when(platforms: [.iOS])),
                .linkedFramework("Accelerate", .when(platforms: [.iOS])),
                .linkedLibrary("c++"),
                .unsafeFlags(["-ObjC"]),
            ]
        )
    ]
)
