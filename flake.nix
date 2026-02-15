{
  description = "MalwareGrad: RDNA 4 Compute Environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };

      # We merge the isolated packages into one "virtual /opt/rocm"
      rocmEnv = pkgs.symlinkJoin {
        name = "rocm-combined";
        paths = with pkgs.rocmPackages; [
          rocminfo
          hipcc
          clr
          rocm-device-libs
        ];
      };
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        name = "rocm-dev-shell";

        nativeBuildInputs = [
          pkgs.cmake
          pkgs.ninja
          pkgs.gdb
          rocmEnv  # Use the combined environment
        ];

        shellHook = ''
          # 1. Point ROCM_PATH to our virtual monolith
          export ROCM_PATH=${rocmEnv}
          
          # 2. HIP still needs to know where Clang is (it's not in the join)
          export HIP_CLANG_PATH=${pkgs.rocmPackages.llvm.clang}/bin

          # 3. Explicitly tell hipcc where the bitcode is (just to be safe)
          # Note: The join puts it at $ROCM_PATH/amdgcn/bitcode
          export DEVICE_LIB_PATH=${rocmEnv}/amdgcn/bitcode
          export HIP_DEVICE_LIB_PATH=${rocmEnv}/amdgcn/bitcode
          export CLANG_RESOURCE_DIR="$(hipcc -print-resource-dir)"
          export CLANG_H_PATH="$CLANG_RESOURCE_DIR/include"
          
          echo "🚀 RDNA 4 Environment Loaded"
          echo "ROCM_PATH: $ROCM_PATH"
        '';
      };
    };
}