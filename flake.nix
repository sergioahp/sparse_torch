{
  description = "PyTorch sparse tensor benchmarking environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python3;
        pythonPackages = python.pkgs;
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            python
            pythonPackages.torch
            pythonPackages.numpy
            pythonPackages.pip
            pythonPackages.virtualenv
            pythonPackages.pandas
            pythonPackages.matplotlib
            pythonPackages.seaborn
          ];

        };
      });
}
