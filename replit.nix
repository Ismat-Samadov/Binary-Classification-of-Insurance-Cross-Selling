{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = [
    pkgs.pgadmin4
    pkgs.postgresql
    pkgs.openssl
    pkgs.python3
    pkgs.python3Packages.psycopg2
  ];
}
