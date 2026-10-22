#!/bin/bash
# Add pair_style mlipcpp (package ML-MLIPCPP) to a LAMMPS source tree, then
# build LAMMPS with -D PKG_ML-MLIPCPP=ON (and -D MLIPCPP_LIBRARY=... unless
# mlip.cpp was built in its build/ directory).
#
#   patch_lammps.sh /path/to/lammps          copy the sources in
#   patch_lammps.sh /path/to/lammps --link   symlink them, for working on them
#
# Running it again updates the tree. It leaves other packages alone, so one
# LAMMPS tree can carry pet-kokkos's ML-PET as well.
set -euo pipefail
lammps=$(cd "${1:?usage: patch_lammps.sh /path/to/lammps [--link]}" && pwd)
here=$(cd "$(dirname "$0")" && pwd)
mlipcpp=$(dirname "$here")
[ -f "$lammps/cmake/CMakeLists.txt" ] && [ -f "$lammps/src/pair.h" ] || { echo "not a LAMMPS source tree: $lammps" >&2; exit 1; }

link=0; [ "${2:-}" = "--link" ] && link=1
put() {  # put SRC DST: copy, or symlink with --link
  rm -rf "$2"
  if [ $link = 1 ]; then ln -s "$1" "$2"; else cp -r "$1" "$2"; fi
}

put "$here/ML-MLIPCPP" "$lammps/src/ML-MLIPCPP"
sed "s|@MLIPCPP_DIR@|$mlipcpp|" "$here/cmake/ML-MLIPCPP.cmake" > "$lammps/cmake/Modules/Packages/ML-MLIPCPP.cmake"

# Register the package: its name, and its module with the other packages that
# link an external library.
cmake="$lammps/cmake/CMakeLists.txt"
if ! grep -q "ML-MLIPCPP" "$cmake"; then
  sed -i -e 's/^\(\s*\)ML-IAP$/&\n\1ML-MLIPCPP/' \
         -e 's/^\(foreach(PKG_WITH_INCL KSPACE .*\)$/\1 ML-MLIPCPP/' "$cmake"
fi
grep -q "^\s*ML-MLIPCPP$" "$cmake" && grep -q "^foreach(PKG_WITH_INCL KSPACE .* ML-MLIPCPP$" "$cmake" \
  || { echo "could not register ML-MLIPCPP in $cmake" >&2; exit 1; }
echo "ML-MLIPCPP added to $lammps (mlip.cpp: $mlipcpp). Configure LAMMPS with -D PKG_ML-MLIPCPP=ON."
