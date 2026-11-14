#!/bin/bash
# Standalone build of the paper's binary; touches nothing under ns-3-dev.
# The patched NtnSib19Broadcaster and NtnUlTimingPrecompensator are compiled under another class name, so nothing overrides the
# library: libns3.43-ntn-rrc-optimized.so keeps its own class and this executable uses its own.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
B=/home/uzair/6g_ntn_ns3/ns-3-dev/build
OUT="$HERE/../bin"
FLAGS="-DEIGEN_MPL2_ONLY -DHAVE_BOOST -DHAVE_BOOST_UNITS -DHAVE_EIGEN3 -DHAVE_GSL -DHAVE_LIBXML2 -DHAVE_PACKET_H -DHAVE_SQLITE3 -DNS3_BUILD_PROFILE_OPTIMIZED -D__LINUX__ -DP5_STANDALONE -DNtnSib19Broadcaster=NtnSib19BroadcasterP5 -DNtnUlTimingPrecompensator=NtnUlTimingPrecompensatorP5 -I$HERE -I$B/include -iquote $B/include/ns3 -I/usr/include/eigen3 -I/usr/include/libxml2 -O3 -DNDEBUG -std=c++20 -fPIE -fno-semantic-interposition -march=native -mtune=native"
LIBS=$(ls $B/lib/libns3.43-*-optimized.so | grep -v -- "-test-" | grep -E "libns3.43-(ntn-rrc|ntn-constellation|mobility|network|stats|core)-optimized" | sort)
nice -n 19 g++ $FLAGS -c "$HERE/ntn-sib19.cc" -o "$OUT/ntn-sib19.o"
nice -n 19 g++ $FLAGS -c "$HERE/ntn-rrc-ul-timing.cc" -o "$OUT/ntn-rrc-ul-timing.o"
nice -n 19 g++ $FLAGS -c "$HERE/ntn-ul-timing-precompensator.cc" -o "$OUT/ntn-ul-timing-precompensator.o"
nice -n 19 g++ -o "$OUT/p5-ul-timing" "$OUT/ntn-rrc-ul-timing.o" "$OUT/ntn-sib19.o" "$OUT/ntn-ul-timing-precompensator.o" $LIBS -Wl,-rpath,$B/lib -Wl,-rpath-link,$B/lib
rm -f "${OUT:?}/ntn-sib19.o" "${OUT:?}/ntn-rrc-ul-timing.o" "${OUT:?}/ntn-ul-timing-precompensator.o"
sha256sum "$OUT/p5-ul-timing"
