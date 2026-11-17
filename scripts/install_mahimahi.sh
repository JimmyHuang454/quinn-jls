#!/usr/bin/env bash
set -euo pipefail

variant=${1:-standard}
source_dir=$(mktemp -d)
sudo sysctl -w net.ipv4.ip_forward=1

if [[ "$variant" == leo ]]; then
    # Keep the replayer revision aligned with the example traces used by the suite.
    revision=69df6406e9985a764b4d52b506cd6ee77596ffb2
    sudo apt-get install -y -qq apache2-bin apache2-dev ssl-cert protobuf-compiler libprotobuf-dev
    curl -fsSL --retry 3 "https://codeload.github.com/SpaceNetLab/LeoCC/tar.gz/$revision" \
        | tar -xz --strip-components=1 -C "$source_dir"
    cd "$source_dir/leoreplayer/replayer"
    # The upstream replayer was built on Ubuntu 22.04. Keep warnings visible on newer compilers.
    sed -i 's/ -Werror"/"/' configure.ac
    ./autogen.sh
    ./configure
    for directory in util packet graphing; do
        make -C "src/$directory" -j"$(nproc)"
    done
    make -C src/frontend -j"$(nproc)" mm-delay mm-link mm-loss
    sudo install -o root -g root -m 4755 src/frontend/{mm-delay,mm-link,mm-loss} /usr/local/bin/
else
    git clone --depth 1 https://github.com/BobAnkh/mahimahi-prune.git "$source_dir"
    cd "$source_dir"
    ./autogen.sh
    ./configure
    make -j"$(nproc)"
    sudo make install
fi
