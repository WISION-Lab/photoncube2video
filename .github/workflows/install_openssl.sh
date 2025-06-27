# Install OpenSSL
if command -v yum &> /dev/null; then
    # If we're running on rhel centos, install needed packages.
    yum update -y && yum install -y perl-core openssl openssl-devel pkgconfig libatomic
else
    # If we're running on debian-based system.
    apt update -y && apt-get install -y libssl-dev openssl pkg-config
fi

# Missing linker envvars
export HOST_CC=gcc
export CC_x86_64_unknown_linux_gnu=/usr/bin/x86_64-linux-gnu-gcc
export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=/usr/bin/x86_64-linux-gnu-gcc
