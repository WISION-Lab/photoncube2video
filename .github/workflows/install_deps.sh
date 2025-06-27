# Install OpenSSL
if command -v yum &> /dev/null; then
    # If we're running on rhel centos, install needed packages.
    yum update -y && yum install -y perl-core openssl openssl-devel pkgconfig libatomic
    yum groupinstall 'Development Tools'
else
    # If we're running on debian-based system.
    apt update -y && apt-get install -y libssl-dev openssl pkg-config build-essential
fi

# Use zig as the linker to avoid common linking issues
python -m ensurepip --upgrade
python -m pip install ziglang
