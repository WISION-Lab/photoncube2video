if command -v yum &> /dev/null; then
    # If we're running on rhel centos, install needed packages.
    yum update -y && yum install -y perl-core openssl openssl-devel pkgconfig libatomic
else
    # If we're running on debian-based system.
    apt update -y && apt-get install -y libssl-dev openssl pkg-config
fi
