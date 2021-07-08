FROM quay.io/fenicsproject/stable:2016.1.0.r5

USER root
RUN apt-get -qq update && \
    apt-get -y --with-new-pkgs -o Dpkg::Options::="--force-confold" upgrade && \
    apt-get -y install gmsh && \
    pip2 --no-cache-dir install pandas==0.19.0 && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
