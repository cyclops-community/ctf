#!/bin/sh
# This configuration file was taken originally from the mpi4py project
# <http://mpi4py.scipy.org/>, and then modified for Julia, MADNESS, PRK, ...

set -e
set -x

MPI_ROOT="$1"
MPI_IMPL="$2"

# 1=yes, else no
MPI_FORTRAN=0

case "$TRAVIS_OS_NAME" in
    osx)
        echo "Mac"
        brew update
        case "$MPI_IMPL" in
            mpich)
                brew install mpich
                ;;
            ompi)
                brew install open-mpi || brew install openmpi
                ;;
            *)
                echo "Unknown MPI implementation: $MPI_IMPL"
                exit 10
                ;;
        esac
        ;;
    linux)
        echo "Linux"
        case "$CC" in
            gcc)
                for gccversion in "-6" "-5" "-5.3" "-5.2" "-5.1" "-4.9" "-4.8" "-4.7" "-4.6" "" ; do
                    if [ -f "`which gcc$gccversion`" ]; then
                        export CTF_CC="gcc$gccversion"
                        export CTF_CXX="g++$gccversion"
                        echo "Found GNU C/C++: $CTF_CC $CTF_CXX"
                        break
                    fi
                done
                ;;
            clang)
                for clangversion in "-omp" "-3.9" "-3.8" "-3.7" "-3.6" "-3.5" "-3.4" "" ; do
                    find /usr/local -name clang$clangversion
                    if [ -f "`which clang$clangversion`" ]; then
                        export CTF_CC="clang$clangversion"
                        export CTF_CXX="clang++$clangversion"
                        echo "Found Clang C/C++: $CTF_CC $CTF_CXX"
                        break
                    fi
                done
                ;;
        esac
        for gccversion in "-6" "-5" "-5.3" "-5.2" "-5.1" "-4.9" "-4.8" "-4.7" "-4.6" "" ; do
            if [ -f "`which gfortran$gccversion`" ]; then
                export CTF_FC="gfortran$gccversion"
                export MPI_FORTRAN=1
                echo "Found GNU Fortran: $CTF_FC"
                break
            fi
        done
        case "$MPI_IMPL" in
            mpich)
                if [ ! -f "$MPI_ROOT/bin/mpichversion" ]; then
                    set +e
                    wget --no-check-certificate -q \
                         http://www.mpich.org/static/downloads/3.2/mpich-3.2.tar.gz
                    set -e
                    if [ ! -f "$MPI_ROOT/mpich-3.2.tar.gz" ]; then
                        echo "MPICH download from mpich.org failed - trying Github mirror"
                        wget --no-check-certificate -q \
                             https://github.com/jeffhammond/mpich/archive/v3.2.tar.gz \
                             -O mpich-3.2.tar.gz
                        tar -xzf mpich-3.2.tar.gz
                        cd mpich-3.2
                    else
                        tar -xzf mpich-3.2.tar.gz
                        cd mpich-3.2
                    fi
                    sh $TRAVIS_HOME/travis/install-autotools.sh $MPI_ROOT
                    ./autogen.sh
                    mkdir build ; cd build
                    if [ "x$MPI_FORTRAN" != "x1" ] ; then
                        ../configure --prefix=$MPI_ROOT CC=$CTF_CC CXX=$CTF_CXX --disable-fortran
                    else
                        ../configure --prefix=$MPI_ROOT CC=$CTF_CC CXX=$CTF_CXX FC=$CTF_FC
                    fi
                    make -j4
                    make install
                else
                    echo "MPICH installed..."
                    find $MPI_ROOT -name mpiexec
                    find $MPI_ROOT -name mpicc
                fi
                ;;
            openmpi)
                if [ ! -f "$MPI_ROOT/bin/ompi_info" ]; then
                    wget --no-check-certificate -q http://www.open-mpi.org/software/ompi/v1.10/downloads/openmpi-1.10.1.tar.bz2
                    tar -xjf http://www.open-mpi.org/software/ompi/v1.10/downloads/openmpi-1.10.1.tar.bz2
                    cd openmpi-1.10.1
                    mkdir build && cd build
                    if [ "x$MPI_FORTRAN" != "x1" ] ; then
                        ../configure --prefix=$MPI_ROOT CC=$CTF_CC CXX=$CTF_CXX --enable-mpi-fortran=none
                    else
                        ../configure --prefix=$MPI_ROOT CC=$CTF_CC CXX=$CTF_CXX FC=$CTF_FC
                    fi
                    make -j4
                    make install
                else
                    echo "OpenMPI installed..."
                    find $MPI_ROOT -name mpiexec
                    find $MPI_ROOT -name mpicc
                fi


                ;;
            *)
                echo "Unknown MPI implementation: $MPI_IMPL"
                exit 20
                ;;
        esac
        ;;
esac
