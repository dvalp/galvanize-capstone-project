Spark includes it's own version of Maven, so just go to the directory for the source code.
Build spark from the latest version using a clone of the github repo using the following line:
`./build/mvn -Pyarn -Phadoop-2.7 -Dhadoop.version=2.7.3 -Dscala-2.11 -Pnetlib-lgpl -DskipTests clean package`

Build OpenBLAS from a clone of the github repo by running `make` in the directory where the repo is saved. Then use `sudo make PREFIX=/opt/OpenBLAS install` to install it in a directory. Add the directory `/opt/OpenBLAS/lib` to `/etc/ld.so.conf.d/openblas.conf` and run `ldconfig` to load the new libraries.

However, in the end I wound up using the line `sudo apt-get install libatlas3-base libopenblas-base` to install the files from the repository because I couldn't get Spark to recognize my version of OpenBLAS built from source. I may revisit this in the future with Mint/Ubuntu alternatives sourcing (ie, `sudo update-alternatives --config libblas.so`)

This page is very helpful in resolving problems getting Spark to use BLAS:
http://www.spark.tc/blas-libraries-in-mllib/
