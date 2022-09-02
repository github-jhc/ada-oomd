source env.sh
mkdir -p build && cd build 
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j10 && cd ..
mkdir -p /data/liuwm/my_spiel/results
mkdir -p /data/liuwm/my_spiel/models
rm -rf results && ln -s /data/liuwm/my_spiel/results results
rm -rf models && ln -s /data/liuwm/my_spiel/models models
git config --global user.name "Weiming Liu"
git config --global user.email "weiming@mail.ustc.edu.cn"