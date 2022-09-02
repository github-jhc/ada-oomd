- Put source codes:
```bash
tar -zxvf my_spiel.tar.gz
copy -r my_spiel /
cd /my_spiel
```
- Set up environment and buid project:
```bash
source pre_docker.sh
```
The pre_docker.sh should looks like:
```bash
source env.sh
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j10 && cd ..
mkdir -p /data/liuwm/my_spiel/results
mkdir -p /data/liuwm/my_spiel/models
rm -rf results && ln -s /data/liuwm/my_spiel/results results
rm -rf models && ln -s /data/liuwm/my_spiel/models models
```
Verify that there are two soft links
```bash
models -> /data/liuwm/my_spiel/models/
results -> /data/liuwm/my_spiel/results/
```
The output should looks like
```bash
...
/usr/bin/cmake -E cmake_progress_start /my_spiel/build/CMakeFiles 0
```
-Run
```bash
./run_deep_cfr.sh # run Deep CFR
./run_ncfrb.sh # run Neural CFR-B
```