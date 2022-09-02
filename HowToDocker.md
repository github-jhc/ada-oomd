- Start a container:
```bash
# --gpus all, selects all gpus.
# -v dir1:dir2, mount dir1 in the host as dir2 in the container.
docker run -it --gpus all  -v /data:/data liuwm/my_spiel
```
- Stop a container: use command `exit`.
- Let a container run in background: push `Ctrl+P+Q`.
- List all containers:
```bash
docker container ls -a
```
- Reenter a container running in background: 
```bash
docker exec -it <ID> /bin/sh
```
-Reenter a stopped container:
```bash
docker start <ID> /bin/sh
docker exec -it <ID> /bin/sh
```
- Stop a container running in background: 
```bash
docker container stop <ID>
```
- Delete a container:
```bash
docker container rm <ID>
```
- Save a image:
```bash
 docker save liuwm/my_spiel:latest | gzip > my_spiel_latest.tar.gz
 ```
 - Load a image:
 ```bash
 gunzip -c my_spiel_latest.tar.gz | docker load
 ```
    