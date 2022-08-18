# ML-Zoomcamp attempt

[course](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp)

ML parts has finished. working on deployment

## deployment

try to mimic how tensorflow serving, a simpler inference server inside [inference](./inference/) with grpc server, a http gateway server that receive raw data from user, preprocessing the data then send to inference server for prediction.

currently ch2 has a working demo. the plan is to have

- [x] a docker compose version
- [ ] a k8s version
- [ ] a kserve version
- [ ] a aws version (maybe)

## Scipy problem on Apple M1

```
brew install openblas
export OPENBLAS=$(/opt/homebrew/bin/brew --prefix openblas)
export CFLAGS="-falign-functions=8 ${CFLAGS}"
```
