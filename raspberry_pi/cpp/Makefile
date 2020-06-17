CXX_FLAG=-std=c++14 -O2 -fPIC
INCLUDE_FLAG=-I${TVM_ROOT} -I${TVM_ROOT}/include -I${TVM_ROOT}/3rdparty/dmlc-core/include -I${TVM_ROOT}/3rdparty/dlpack/include `pkg-config --cflags opencv4`
LD_FLAG=-L${TVM_ROOT}/build -ldl -lpthread `pkg-config --libs opencv4`

.PHONY: clean all
all: bin/inference bin/libtvm_runtime_pack.o

bin/libtvm_runtime_pack.o: tvm_runtime_pack.cc
	@mkdir -p $(@D)
	$(CXX) -c $(CXX_FLAG) $(INCLUDE_FLAG) -o $@ $^

bin/inference: inference.cc bin/libtvm_runtime_pack.o libtvm_resnet18_rpi.o
	@mkdir -p $(@D)
	$(CXX) $(CXX_FLAG) $(INCLUDE_FLAG) -o $@ $^ $(LD_FLAG)

