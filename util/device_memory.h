#pragma once
struct NoInit {
};

template<typename T, bool need_host>
struct DeviceMemory {
    T *p = nullptr;
    T *dev_p;
    unsigned size;
    cudaStream_t stream;

    template<typename Initializer = NoInit>
    explicit DeviceMemory(const unsigned size, Initializer initializer = Initializer{}) : size(size), stream() {
        cudaStreamCreate(&stream);
        cudaMallocAsync(&dev_p, size * sizeof(T), stream);
        if constexpr (need_host) {
            p = static_cast<T *>(malloc(size * sizeof(T)));
            if constexpr (!std::is_same_v<NoInit, Initializer>) {
                for (unsigned i = 0; i < size; ++i) {
                    p[i] = initializer(i);
                }
                cudaMemcpyAsync(dev_p, p, size * sizeof(T), cudaMemcpyHostToDevice, stream);
            }
        }
    }

    DeviceMemory(const DeviceMemory &x) = delete;

    ~DeviceMemory() {
        cudaFreeAsync(dev_p, stream);
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
        if constexpr (need_host) {
            free(p);
        }
    }

    void resetDevice() const {
        cudaMemset(dev_p, 0, size * sizeof(T));
    }

    void deviceToHost() const {
        if constexpr (need_host) {
            cudaMemcpy(p, dev_p, size * sizeof(T), cudaMemcpyDeviceToHost);
        }
    }
};
