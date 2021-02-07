//
// Created by auyar on 3.02.2021.
//

#include <glog/logging.h>
#include <chrono>
#include <thread>

#include <net/ops/all_to_all.hpp>
#include <net/mpi/mpi_communicator.hpp>
#include <ctx/cylon_context.hpp>
#include <status.hpp>
#include <net/buffer.hpp>
#include <cuda_runtime.h>

class CUDABuffer : public cylon::Buffer {
public:
    int64_t GetLength() override {
        return length;
    }
    uint8_t * GetByteBuffer() override {
        return buf;
    }
    CUDABuffer(uint8_t *buf, int64_t length) : buf(buf), length(length) {}
private:
    uint8_t *buf;
    int64_t length;
};

class CUDAAllocator : public cylon::Allocator {
public:
    cylon::Status Allocate(int64_t length, std::shared_ptr<cylon::Buffer> *buffer) override {
        uint8_t *b;
        cudaMallocManaged(&b, length * sizeof(char));
        *buffer = std::make_shared<CUDABuffer>(b, length);
        return cylon::Status::OK();
    }
};


class RCB : public cylon::ReceiveCallback {

public:
   /**
    * This function is called when a data is received
    * @param source the source
    * @param buffer the buffer allocated by the system, we need to free this
    * @param length the length of the buffer
    * @return true if we accept this buffer
    */
    bool onReceive(int source, std::shared_ptr<cylon::Buffer> buffer, int length) {
       std::string message;
       message.append(reinterpret_cast<const char*>(buffer->GetByteBuffer()));
        LOG(INFO) << "received a buffer through onReceive with length: " << length << ", " << message;
     return true;
    }

    /**
     * Receive the header, this happens before we receive the actual data
     * @param source the source
     * @param buffer the header buffer, which can be 6 integers
     * @param length the length of the integer array
     * @return true if we accept the header
     */
    bool onReceiveHeader(int source, int finished, int *buffer, int length) {
        LOG(INFO) << "received a header with length: " << length;
        return true;
    }

    /**
     * This method is called after we successfully send a buffer
     * @param target
     * @param buffer
     * @param length
     * @return
     */
    bool onSendComplete(int target, void *buffer, int length) {
        LOG(INFO) << "called onSendComplete with length: " << length << " for the target: " << target;
        return true;
    }

};

int main(int argc, char *argv[]) {

    auto start_start = std::chrono::steady_clock::now();
    auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
    auto ctx = cylon::CylonContext::InitDistributed(mpi_config);
    int myrank = ctx->GetRank();

    LOG(INFO) << "myrank: "  << myrank << ", world size: " << ctx->GetWorldSize();

    std::vector<int> allWorkers{};
    for (int i = 0; i < ctx->GetWorldSize(); ++i) {
        allWorkers.push_back(i);
    }

    RCB * rcb = new RCB();
    CUDAAllocator allocator{};
    std::shared_ptr<cylon::Buffer> buffer = std::make_shared<cylon::DefaultBuffer>(nullptr, 0);
    cylon::Status stat = allocator.Allocate(20, &buffer);
    if (!stat.is_ok()) {
        LOG(FATAL) << "Failed to allocate buffer with length " << 20;
    }

    std::shared_ptr<cylon::AllToAll> all =
            std::make_shared<cylon::AllToAll>(ctx, allWorkers, allWorkers, ctx->GetNextSequence(), rcb, &allocator);

    char message[20];
    using namespace std::chrono_literals;
    if (myrank == 0) {
        strcpy(message, "Hello, there");
        all->insert(message, strlen(message) + 1, 1);
        all->insert(message, strlen(message) + 1, 1);
    }

    if (myrank == 1) {
        strcpy(message, "Hello, there ..");
        all->insert(message, strlen(message) + 1, 0);
        all->insert(message, strlen(message) + 1, 0);
    }

    all->finish();

    while(!all->isComplete()) {
        LOG(INFO) << myrank <<  ", has not completed yet.";
        std::this_thread::sleep_for(1000ms);
    }

    ctx->Finalize();
    return 0;
}
