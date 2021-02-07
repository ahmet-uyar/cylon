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

int myrank = -1;

class RCB : public cylon::ReceiveCallback {

public:
    /**
     * This function is called when a data is received
     */
    bool onReceive(int source, std::shared_ptr<cylon::Buffer> buffer, int length) {
        std::string message;
        message.append(reinterpret_cast<const char*>(buffer->GetByteBuffer()));
        LOG(INFO) << myrank << ", Received a buffer: " << message;
        return true;
    }

    /**
     * Receive the header, this happens before we receive the actual data
     */
    bool onReceiveHeader(int source, int finished, int *buffer, int length) {
//        LOG(INFO) << "received a header with length: " << length;
        return true;
    }

    /**
     * This method is called after we successfully send a buffer
     * @return
     */
    bool onSendComplete(int target, void *buffer, int length) {
//        LOG(INFO) << "called onSendComplete with length: " << length << " for the target: " << target;
        return true;
    }

};

int main(int argc, char *argv[]) {

    auto start_start = std::chrono::steady_clock::now();
    auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
    auto ctx = cylon::CylonContext::InitDistributed(mpi_config);
    myrank = ctx->GetRank();

    LOG(INFO) << "myrank: "  << myrank << ", world size: " << ctx->GetWorldSize();

    std::vector<int> allWorkers{};
    for (int i = 0; i < ctx->GetWorldSize(); ++i) {
        allWorkers.push_back(i);
    }

    RCB * rcb = new RCB();
    cylon::DefaultAllocator allocator{};
    std::shared_ptr<cylon::Buffer> buffer;
    cylon::Status stat = allocator.Allocate(20, &buffer);
    if (!stat.is_ok()) {
        LOG(FATAL) << "Failed to allocate buffer with length " << 20;
    }

    std::shared_ptr<cylon::AllToAll> all =
            std::make_shared<cylon::AllToAll>(ctx, allWorkers, allWorkers, ctx->GetNextSequence(), rcb, &allocator);

    std::string message("hello from ");
    message += std::to_string(myrank);

    char msg[20];
    strcpy(msg, message.c_str());
    using namespace std::chrono_literals;
    for (int wID: allWorkers) {
        all->insert(msg, message.length() + 1, wID);
    }

    all->finish();

    int i = 1;
    while(!all->isComplete()) {
        if (i % 10 == 0) {
            LOG(INFO) << myrank << ", has not completed yet.";
            std::this_thread::sleep_for(1000ms);
        }
        i++;
    }

    ctx->Finalize();
    return 0;
}
