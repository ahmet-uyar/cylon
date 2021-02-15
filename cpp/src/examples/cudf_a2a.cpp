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

#include <cudf/types.hpp>
#include <cudf/copying.hpp>
#include <cudf/io/types.hpp>
#include <cudf/io/csv.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/column/column.hpp>
#include <cuda_runtime.h>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>

int myrank = -1;

class CudfBuffer : public cylon::Buffer {
public:
    CudfBuffer(std::shared_ptr<rmm::device_buffer> rmmBuf) :
        rmmBuf(std::move(rmmBuf)), length(rmmBuf->size()) {}

    int64_t GetLength() override {
        return length;
    }

    uint8_t *GetByteBuffer() override {
        return (uint8_t *)rmmBuf->data();
    }

    std::shared_ptr<rmm::device_buffer> getBuf() const {
        return rmmBuf;
    }

private:
    std::shared_ptr<rmm::device_buffer> rmmBuf;
    int64_t length;
};


class CudfAllocator : public cylon::Allocator {
public:
    cylon::Status Allocate(int64_t length, std::shared_ptr<cylon::Buffer> *buffer) override {
        std::shared_ptr<rmm::device_buffer> rmmBuf;
        try {
            rmmBuf = std::make_shared<rmm::device_buffer>(length);
        } catch (rmm::bad_alloc	badAlloc) {
            LOG(ERROR) << "failed to allocate gpu memory with rmm: " << badAlloc.what();
            return cylon::Status(cylon::Code::GpuMemoryError);
        }
        *buffer = std::make_shared<CudfBuffer>(rmmBuf);
        return cylon::Status::OK();
    }
};

class RCB : public cylon::ReceiveCallback {

public:
    RCB() : data_type(-1) {}

   /**
    * This function is called when a data is received
    */
   bool onReceive(int source, std::shared_ptr<cylon::Buffer> buffer, int length) {
       std::shared_ptr<CudfBuffer> cb = std::dynamic_pointer_cast<CudfBuffer>(buffer);

       uint8_t *hostArray= (uint8_t*)malloc(length);
       cudaMemcpy(hostArray, cb->GetByteBuffer(), length, cudaMemcpyDeviceToHost);

       if (data_type == 3) {
           int32_t * hdata = (int32_t *) hostArray;
           LOG(INFO) << "==== data[0]: " << hdata[0] << ", data[1]: " << hdata[1];
       } else if (data_type == 4) {
           int64_t *hdata = (int64_t *) hostArray;
           LOG(INFO) << "==== data[0]: " << hdata[0] << ", data[1]: " << hdata[1];
       }

       return true;
   }

    /**
     * Receive the header, this happens before we receive the actual data
     */
    bool onReceiveHeader(int source, int finished, int *buffer, int length) {
        data_type = buffer[0];
        LOG(INFO) << "----received a header buffer with length: " << length;
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

private:
    int data_type;
};

// sizes of cudf types in tables
// ref: type_id in cudf/types.hpp
int type_bytes[] = {0, 1, 2, 4, 8, 1, 2, 4, 8, 4, 8, 1, 4, 8, 8, 8, 8, 4, 8, 8, 8, 8, -1, -1, -1, 4, 8, -1, -1};

/**
 * data buffer length of a column in bytes
 * @param input
 * @return
 */
cudf::size_type dataLength(cudf::column_view const& input){
    int elementSize = type_bytes[static_cast<int>(input.type().id())];
    if (elementSize == -1) {
        std::cout << "ERRORRRRRR unsupported type id: " << static_cast<int>(input.type().id()) << std::endl;
        return -1;
    }

    // even null values exist in the buffer with unspecified values
    return elementSize * input.size();
}


int main(int argc, char *argv[]) {

    if (argc != 2) {
        std::cout << "You must specify a CSV input file.\n";
        return 1;
    }

    std::string input_csv_file = argv[1];
    cudf::io::source_info si(input_csv_file);
    cudf::io::csv_reader_options options = cudf::io::csv_reader_options::builder(si);
    cudf::io::table_with_metadata ctable = cudf::io::read_csv(options);
    std::cout << "number of columns: " << ctable.tbl->num_columns() << std::endl;
    int columnIndex = myrank;
    cudf::column_view cw = ctable.tbl->get_column(columnIndex).view();
    std::cout << "column[" << columnIndex << "] size: " << cw.size() << std::endl;


    auto start_start = std::chrono::steady_clock::now();
    auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
    auto ctx = cylon::CylonContext::InitDistributed(mpi_config);
    myrank = ctx->GetRank();

    LOG(INFO) << "myrank: "  << myrank << ", world size: " << ctx->GetWorldSize();

    std::vector<int> allWorkers{};
    for (int i = 0; i < ctx->GetWorldSize(); ++i) {
        allWorkers.push_back(i);
    }

    int numberOfGPUs;
    cudaGetDeviceCount(&numberOfGPUs);
    LOG(INFO) << "myrank: "  << myrank << ", number of GPUs: " << numberOfGPUs;

    // set the gpu
    cudaSetDevice(myrank % numberOfGPUs);

    RCB * rcb = new RCB();
    CudfAllocator allocator{};
    std::shared_ptr<cylon::Buffer> buffer;
    cylon::Status stat = allocator.Allocate(20, &buffer);
    if (!stat.is_ok()) {
        LOG(FATAL) << "Failed to allocate buffer with length " << 20;
    }

    std::shared_ptr<cylon::AllToAll> all =
            std::make_shared<cylon::AllToAll>(ctx, allWorkers, allWorkers, ctx->GetNextSequence(), rcb, &allocator);

    auto headers = std::make_shared<std::vector<int32_t>>();
    headers->push_back((int32_t)(cw.type().id()));
    headers->push_back(myrank);

    uint8_t *sendBuffer = (uint8_t *)cw.data<uint8_t>();
    int dataLen = dataLength(cw);

    for (int wID: allWorkers) {
        all->insert(sendBuffer, dataLen, wID, headers->data(), headers->size());
    }

    all->finish();

    using namespace std::chrono_literals;
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
