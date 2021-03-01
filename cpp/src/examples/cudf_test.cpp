#include "include/cudf_a2a.hpp"

#include <iostream>
#include <cmath>
#include <bitset>

#include <cudf/copying.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/io/types.hpp>
#include <cudf/io/csv.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/detail/get_value.cuh>

using std::cout;
using std::endl;
using std::string;

std::unique_ptr<cudf::column> emptyLike(cudf::column_view const& input){
    int dl = dataLength(input);
    rmm::device_buffer dataBuf(dl);
    cudaMemcpy(dataBuf.data(), input.data<uint8_t>(), dl, cudaMemcpyDeviceToDevice);

    int nullBufSize = ceil(input.size() / 8.0);
    if (!input.nullable()) {
        nullBufSize = 0;
    }
    rmm::device_buffer nullBuf(nullBufSize);
    cudaMemcpy(nullBuf.data(), input.null_mask(), nullBufSize, cudaMemcpyDeviceToDevice);

    return std::make_unique<cudf::column>(
            input.type(), input.size(), dataBuf, nullBuf, input.null_count());
}

void testEmptyLike(cudf::column_view const& input) {
    std::unique_ptr<cudf::column> copyColumn = emptyLike(input);
    cudf::column_view cw = copyColumn->view();

    // copy column data to host memory and print
    uint8_t *hostArray= (uint8_t*)malloc(static_cast<int>(dataLength(cw)));
    cudaMemcpy(hostArray, cw.data<uint8_t>(), static_cast<int>(dataLength(cw)), cudaMemcpyDeviceToHost);
    int64_t * hdata = (int64_t *) hostArray;
    //char * hdata = (char *) hostArray;
    std::cout << "first data: " << hdata[0] << std::endl;
}

void printCharArray(cudf::column_view cv, int charSize) {
    char *hostArray= new char[charSize+1];
    cudaMemcpy(hostArray, cv.data<char>(), charSize, cudaMemcpyDeviceToHost);
    hostArray[charSize] = '\0';
    std::cout << "chars:: " << hostArray << std::endl;
}

void printOffsetsArray(cudf::column_view cv) {
    std::cout << "offsets column size: " << cv.size() << ", type: " << static_cast<int>(cv.type().id()) << std::endl;
    uint8_t *hostArray= new uint8_t[dataLength(cv)];
    cudaMemcpy(hostArray, cv.data<uint8_t>(), dataLength(cv), cudaMemcpyDeviceToHost);
    int32_t * hdata = (int32_t *) hostArray;
    for (int i = 0; i < cv.size(); ++i) {
        std::cout << hdata[i] << ", ";
    }
    std::cout << std::endl;
}

void printNullMask(cudf::column_view cv) {
    if (!cv.nullable()) {
        std::cout << "the column is not nullable ......................: " << std::endl;
        return;
    }
    std::cout << "number of nulls in the column: " << cv.null_count() << std::endl;
    std::cout << "offset for the first element in the column: " << cv.offset() << std::endl;
    std::size_t size = cudf::bitmask_allocation_size_bytes(cv.size());
    uint8_t *hostArray= new uint8_t[size];
    cudaMemcpy(hostArray, (uint8_t *)cv.null_mask(), size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < size; ++i) {
        std::bitset<8> x(hostArray[i]);
        std::cout << i << ":" << x << " ";
    }
    std::cout << std::endl;
}

void printChildColumns(cudf::table_view tview) {

    for (int i = 0; i < tview.num_columns(); ++i) {
        cudf::column_view cw = tview.column(i);
        std::cout << "column[" << i << "] children: " << cw.num_children() << std::endl;
        printNullMask(cw);
        if (cw.type().id() == cudf::type_id::STRING) {
            std::cout << "column type is STRING ------------------------------------------------ " << std::endl;
            cudf::strings_column_view scv(cw);
            std::cout << "number of strings: " << scv.size() << std::endl;
            std::cout << "number of nulls: " << scv.null_count() << std::endl;
            std::cout << "offsets_column_index: " << scv.offsets_column_index << std::endl;
            std::cout << "chars_column_index: " << scv.chars_column_index << std::endl;
            std::cout << "chars_size: " << scv.chars_size() << std::endl;
            printCharArray(scv.chars(), scv.chars_size());
            printOffsetsArray(scv.offsets());
        }
    }
}

int tableConcat(int argc, char** argv) {

    if (argc != 3) {
        std::cout << "You must specify two CSV input files.\n";
        return 1;
    }
    string input_csv_file1 = argv[1];
    cudf::io::source_info si1(input_csv_file1);
    cudf::io::csv_reader_options options1 = cudf::io::csv_reader_options::builder(si1);
    cudf::io::table_with_metadata ctable1 = cudf::io::read_csv(options1);
    cout << "table from: " << input_csv_file1 << ", number of columns: " << ctable1.tbl->num_columns()
         << ", first column size: " << ctable1.tbl->get_column(0).size() << endl;

    std::string input_csv_file2 = argv[2];
    cudf::io::source_info si2(input_csv_file2);
    cudf::io::csv_reader_options options2 = cudf::io::csv_reader_options::builder(si2);
    cudf::io::table_with_metadata ctable2 = cudf::io::read_csv(options2);
    cout << "table from: " << input_csv_file2 << ", number of columns: " << ctable2.tbl->num_columns()
         << ", first column size: " << ctable2.tbl->get_column(0).size() << endl;

    std::vector<cudf::table_view> tables_to_concat{};
    tables_to_concat.push_back(ctable1.tbl->view());
    tables_to_concat.push_back(ctable2.tbl->view());
    std::unique_ptr<cudf::table> contatTable = cudf::concatenate(tables_to_concat);
    cout << "concatenated table, number of columns: " << contatTable->num_columns()
         << ", first column size: " << contatTable->get_column(0).size() << endl;

    return 0;
}

void printLongColumn(cudf::column_view const& input, int index) {
    cout << "column[" << index << "]: ";
    for (cudf::size_type i = 0; i < input.size(); ++i) {
        cout << cudf::detail::get_value<int64_t>(input, i, rmm::cuda_stream_default) << ", ";
    }
    cout << endl;
}


int tablePartition(int argc, char** argv) {

    if (argc != 2) {
        std::cout << "You must specify two CSV input files.\n";
        return 1;
    }
    string input_csv_file1 = argv[1];
    cudf::io::source_info si1(input_csv_file1);
    cudf::io::csv_reader_options options1 = cudf::io::csv_reader_options::builder(si1);
    cudf::io::table_with_metadata ctable1 = cudf::io::read_csv(options1);
    cout << "table from: " << input_csv_file1 << ", number of columns: " << ctable1.tbl->num_columns()
         << ", first column size: " << ctable1.tbl->get_column(0).size() << endl;

    printLongColumn(ctable1.tbl->get_column(0).view(), 0);

    std::vector<cudf::size_type> columns_to_hash{};
    columns_to_hash.push_back(0);

    std::pair<std::unique_ptr<cudf::table>, std::vector<cudf::size_type>> pr
        = cudf::hash_partition(ctable1.tbl->view(), columns_to_hash, 10);

    cout << "partitioned table, number of columns: " << pr.first->num_columns()
         << ", first column size: " << pr.first->get_column(0).size() << endl;

    cout << "offsets: ";
    for (auto offset: pr.second) {
        cout << offset << ", ";
    }
    cout << endl;

    printLongColumn(pr.first->get_column(0).view(), 0);

    return 0;
}

int main(int argc, char** argv) {
    cout << "CUDF Example\n";

//    return tableConcat(argc, argv);
    return tablePartition(argc, argv);

    if (argc != 2) {
        std::cout << "You must specify a CSV input file.\n";
        return 1;
    }
    std::string input_csv_file = argv[1];
    cudf::io::source_info si(input_csv_file);
    cudf::io::csv_reader_options options = cudf::io::csv_reader_options::builder(si);
    cudf::io::table_with_metadata ctable = cudf::io::read_csv(options);
    std::cout << "number of columns: " << ctable.tbl->num_columns() << std::endl;

    cudf::table_view tview = ctable.tbl->view();
    printChildColumns(tview);
    return 0;

    cudf::column column1 = ctable.tbl->get_column(0);
    std::cout << "column size: " << column1.size() << std::endl;

    cudf::column_view cw = column1.view();
    std::cout << "column view size: " << cw.size() << std::endl;
    cudf::data_type dt = cw.type();
    if (dt.id() == cudf::type_id::STRING){
        std::unique_ptr<cudf::scalar> sp = cudf::get_element(cw, 0);
        cudf::string_scalar *ssp = static_cast<cudf::string_scalar*>(sp.get());
        //std::unique_ptr<cudf::string_scalar>  ssp(static_cast<cudf::string_scalar*>(sp.release()));
        std::cout << "element 0: " << ssp->to_string() << std::endl;
    }

    testEmptyLike(cw);

//    std::cout << "data type int value: " << static_cast<int>(dt.id()) << std::endl;
//    std::cout << "INT32 data type int value: " << static_cast<int>(cudf::type_id::INT32) << std::endl;
//    if (dt.scale() == cudf::type_id::UINT32) {
//       std::cout << "data type: UINT32" << std::endl;

//    int dataSize = cw.end<uint8_t>() - cw.begin<uint8_t>();
//    std::cout << "dataSize: " << dataSize << std::endl;
//    auto * data = cw.data<uint8_t>();


//    static_cast<cudf::type_id>(dt.id()) * data = cw.data();
//    std::cout << "first two data: " << cw.data()[0] << ", " << cw.data()[1] << std::endl;

//    cudf::column::contents ccontents = column1.release();
//    std::cout << "column data size: " << ccontents.data->size() << std::endl;
//    std::cout << "column null_mask size: " << ccontents.null_mask->size() << std::endl;

    return 0;
}