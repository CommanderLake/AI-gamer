#include "common.h"
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>
#include <sstream>
std::vector<std::string> trainDataFiles = {"E:\\TrainingData\\DeltaHalo0X11.bin", "E:\\TrainingData\\DeltaHalo1X14.bin"};
//std::vector<std::string> trainDataFiles = {"E:\\TrainingData\\MBU0.bin", "E:\\TrainingData\\MBU1.bin"};
std::string valDataFile = "E:\\TrainingData\\DeltaHaloValidation.bin";
std::string trainDataOutFileName = "E:\\TrainingData.bin";
std::string ckptFileName = "E:\\AIGamer.ckpt";
std::string optFileName = "E:\\AIGamer.opt";
std::vector<RecordIndex> trainRecordIndices;
std::vector<RecordIndex> valRecordIndices;
ThreadPool threadPool(8);
static std::atomic<int> gLoadBatchFailureCount{0};
static std::mutex gBatchOrderMutex;
static std::vector<size_t> gTrainBatchOrder;
static std::vector<size_t> gValBatchOrder;
static size_t gTrainBatchCursor = 0;
static size_t gValBatchCursor = 0;
static std::vector<size_t> gTrainSequenceOrder;
static std::vector<size_t> gValSequenceOrder;
static std::vector<size_t> gTrainSequenceStarts;
static std::vector<size_t> gValSequenceStarts;
static size_t gTrainSequenceCursor = 0;
static size_t gValSequenceCursor = 0;
unsigned char keyMap[] = {
	0x11, // W
	0x1E, // A
	0x1F, // S
	0x20, // D
	0x39, // Space
	0x1D, // Ctrl
	0x10, // Q
	0x13, // R
	0x12, // E
	0x02, // 1
	0x03, // 2
	0x0B, // Mouse button 1
	0x0C, // Mouse button 2
	0x0D  // Mouse button 3
};
void ReportStreamState(std::ifstream& file){
	if(file.eof()){ std::cerr << "End of file reached prematurely\n"; } else if(file.fail()){ std::cerr << "Logical error on I/O operation\n"; } else if(file.bad()){ std::cerr << "Read/writing error on I/O operation\n"; } else{
		std::cerr << "Unknown error occurred\n";
	}
	std::cerr << "Current stream position: " << file.tellg() << "\n";
}
static std::ifstream& GetThreadFile(const std::string& fileName){
	using StreamPtr = std::unique_ptr<std::ifstream>;
	static thread_local std::unordered_map<std::string, StreamPtr> fileMap;
	auto it = fileMap.find(fileName);
	if(it == fileMap.end() || !it->second || !it->second->is_open()){
		auto stream = std::make_unique<std::ifstream>(fileName, std::ios::binary | std::ios::in);
		if(!stream->is_open()){
			std::cerr << "Failed to open file: " << fileName << "\n";
			throw std::runtime_error("Failed to open file");
		}
		const auto emplaceResult = fileMap.emplace(fileName, std::move(stream));
		it = emplaceResult.first;
	}
	it->second->clear();
	return *it->second;
}
static void GetBatchRecords(const std::vector<RecordIndex>* recordIndices, const bool validation, const int batchSize, std::vector<RecordIndex>* batchRecords){
	std::lock_guard<std::mutex> lock(gBatchOrderMutex);
	std::vector<size_t>& batchOrder = validation ? gValBatchOrder : gTrainBatchOrder;
	size_t& cursor = validation ? gValBatchCursor : gTrainBatchCursor;
	if(batchOrder.size() != recordIndices->size()){
		batchOrder.resize(recordIndices->size());
		std::iota(batchOrder.begin(), batchOrder.end(), 0);
	}
	for(size_t i = 0; i < static_cast<size_t>(batchSize); ++i){
		if(cursor >= batchOrder.size()){ cursor = 0; }
		(*batchRecords)[i] = (*recordIndices)[batchOrder[cursor]];
		++cursor;
	}
}
void LoadBatch(StateBatch* batch, const int batchSize, const int stateSize, const bool validation){
	const std::vector<RecordIndex>* recordIndices = validation ? &valRecordIndices : &trainRecordIndices;
	if(recordIndices->empty()){
		std::cerr << "No records available to fill the batch\n";
		return;
	}
	std::vector<RecordIndex> batchRecords(batchSize);
	GetBatchRecords(recordIndices, validation, batchSize, &batchRecords);
	for(size_t i = 0; i < static_cast<size_t>(batchSize); ++i){
		threadPool.Enqueue([i, batch, stateSize, record = batchRecords[i]]{
			try{
				auto& file = GetThreadFile(*record.fileName);
				file.seekg(record.position);
				if(file.fail()){
					std::cerr << "Failed to seek to position: " << record.position << " in file: " << *record.fileName << " (batch index " << i << ")\n";
					gLoadBatchFailureCount.fetch_add(1, std::memory_order_relaxed);
					return;
				}
				if(!file.read(reinterpret_cast<char*>(&batch->inputStates[i]), sizeof(InputState))){
					std::cerr << "Failed to read input states at index " << i << " from file: " << *record.fileName << "\n";
					gLoadBatchFailureCount.fetch_add(1, std::memory_order_relaxed);
					return;
				}
				if(!file.read(reinterpret_cast<char*>(batch->stateData + i*stateSize), stateSize)){
					std::cerr << "Failed to read stateData at index " << i << " from file: " << *record.fileName << "\n";
					gLoadBatchFailureCount.fetch_add(1, std::memory_order_relaxed);
				}
			} catch(const std::exception& e){
				std::cerr << "LoadBatch exception for file " << *record.fileName << " at batch index " << i << ": " << e.what() << "\n";
				gLoadBatchFailureCount.fetch_add(1, std::memory_order_relaxed);
			}
		});
	}
}
static void EnsureSequenceStarts(const bool validation, const int sequenceLength){
	const std::vector<RecordIndex>* recordIndices = validation ? &valRecordIndices : &trainRecordIndices;
	std::vector<size_t>& starts = validation ? gValSequenceStarts : gTrainSequenceStarts;
	if(sequenceLength <= 0){ starts.clear(); return; }
	starts.clear();
	if(recordIndices->size() < static_cast<size_t>(sequenceLength)){ return; }
	for(size_t i = 0; i + static_cast<size_t>(sequenceLength) <= recordIndices->size(); ++i){
		bool valid = true;
		const auto* fileName = (*recordIndices)[i].fileName;
		for(int step = 1; step < sequenceLength; ++step){
			if((*recordIndices)[i + static_cast<size_t>(step)].fileName != fileName){
				valid = false;
				break;
			}
		}
		if(valid){ starts.push_back(i); }
	}
}
void ShuffleSequenceOrder(const bool validation, const int sequenceLength){
	std::lock_guard<std::mutex> lock(gBatchOrderMutex);
	EnsureSequenceStarts(validation, sequenceLength);
	std::vector<size_t>& sequenceOrder = validation ? gValSequenceOrder : gTrainSequenceOrder;
	std::vector<size_t>& starts = validation ? gValSequenceStarts : gTrainSequenceStarts;
	size_t& cursor = validation ? gValSequenceCursor : gTrainSequenceCursor;
	sequenceOrder.resize(starts.size());
	std::iota(sequenceOrder.begin(), sequenceOrder.end(), 0);
	static std::mt19937 shuffleGenerator(std::random_device{}());
	std::shuffle(sequenceOrder.begin(), sequenceOrder.end(), shuffleGenerator);
	cursor = 0;
}
bool GetSequenceBatchStarts(std::vector<size_t>* starts, const int batchSize, const int sequenceLength, const bool validation){
	std::lock_guard<std::mutex> lock(gBatchOrderMutex);
	EnsureSequenceStarts(validation, sequenceLength);
	const std::vector<size_t>& sequenceOrder = validation ? gValSequenceOrder : gTrainSequenceOrder;
	const std::vector<size_t>& availableStarts = validation ? gValSequenceStarts : gTrainSequenceStarts;
	size_t& cursor = validation ? gValSequenceCursor : gTrainSequenceCursor;
	if(availableStarts.empty()){ return false; }
	if(sequenceOrder.size() != availableStarts.size()){
		starts->clear();
		return false;
	}
	starts->resize(batchSize);
	for(int i = 0; i < batchSize; ++i){
		if(cursor >= sequenceOrder.size()){ cursor = 0; }
		(*starts)[i] = availableStarts[sequenceOrder[cursor]];
		++cursor;
	}
	return true;
}
void LoadSequenceBatchStep(StateBatch* batch, const std::vector<size_t>& sequenceStarts, const int step, const int stateSize, const bool validation){
	const std::vector<RecordIndex>* recordIndices = validation ? &valRecordIndices : &trainRecordIndices;
	if(sequenceStarts.empty()){
		std::cerr << "No sequence starts available to fill the batch\n";
		return;
	}
	for(size_t i = 0; i < sequenceStarts.size(); ++i){
		const size_t recordIndex = sequenceStarts[i] + static_cast<size_t>(step);
		if(recordIndex >= recordIndices->size()){
			std::cerr << "Sequence step out of range at batch index " << i << "\n";
			gLoadBatchFailureCount.fetch_add(1, std::memory_order_relaxed);
			continue;
		}
		const auto record = (*recordIndices)[recordIndex];
		threadPool.Enqueue([i, batch, stateSize, record]{
			try{
				auto& file = GetThreadFile(*record.fileName);
				file.seekg(record.position);
				if(file.fail()){
					std::cerr << "Failed to seek to position: " << record.position << " in file: " << *record.fileName << " (seq batch index " << i << ")\n";
					gLoadBatchFailureCount.fetch_add(1, std::memory_order_relaxed);
					return;
				}
				if(!file.read(reinterpret_cast<char*>(&batch->inputStates[i]), sizeof(InputState))){
					std::cerr << "Failed to read sequence input state at index " << i << " from file: " << *record.fileName << "\n";
					gLoadBatchFailureCount.fetch_add(1, std::memory_order_relaxed);
					return;
				}
				if(!file.read(reinterpret_cast<char*>(batch->stateData + i*stateSize), stateSize)){
					std::cerr << "Failed to read sequence stateData at index " << i << " from file: " << *record.fileName << "\n";
					gLoadBatchFailureCount.fetch_add(1, std::memory_order_relaxed);
				}
			} catch(const std::exception& e){
				std::cerr << "LoadSequenceBatchStep exception for file " << *record.fileName << " at batch index " << i << ": " << e.what() << "\n";
				gLoadBatchFailureCount.fetch_add(1, std::memory_order_relaxed);
			}
		});
	}
}
void LoadBatchFromVector(const std::vector<StateSingle*>& states, StateBatch* batch, const int batchSize, const int stateSize){
	if(states.size() < batchSize){
		std::cerr << "Not enough RecordState instances to fill the batch\n";
		return;
	}
	std::uniform_int_distribution<size_t> dist(0, states.size() - 1);
	for(int i = 0; i < batchSize; ++i){
		threadPool.Enqueue([i, batch, stateSize, &states, dist]() mutable{
			const size_t randomIndex = dist(threadPool.GetThreadGenerator());
			const auto& record = states[randomIndex];
			batch->inputStates[i] = record->inputState;
			if(batch->stateData && record->stateData){
				std::memcpy(batch->stateData + i*stateSize, record->stateData, stateSize);
			} else{
				std::cerr << "Invalid stateData pointer for RecordState at index " << randomIndex << " (batch index " << i << ")\n";
				gLoadBatchFailureCount.fetch_add(1, std::memory_order_relaxed);
			}
			});
	}
}
void ResetLoadBatchFailureCount(){
	gLoadBatchFailureCount.store(0, std::memory_order_relaxed);
}
int GetLoadBatchFailureCount(){
	return gLoadBatchFailureCount.load(std::memory_order_relaxed);
}
void ShuffleBatchOrder(const bool validation){
	const std::vector<RecordIndex>* recordIndices = validation ? &valRecordIndices : &trainRecordIndices;
	std::lock_guard<std::mutex> lock(gBatchOrderMutex);
	std::vector<size_t>& batchOrder = validation ? gValBatchOrder : gTrainBatchOrder;
	size_t& cursor = validation ? gValBatchCursor : gTrainBatchCursor;
	batchOrder.resize(recordIndices->size());
	std::iota(batchOrder.begin(), batchOrder.end(), 0);
	static std::mt19937 shuffleGenerator(std::random_device{}());
	std::shuffle(batchOrder.begin(), batchOrder.end(), shuffleGenerator);
	cursor = 0;
}
