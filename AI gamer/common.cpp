#include "common.h"
#include <fstream>
#include <algorithm>
#include <atomic>
#include <mutex>
#include <random>
#include <unordered_map>
#include <iostream>
#include <numeric>
#include <vector>
#include <sstream>
//std::vector<std::string> trainDataFiles = {"E:\\TrainingData\\DeltaHalo0X11.bin", "E:\\TrainingData\\DeltaHalo1X14.bin"};
std::vector<std::string> trainDataFiles = {"E:\\TrainingData\\MBU0.bin", "E:\\TrainingData\\MBU1.bin"};
std::string valDataFile = "E:\\TrainingData\\DeltaHaloValidation.bin";
std::string trainDataOutFileName = "E:\\TrainingData.bin";
std::string ckptFileName = "E:\\AIGamer.ckpt";
std::string optFileName = "E:\\AIGamer.opt";
std::vector<RecordIndex> trainRecordIndices;
std::vector<RecordIndex> valRecordIndices;
std::vector<SequenceRecordIndex> trainSequenceRecordIndices;
std::vector<SequenceRecordIndex> valSequenceRecordIndices;
SequenceSamplingConfig sequenceSamplingConfig{};
ThreadPool threadPool(8);
static std::atomic<int> gLoadBatchFailureCount{0};
static std::mutex gBatchOrderMutex;
static std::vector<size_t> gTrainBatchOrder;
static std::vector<size_t> gValBatchOrder;
static std::vector<size_t> gTrainSequenceBatchOrder;
static std::vector<size_t> gValSequenceBatchOrder;
static size_t gTrainBatchCursor = 0;
static size_t gValBatchCursor = 0;
static size_t gTrainSequenceBatchCursor = 0;
static size_t gValSequenceBatchCursor = 0;
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
static void BuildSequenceIndicesFromFrames(const std::vector<RecordIndex>& frames, std::vector<SequenceRecordIndex>* out){
	out->clear();
	const int sequenceLength = std::max(sequenceSamplingConfig.length, 1);
	const int sequenceStride = std::max(sequenceSamplingConfig.stride, 1);
	if(sequenceLength <= 1){
		out->reserve(frames.size());
		for(const auto& frame : frames){ out->push_back({frame.fileName, frame.position}); }
		return;
	}
	if(frames.empty()) return;
	size_t segmentStart = 0;
	while(segmentStart < frames.size()){
		size_t segmentEnd = segmentStart + 1;
		while(segmentEnd < frames.size() && frames[segmentEnd].fileName == frames[segmentStart].fileName){ ++segmentEnd; }
		const size_t segmentSize = segmentEnd - segmentStart;
		const size_t safeLength = static_cast<size_t>(sequenceLength);
		const size_t safeStride = static_cast<size_t>(sequenceStride);
		const size_t requiredSpan = 1 + (safeLength - 1)*safeStride;
		if(segmentSize >= requiredSpan){
			const size_t maxStart = segmentSize - requiredSpan;
			for(size_t localStart = 0; localStart <= maxStart; ++localStart){
				out->push_back({frames[segmentStart + localStart].fileName, frames[segmentStart + localStart].position});
			}
		}
		segmentStart = segmentEnd;
	}
}
void RebuildSequenceIndices(){
	BuildSequenceIndicesFromFrames(trainRecordIndices, &trainSequenceRecordIndices);
	BuildSequenceIndicesFromFrames(valRecordIndices, &valSequenceRecordIndices);
	std::lock_guard<std::mutex> lock(gBatchOrderMutex);
	gTrainSequenceBatchOrder.clear();
	gValSequenceBatchOrder.clear();
	gTrainSequenceBatchCursor = 0;
	gValSequenceBatchCursor = 0;
}
void ConfigureSequenceSampling(const int length, const int stride, const int targetOffset){
	const int safeLength = std::max(length, 1);
	const int safeStride = std::max(stride, 1);
	const int maxTargetOffset = safeLength - 1;
	sequenceSamplingConfig.length = safeLength;
	sequenceSamplingConfig.stride = safeStride;
	sequenceSamplingConfig.targetOffset = std::clamp(targetOffset, 0, maxTargetOffset);
	RebuildSequenceIndices();
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
static void GetBatchSequenceRecords(const std::vector<SequenceRecordIndex>* recordIndices, const bool validation, const int batchSize, std::vector<SequenceRecordIndex>* batchRecords){
	std::lock_guard<std::mutex> lock(gBatchOrderMutex);
	std::vector<size_t>& batchOrder = validation ? gValSequenceBatchOrder : gTrainSequenceBatchOrder;
	size_t& cursor = validation ? gValSequenceBatchCursor : gTrainSequenceBatchCursor;
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
void LoadBatchSequence(StateBatchSequence* batch, const int batchSize, const int stateSize, const bool validation){
	const std::vector<SequenceRecordIndex>* recordIndices = validation ? &valSequenceRecordIndices : &trainSequenceRecordIndices;
	if(recordIndices->empty()){
		std::cerr << "No sequence records available to fill the batch\n";
		return;
	}
	const int sequenceLength = std::max(sequenceSamplingConfig.length, 1);
	const int sequenceStride = std::max(sequenceSamplingConfig.stride, 1);
	const int targetOffset = std::clamp(sequenceSamplingConfig.targetOffset, 0, sequenceLength - 1);
	const size_t recordSize = sizeof(InputState) + static_cast<size_t>(stateSize);
	std::vector<SequenceRecordIndex> batchRecords(batchSize);
	GetBatchSequenceRecords(recordIndices, validation, batchSize, &batchRecords);
	for(size_t batchIndex = 0; batchIndex < static_cast<size_t>(batchSize); ++batchIndex){
		threadPool.Enqueue([batch, batchIndex, stateSize, sequenceLength, sequenceStride, targetOffset, recordSize, record = batchRecords[batchIndex]]{
			try{
				auto& file = GetThreadFile(*record.fileName);
				for(int step = 0; step < sequenceLength; ++step){
					const std::streamoff frameOffset = static_cast<std::streamoff>(recordSize*static_cast<size_t>(step*sequenceStride));
					const std::streampos framePos = record.startPosition + frameOffset;
					const auto inputIndex = static_cast<size_t>(batchIndex)*sequenceLength + static_cast<size_t>(step);
					file.seekg(framePos);
					if(file.fail()){
						std::cerr << "Failed to seek to sequence frame position " << framePos << " in file: " << *record.fileName << " (batch index " << batchIndex << ", step " << step << ")\n";
						gLoadBatchFailureCount.fetch_add(1, std::memory_order_relaxed);
						return;
					}
					if(!file.read(reinterpret_cast<char*>(&batch->sequenceInputStates[inputIndex]), sizeof(InputState))){
						std::cerr << "Failed to read sequence input state at batch index " << batchIndex << ", step " << step << " from file: " << *record.fileName << "\n";
						gLoadBatchFailureCount.fetch_add(1, std::memory_order_relaxed);
						return;
					}
					const size_t stateOffset = (static_cast<size_t>(batchIndex)*sequenceLength + static_cast<size_t>(step))*static_cast<size_t>(stateSize);
					if(!file.read(reinterpret_cast<char*>(batch->sequenceStateData + stateOffset), stateSize)){
						std::cerr << "Failed to read sequence state data at batch index " << batchIndex << ", step " << step << " from file: " << *record.fileName << "\n";
						gLoadBatchFailureCount.fetch_add(1, std::memory_order_relaxed);
						return;
					}
				}
				const auto targetIndex = static_cast<size_t>(batchIndex)*sequenceLength + static_cast<size_t>(targetOffset);
				batch->targetInputStates[batchIndex] = batch->sequenceInputStates[targetIndex];
			} catch(const std::exception& e){
				std::cerr << "LoadBatchSequence exception for file " << *record.fileName << " at batch index " << batchIndex << ": " << e.what() << "\n";
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
	const std::vector<SequenceRecordIndex>* sequenceIndices = validation ? &valSequenceRecordIndices : &trainSequenceRecordIndices;
	std::lock_guard<std::mutex> lock(gBatchOrderMutex);
	std::vector<size_t>& batchOrder = validation ? gValBatchOrder : gTrainBatchOrder;
	size_t& cursor = validation ? gValBatchCursor : gTrainBatchCursor;
	batchOrder.resize(recordIndices->size());
	std::iota(batchOrder.begin(), batchOrder.end(), 0);
	std::vector<size_t>& sequenceBatchOrder = validation ? gValSequenceBatchOrder : gTrainSequenceBatchOrder;
	size_t& sequenceCursor = validation ? gValSequenceBatchCursor : gTrainSequenceBatchCursor;
	sequenceBatchOrder.resize(sequenceIndices->size());
	std::iota(sequenceBatchOrder.begin(), sequenceBatchOrder.end(), 0);
	static std::mt19937 shuffleGenerator(std::random_device{}());
	std::shuffle(batchOrder.begin(), batchOrder.end(), shuffleGenerator);
	std::shuffle(sequenceBatchOrder.begin(), sequenceBatchOrder.end(), shuffleGenerator);
	cursor = 0;
	sequenceCursor = 0;
}
