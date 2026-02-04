#include "common.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <sstream>
std::vector<std::string> trainDataFiles = {"E:\\TrainingData\\DeltaHalo0X11.bin", "E:\\TrainingData\\DeltaHalo1X14.bin"};
std::string valDataFile = "E:\\TrainingData\\DeltaHaloValidation.bin";
std::string trainDataOutFileName = "E:\\TrainingData.bin";
std::string ckptFileName = "E:\\AIGamer.ckpt";
std::string optFileName = "E:\\AIGamer.opt";
std::vector<RecordIndex> trainRecordIndices;
std::vector<RecordIndex> valRecordIndices;
ThreadPool threadPool(8);
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
void LoadBatch(StateBatch* batch, const int batchSize, const int stateSize, const bool validation){
	const std::vector<RecordIndex>* recordIndices = validation ? &valRecordIndices : &trainRecordIndices;
	if(recordIndices->size() < batchSize){
		std::cerr << "Not enough records to fill the batch\n";
		return;
	}
	for(size_t i = 0; i < batchSize; ++i){
		threadPool.Enqueue([i, batch, stateSize, recordIndices]{
			const std::uniform_int_distribution<size_t> dist(0, recordIndices->size() - 1);
			const size_t randomIndex = dist(threadPool.GetThreadGenerator());
			const RecordIndex record = (*recordIndices)[randomIndex];
			try{
				auto& file = GetThreadFile(*record.fileName);
				file.seekg(record.position);
				if(file.fail()){
					std::cerr << "Failed to seek to position: " << record.position << " in file: " << *record.fileName << "\n";
					return;
				}
				if(!file.read(reinterpret_cast<char*>(&batch->inputStates[i]), sizeof(InputState))){
					std::cerr << "Failed to read input states at index " << i << " from file: " << *record.fileName << "\n";
					return;
				}
				if(!file.read(reinterpret_cast<char*>(batch->stateData + i*stateSize), stateSize)){ std::cerr << "Failed to read stateData at index " << i << " from file: " << *record.fileName << "\n"; }
			} catch(const std::exception&){}
		});
	}
}
void LoadBatchFromVector(const std::vector<StateSingle*>& states, StateBatch* batch, const int batchSize, const int stateSize){
	if(states.size() < batchSize){
		std::cerr << "Not enough RecordState instances to fill the batch\n";
		return;
	}
	std::uniform_int_distribution<size_t> dist(0, states.size() - 1);
	for(size_t i = 0; i < batchSize; ++i){
		threadPool.Enqueue([batch, stateSize, &states, dist]() mutable{
			const size_t randomIndex = dist(threadPool.GetThreadGenerator());
			const auto& record = states[randomIndex];
			batch->inputStates[1] = record->inputState;
			if(batch->stateData && record->stateData){ std::memcpy(batch->stateData + 1*stateSize, record->stateData, stateSize); } else{ std::cerr << "Invalid stateData pointer for RecordState at index " << randomIndex << "\n"; }
		});
	}
}