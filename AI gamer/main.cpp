#include "Train.h"
#include "Infer.h"
#include "Viewer.h"
#include "Record.h"
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <Windows.h>
#undef min
#undef max
Record* gRecord = nullptr;
Train* gTrain = nullptr;
Viewer* gViewer = nullptr;
Infer* gInfer = nullptr;
BOOL WINAPI ConsoleShutdownHandler(const DWORD ctrlType){
	switch(ctrlType){
		case CTRL_C_EVENT:
		case CTRL_BREAK_EVENT:
		case CTRL_CLOSE_EVENT:
		case CTRL_SHUTDOWN_EVENT:
		case CTRL_LOGOFF_EVENT:
			std::cerr << "\nConsole shutdown signal received. Cleaning up...\n";
			if(gInfer){
				gInfer->stop_ = true;
				gInfer->inferEnable_ = false;
			}
			if(gRecord){
				gRecord->stop_ = true;
				gRecord->recording_ = false;
			}
			if(gTrain){
				gTrain->Free();
				ExitProcess(0);
			}
			if(gViewer){ PostQuitMessage(0); }
			return TRUE;
		default:
			return FALSE;
	}
}
std::filesystem::path GetIndexCachePath(const std::filesystem::path& dataPath){
	auto cachePath = dataPath;
	cachePath += ".idxcache";
	return cachePath;
}
bool LoadIndexCache(const std::filesystem::path& cachePath, std::uintmax_t expectedFileSize, int* width, int* height, std::string* fileName, std::vector<RecordIndex>* index){
	std::ifstream cache(cachePath, std::ios::binary | std::ios::in);
	if(!cache.is_open()){ return false; }
	std::uint64_t cachedSize = 0;
	std::int32_t cachedWidth = 0;
	std::int32_t cachedHeight = 0;
	std::uint64_t recordCount = 0;
	cache.read(reinterpret_cast<char*>(&cachedSize), sizeof(cachedSize));
	cache.read(reinterpret_cast<char*>(&cachedWidth), sizeof(cachedWidth));
	cache.read(reinterpret_cast<char*>(&cachedHeight), sizeof(cachedHeight));
	cache.read(reinterpret_cast<char*>(&recordCount), sizeof(recordCount));
	if(cache.fail()){
		std::cerr << "Failed to read cache header from: " << cachePath << "\n";
		return false;
	}
	if(cachedSize != expectedFileSize){
		std::cerr << "Cache file size mismatch for: " << cachePath << " (expected " << expectedFileSize << ", cached " << cachedSize << ")\n";
		return false;
	}
	if(recordCount > std::numeric_limits<std::size_t>::max()){
		std::cerr << "Cache record count too large in: " << cachePath << "\n";
		return false;
	}
	std::vector<std::uint64_t> positions((recordCount));
	if(recordCount > 0U){
		cache.read(reinterpret_cast<char*>(positions.data()), static_cast<std::streamsize>(recordCount*sizeof(std::uint64_t)));
		if(cache.fail()){
			std::cerr << "Failed to read cache records from: " << cachePath << "\n";
			return false;
		}
	}
	if(width){ *width = cachedWidth; }
	if(height){ *height = cachedHeight; }
	index->reserve(index->size() + positions.size());
	for(const std::uint64_t posValue : positions){
		const std::streamoff offset = static_cast<std::streamoff>(posValue);
		index->push_back({fileName, std::streampos(offset)});
	}
	std::cerr << "Loaded " << recordCount << " cached records for file: " << *fileName << std::endl;
	return true;
}
void SaveIndexCache(const std::filesystem::path& cachePath, std::uintmax_t fileSize, int width, int height, const std::vector<std::uint64_t>& positions){
	std::ofstream cache(cachePath, std::ios::binary | std::ios::trunc | std::ios::out);
	if(!cache.is_open()){
		std::cerr << "Failed to write index cache file: " << cachePath << "\n";
		return;
	}
	const std::uint64_t size64 = fileSize;
	const std::int32_t width32 = width;
	const std::int32_t height32 = height;
	const std::uint64_t recordCount = positions.size();
	cache.write(reinterpret_cast<const char*>(&size64), sizeof(size64));
	cache.write(reinterpret_cast<const char*>(&width32), sizeof(width32));
	cache.write(reinterpret_cast<const char*>(&height32), sizeof(height32));
	cache.write(reinterpret_cast<const char*>(&recordCount), sizeof(recordCount));
	if(recordCount > 0U){ cache.write(reinterpret_cast<const char*>(positions.data()), static_cast<std::streamsize>(recordCount*sizeof(std::uint64_t))); }
	if(cache.fail()){
		std::cerr << "Failed to fully write index cache file: " << cachePath << "\n";
		return;
	}
	std::cerr << "Saved " << recordCount << " records to index cache: " << cachePath << std::endl;
}
void ReadStateDataFile(int* width, int* height, std::string* fileName, std::vector<RecordIndex>* index){
	const std::filesystem::path dataPath(*fileName);
	std::uintmax_t fileSize = 0;
	try{ fileSize = file_size(dataPath); } catch(const std::filesystem::filesystem_error& e){
		std::cerr << "Failed to get size of training data file: " << *fileName << " (" << e.what() << ")\n";
		return;
	}
	std::cerr << "File: " << *fileName << " Size: " << fileSize << " bytes" << std::endl;
	const auto cachePath = GetIndexCachePath(dataPath);
	if(LoadIndexCache(cachePath, fileSize, width, height, fileName, index)){
		if(width && height){
			const auto stateSize = static_cast<std::uintmax_t>(*width)*static_cast<std::uintmax_t>(*height)*3U;
			std::cerr << "State size calculated: " << stateSize << " bytes (from cache)" << std::endl;
		}
		return;
	}
	std::ifstream file(*fileName, std::ios::binary | std::ios::in);
	if(!file.is_open()){
		std::cerr << "Failed to open training data file: " << *fileName << std::endl;
		return;
	}
	file.read(reinterpret_cast<char*>(width), sizeof*width);
	file.read(reinterpret_cast<char*>(height), sizeof*height);
	if(file.fail() || file.eof()){
		std::cerr << "Failed to read width/height from file: " << *fileName << "\n";
		return;
	}
	const auto stateSize = static_cast<std::uintmax_t>(*width)*static_cast<std::uintmax_t>(*height)*3U;
	std::cerr << "State size calculated: " << stateSize << " bytes" << std::endl;
	const auto recordSize = static_cast<std::uintmax_t>(sizeof(InputState)) + stateSize;
	std::vector<std::uint64_t> recordPositions;
	int fileRecordsCount = 0;
	bool encounteredReadError = false;
	while(true){
		const std::streampos pos = file.tellg();
		if(pos == std::streampos(-1)){
			std::cerr << "Failed to determine position in file: " << *fileName << "\n";
			encounteredReadError = true;
			break;
		}
		const std::streamoff offset = pos;
		if(offset < 0){
			std::cerr << "Encountered negative offset in file: " << *fileName << " at position: " << offset << "\n";
			encounteredReadError = true;
			break;
		}
		const auto posValue = static_cast<std::uintmax_t>(offset);
		if(posValue > fileSize){
			std::cerr << "Invalid record position in file: " << *fileName << " at position: " << pos << "\n";
			encounteredReadError = true;
			break;
		}
		const auto bytesRemaining = fileSize - posValue;
		if(bytesRemaining < recordSize){
			std::cerr << "Not enough bytes remaining for a full record in file: " << *fileName << " at position: " << pos << " (Remaining: " << bytesRemaining << " bytes)\n";
			break;
		}
		index->push_back({fileName, pos});
		recordPositions.push_back(static_cast<std::uint64_t>(posValue));
		++fileRecordsCount;
		file.seekg(static_cast<std::streamoff>(recordSize), std::ios::cur);
		if(file.fail()){
			std::cerr << "Failed to seek to next record in file: " << *fileName << " at position: " << pos << "\n";
			encounteredReadError = true;
			break;
		}
		const std::streampos nextPos = file.tellg();
		if(nextPos == std::streampos(-1)){
			std::cerr << "Failed to determine next position after seeking in file: " << *fileName << "\n";
			encounteredReadError = true;
			break;
		}
		const std::streamoff nextOffsetOff = nextPos;
		if(nextOffsetOff < 0){
			std::cerr << "Encountered negative offset after seeking in file: " << *fileName << "\n";
			encounteredReadError = true;
			break;
		}
		const auto nextOffset = static_cast<std::uintmax_t>(nextOffsetOff);
		if(nextOffset >= fileSize || file.peek() == std::char_traits<char>::eof()){
			std::cerr << "Reached EOF or invalid position after seeking to next record in file: " << *fileName << " at position: " << pos << "\n";
			break;
		}
	}
	std::cerr << "Total records found: " << fileRecordsCount << " in file: " << *fileName << std::endl;
	file.close();
	if(!encounteredReadError){ SaveIndexCache(cachePath, fileSize, *width, *height, recordPositions); }
}
void ReadStateData(int* width, int* height){
	for(std::string& fileName : trainDataFiles){ ReadStateDataFile(width, height, &fileName, &trainRecordIndices); }
	ReadStateDataFile(width, height, &valDataFile, &valRecordIndices);
	RebuildSequenceIndices();
}
int main(){
	std::ios::sync_with_stdio(false);
	std::cout << std::fixed << std::setprecision(6);
	SetConsoleCtrlHandler(ConsoleShutdownHandler, TRUE);
	std::cout << "R for Record mode, T for Train mode, V for View mode, I for Infer mode... ";
	char mode;
	std::cin >> mode;
	std::cout << "\n";
	if(mode == 'r' || mode == 'R'){
		gRecord = new Record();
		gRecord->Run();
		delete gRecord;
		gRecord = nullptr;
	} else if(mode == 't' || mode == 'T'){
		int width = 0, height = 0;
		ReadStateData(&width, &height);
		std::cout << "Training data resolution: " << width << "x" << height << "\n";
		gTrain = new Train();
		gTrain->TrainModel(width, height);
		delete gTrain;
		gTrain = nullptr;
	} else if(mode == 'v' || mode == 'V'){
		std::cout << "Training data file: ";
		std::string fileName;
		std::cin >> fileName;
		gViewer = new Viewer();
		gViewer->Play(fileName);
		delete gViewer;
		gViewer = nullptr;
	} else if(mode == 'i' || mode == 'I'){
		gInfer = new Infer();
		gInfer->Run();
		delete gInfer;
		gInfer = nullptr;
	}
	return 0;
}