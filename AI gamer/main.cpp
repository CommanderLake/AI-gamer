#include "Train.h"
#include "Infer.h"
#include "Viewer.h"
#include "Record.h"
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <Windows.h>
void ReadStateData(int* width, int* height, int* stateSize){
	for(const auto& fileName : trainingDataFiles){
		std::ifstream file(fileName, std::ios::binary|std::ios::in);
		if(!file.is_open()){
			std::cerr<<"Failed to open training data file: "<<fileName<<std::endl;
			continue;
		}
		// Get the file size using std::filesystem
		const std::uintmax_t fileSize = std::filesystem::file_size(fileName);
		std::cerr<<"File: "<<fileName<<" Size: "<<fileSize<<" bytes"<<std::endl;
		// Read width and height
		file.read(reinterpret_cast<char*>(width), sizeof(*width));
		file.read(reinterpret_cast<char*>(height), sizeof(*height));
		if(file.fail()||file.eof()){
			std::cerr<<"Failed to read width/height from file: "<<fileName<<"\r\n";
			continue;
		}
		*stateSize = (*width)*(*height)*3;
		std::cerr<<"State size calculated: "<<*stateSize<<" bytes"<<std::endl;
		int fileRecordsCount = 0;
		while(true){
			std::streampos pos = file.tellg();
			std::streampos bytesRemaining = fileSize-pos;
			// Check if there are enough bytes left in the file for a full record
			if(bytesRemaining<(10+*stateSize)){
				std::cerr<<"Not enough bytes remaining for a full record in file: "<<fileName<<" at position: "<<pos<<" (Remaining: "<<bytesRemaining<<" bytes)\r\n";
				break;
			}
			// Store the record index
			recordIndices.push_back({&fileName, pos});
			++fileRecordsCount;
			// Move to the next record
			file.seekg(10+*stateSize, std::ios::cur);
			if(file.fail()){
				std::cerr<<"Failed to seek to next record in file: "<<fileName<<" at position: "<<pos<<"\r\n";
				break;
			}
			// Ensure that after seeking, we are not past the end of the file
			if(file.peek()==EOF||file.tellg()>fileSize){
				std::cerr<<"Reached EOF or invalid position after seeking to next record in file: "<<fileName<<" at position: "<<pos<<"\r\n";
				break;
			}
		}
		std::cerr<<"Total records found: "<<fileRecordsCount<<" in file: "<<fileName<<std::endl;
		file.close();
	}
}
int main(){
	SetEnvironmentVariableA("CUDNN_LOGDEST_DBG", "E:\\cudnn_debug_log.txt");
	SetEnvironmentVariableA("CUDNN_LOGLEVEL_DBG", "3");
	std::ios::sync_with_stdio(false);
	std::cout << std::fixed << std::setprecision(8);
	std::cout << "R for Record mode, T for Train mode, V for View mode, I for Infer mode, F for Fine tune infer mode... ";
	char mode;
	std::cin >> mode;
	std::cout << "\r\n";
	if(mode == 'r' || mode == 'R'){
		const auto recorder = new Record();
		recorder->Run();
	} else if(mode == 't' || mode == 'T'){
		int width, height, stateSize;
		ReadStateData(&width, &height, &stateSize);
		const auto train = new Train();
		train->TrainModel(width, height);
		delete train;
	} else if(mode == 'v' || mode == 'V'){
		std::cout << "Training data file: ";
		std::string fileName;
		std::cin >> fileName;
		const auto viewer = new Viewer();
		viewer->Play(fileName);
		delete viewer;
	} else if(mode == 'i' || mode == 'I'){
		const auto infer = new Infer(false);
		infer->Run();
		delete infer;
	} else if(mode == 'f' || mode == 'F'){
		const auto infer = new Infer(true);
		infer->Run();
		delete infer;
	}
	return 0;
}