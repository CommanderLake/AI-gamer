#pragma once
#define WINDOWS_LEAN_AND_MEAN
#include <map>
#include <windows.h>
#include "NvFBC\nvFBC.h"
#include <string>
#define NVFBC64_LIBRARY_NAME "NvFBC64.dll"
#define NVFBC_LIBRARY_NAME "NvFBC.dll"
// Wraps loading and using NvFBC
class NvFBCLibrary{
    NvFBCLibrary(const NvFBCLibrary&);
    NvFBCLibrary& operator=(const NvFBCLibrary&);
	const int magic[4] = {0x0D7BC620, 0x4C17E142, 0x5E6B5997, 0x4B5A855B};
public:
    NvFBCLibrary() : fnIsWow64Process(nullptr), m_handle(nullptr), pfn_get_status(nullptr), pfn_set_global_flags(nullptr), pfn_create(nullptr), pfn_enable(nullptr){}
    ~NvFBCLibrary(){ if(nullptr != m_handle) close(); }
	static std::string NVFBCResultToString(NVFBCRESULT result);
	// Attempts to load NvFBC from system directory.
    // on 32-bit OS: looks for NvFBC.dll in system32
    // for 32-bit app on 64-bit OS: looks for NvFBC.dll in syswow64
    // for 64-bit app on 64-bit OS: looks for NvFBC64.dll in system32
    void load(std::string fileName = std::string()){
        if(m_handle != nullptr) return;
        if(!fileName.empty()) m_handle = LoadLibraryA(fileName.c_str());
        if(m_handle == nullptr){
	        m_handle = LoadLibraryA(getDefaultPath().c_str());
        }
        if(m_handle == nullptr){
			throw std::exception("Unable to load NvFBC.");
        }
        // Load the three functions exported by NvFBC
        pfn_create = reinterpret_cast<NvFBC_CreateFunctionExType>(GetProcAddress(m_handle, "NvFBC_CreateEx"));
        pfn_set_global_flags = reinterpret_cast<NvFBC_SetGlobalFlagsType>(GetProcAddress(m_handle, "NvFBC_SetGlobalFlags"));
        pfn_get_status = reinterpret_cast<NvFBC_GetStatusExFunctionType>(GetProcAddress(m_handle, "NvFBC_GetStatusEx"));
        pfn_enable = reinterpret_cast<NvFBC_EnableFunctionType>(GetProcAddress(m_handle, "NvFBC_Enable"));
        if(pfn_create == nullptr || pfn_set_global_flags == nullptr || pfn_get_status == nullptr || pfn_enable == nullptr){
            close();
			throw std::exception("Unable to load the NvFBC function pointers.");
        }
    }
    // Close the NvFBC dll
    void close(){
        if(nullptr != m_handle) FreeLibrary(m_handle);
        m_handle = nullptr;
        pfn_create = nullptr;
        pfn_get_status = nullptr;
        pfn_enable = nullptr;
    }
    // Get the status for the provided adapter, if no adapter is 
    // provided the default adapter is used.
    NVFBCRESULT getStatus(NvFBCStatusEx* status) const{ return pfn_get_status(static_cast<void*>(status)); }
    // Sets the global flags for the provided adapter, if 
    // no adapter is provided the default adapter is used
    void setGlobalFlags(DWORD flags, int adapter = 0) const{
        setTargetAdapter(adapter);
        pfn_set_global_flags(flags);
    }
    // Creates an instance of the provided NvFBC type if possible
    NVFBCRESULT createEx(NvFBCCreateParams* pParams) const{ return pfn_create(static_cast<void *>(pParams)); }
    // Creates an instance of the provided NvFBC type if possible.  
    void* create(DWORD type, DWORD* maxWidth, DWORD* maxHeight, int adapter = 0, void* devicePtr = nullptr) const{
        if(nullptr == m_handle) return nullptr;
        NvFBCStatusEx status = {0};
        status.dwVersion = NVFBC_STATUS_VER;
        status.dwAdapterIdx = adapter;
        const auto res = getStatus(&status);
        if(res != NVFBC_SUCCESS){
			throw std::runtime_error("NVFBC CUDA setup failed, result:\n\n" + NVFBCResultToString(res));
        }
        // Check to see if the device and driver are supported
        if(!status.bIsCapturePossible){
			throw std::exception("NvFBC Unsupported device or driver.");
        }
        // Check to see if an instance can be created
        if(!status.bCanCreateNow){
			throw std::exception("Unable to create an instance of NvFBC.");
        }
        NvFBCCreateParams createParams = {0};
		createParams.dwVersion = NVFBC_CREATE_PARAMS_VER;
        createParams.dwInterfaceType = type;
        createParams.pDevice = devicePtr;
        createParams.dwAdapterIdx = adapter;
		createParams.pPrivateData = (void*)&magic;
		createParams.dwPrivateDataSize = sizeof magic;
        pfn_create(&createParams);
        *maxWidth = createParams.dwMaxDisplayWidth;
        *maxHeight = createParams.dwMaxDisplayHeight;
        return createParams.pNvFBC;
    }
    // enable/disable NVFBC
    void enable(NVFBC_STATE nvFBCState) const{
        const auto res = pfn_enable(nvFBCState);
        if(res != NVFBC_SUCCESS){
			if(nvFBCState == 0) throw std::exception("Failed to disable NVFBC.");
        	throw std::exception("Failed to enable NVFBC.");
        }
    }
protected:
    // Get the default NvFBC library path
    typedef BOOL (WINAPI *pfnIsWow64Process)(HANDLE, PBOOL);
    pfnIsWow64Process fnIsWow64Process;
    BOOL IsWow64(){
        auto bIsWow64 = FALSE;
		const auto handle = GetModuleHandle(TEXT("kernel32.dll"));
        if(handle) fnIsWow64Process = reinterpret_cast<pfnIsWow64Process>(GetProcAddress(handle, "IsWow64Process"));
        if(fnIsWow64Process != nullptr){ if(!fnIsWow64Process(GetCurrentProcess(), &bIsWow64)){ bIsWow64 = false; } }
        return bIsWow64;
    }
	static std::string getDefaultPath(){
        std::string defaultPath;
        size_t pathSize;
        char* libPath;
        if(_dupenv_s(&libPath, &pathSize, "SystemRoot") != 0){
			throw std::exception("Unable to get the SystemRoot environment variable.");
        }
        if(pathSize == 0){
			throw std::exception("The SystemRoot environment variable is not set.");
        }
		if(libPath != nullptr){
#ifdef _WIN64
			defaultPath = std::string(libPath) + "\\System32\\" + NVFBC64_LIBRARY_NAME;
#else
			if(IsWow64()){ defaultPath = std::string(libPath) + "\\Syswow64\\" + NVFBC_LIBRARY_NAME; } else{
				defaultPath = std::string(libPath) + "\\System32\\" + NVFBC_LIBRARY_NAME;
			}
#endif
		}
        return defaultPath;
    }
    void setTargetAdapter(int adapter = 0) const{
        char targetAdapter[10] = {0};
        _snprintf_s(targetAdapter, 10, 9, "%d", adapter);
        SetEnvironmentVariableA("NVFBC_TARGET_ADAPTER", targetAdapter);
    }
    HMODULE m_handle;
    NvFBC_GetStatusExFunctionType pfn_get_status;
    NvFBC_SetGlobalFlagsType pfn_set_global_flags;
    NvFBC_CreateFunctionExType pfn_create;
    NvFBC_EnableFunctionType pfn_enable;
};
inline std::string NvFBCLibrary::NVFBCResultToString(NVFBCRESULT result){
	static const std::map<NVFBCRESULT, std::string> errorStrings = {
		{NVFBC_SUCCESS, "NVFBC_SUCCESS"},
		{NVFBC_ERROR_GENERIC, "NVFBC_ERROR_GENERIC"},
		{NVFBC_ERROR_INVALID_PARAM, "NVFBC_ERROR_INVALID_PARAM"},
		{NVFBC_ERROR_INVALIDATED_SESSION, "NVFBC_ERROR_INVALIDATED_SESSION"},
		{NVFBC_ERROR_PROTECTED_CONTENT, "NVFBC_ERROR_PROTECTED_CONTENT"},
		{NVFBC_ERROR_DRIVER_FAILURE, "NVFBC_ERROR_DRIVER_FAILURE"},
		{NVFBC_ERROR_CUDA_FAILURE, "NVFBC_ERROR_CUDA_FAILURE"},
		{NVFBC_ERROR_UNSUPPORTED, "NVFBC_ERROR_UNSUPPORTED"},
		{NVFBC_ERROR_HW_ENC_FAILURE, "NVFBC_ERROR_HW_ENC_FAILURE"},
		{NVFBC_ERROR_INCOMPATIBLE_DRIVER, "NVFBC_ERROR_INCOMPATIBLE_DRIVER"},
		{NVFBC_ERROR_UNSUPPORTED_PLATFORM, "NVFBC_ERROR_UNSUPPORTED_PLATFORM"},
		{NVFBC_ERROR_OUT_OF_MEMORY, "NVFBC_ERROR_OUT_OF_MEMORY"},
		{NVFBC_ERROR_INVALID_PTR, "NVFBC_ERROR_INVALID_PTR"},
		{NVFBC_ERROR_INCOMPATIBLE_VERSION, "NVFBC_ERROR_INCOMPATIBLE_VERSION"},
		{NVFBC_ERROR_OPT_CAPTURE_FAILURE, "NVFBC_ERROR_OPT_CAPTURE_FAILURE"},
		{NVFBC_ERROR_INSUFFICIENT_PRIVILEGES, "NVFBC_ERROR_INSUFFICIENT_PRIVILEGES"},
		{NVFBC_ERROR_INVALID_CALL, "NVFBC_ERROR_INVALID_CALL"},
		{NVFBC_ERROR_SYSTEM_ERROR, "NVFBC_ERROR_SYSTEM_ERROR"},
		{NVFBC_ERROR_INVALID_TARGET, "NVFBC_ERROR_INVALID_TARGET"},
		{NVFBC_ERROR_NVAPI_FAILURE, "NVFBC_ERROR_NVAPI_FAILURE"},
		{NVFBC_ERROR_DYNAMIC_DISABLE, "NVFBC_ERROR_DYNAMIC_DISABLE"},
		{NVFBC_ERROR_IPC_FAILURE, "NVFBC_ERROR_IPC_FAILURE"},
		{NVFBC_ERROR_CURSOR_CAPTURE_FAILURE, "NVFBC_ERROR_CURSOR_CAPTURE_FAILURE"}
	};
	const auto it = errorStrings.find(result);
	if(it != errorStrings.end()){ return it->second; }
	return "Unknown error";
}
