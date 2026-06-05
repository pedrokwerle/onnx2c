/* This file is part of onnx2c.
 *
 * sauria_manifest.cc
 * Implementation of SAURIA manifest loader
 * 
 * Uses a simple JSON parser - if onnx2c already uses nlohmann/json,
 * this can be replaced with that.
 */

#include "sauria_manifest.h"
#include "error.h"
#include "aixlog.hpp"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>

namespace toC {

// Simple JSON tokenizer/parser for manifest format
// (A full JSON library would be better, but this keeps dependencies minimal)

class SimpleJsonParser {
public:
    SimpleJsonParser(const std::string& json) : data(json), pos(0) {}
    
    bool parse(SauriaManifest& manifest, 
               std::string& model_name, 
               std::string& hardware_config,
               std::map<std::string, SauriaConvInfo>& convolutions) 
    {
        skipWhitespace();
        if (!expect('{')) return false;
        
        while (pos < data.size()) {
            skipWhitespace();
            if (peek() == '}') { pos++; break; }
            
            std::string key = parseString();
            skipWhitespace();
            if (!expect(':')) return false;
            skipWhitespace();
            
            if (key == "model") {
                parseString(); // skip model path
            }
            else if (key == "model_name") {
                model_name = parseString();
            }
            else if (key == "hardware") {
                hardware_config = parseString();
            }
            else if (key == "convolutions") {
                if (!parseConvolutions(convolutions)) return false;
            }
            else {
                skipValue(); // skip unknown keys
            }
            
            skipWhitespace();
            if (peek() == ',') pos++;
        }
        
        return true;
    }

private:
    std::string data;
    size_t pos;
    
    char peek() { return pos < data.size() ? data[pos] : '\0'; }
    
    void skipWhitespace() {
        while (pos < data.size() && std::isspace(data[pos])) pos++;
    }
    
    bool expect(char c) {
        skipWhitespace();
        if (peek() == c) { pos++; return true; }
        return false;
    }
    
    std::string parseString() {
        skipWhitespace();
        if (peek() != '"') return "";
        pos++; // skip opening quote
        
        std::string result;
        while (pos < data.size() && data[pos] != '"') {
            if (data[pos] == '\\' && pos + 1 < data.size()) {
                pos++;
                switch (data[pos]) {
                    case 'n': result += '\n'; break;
                    case 't': result += '\t'; break;
                    case '"': result += '"'; break;
                    case '\\': result += '\\'; break;
                    default: result += data[pos]; break;
                }
            } else {
                result += data[pos];
            }
            pos++;
        }
        if (peek() == '"') pos++; // skip closing quote
        return result;
    }
    
    int parseInt() {
        skipWhitespace();
        int result = 0;
        bool negative = false;
        if (peek() == '-') { negative = true; pos++; }
        while (pos < data.size() && std::isdigit(data[pos])) {
            result = result * 10 + (data[pos] - '0');
            pos++;
        }
        return negative ? -result : result;
    }
    
    bool parseBool() {
        skipWhitespace();
        if (data.substr(pos, 4) == "true") { pos += 4; return true; }
        if (data.substr(pos, 5) == "false") { pos += 5; return false; }
        return false;
    }
    
    bool parseNull() {
        skipWhitespace();
        if (data.substr(pos, 4) == "null") { pos += 4; return true; }
        return false;
    }
    
    std::vector<int> parseIntArray() {
        std::vector<int> result;
        skipWhitespace();
        if (!expect('[')) return result;
        
        while (pos < data.size()) {
            skipWhitespace();
            if (peek() == ']') { pos++; break; }
            result.push_back(parseInt());
            skipWhitespace();
            if (peek() == ',') pos++;
        }
        return result;
    }
    
    void skipValue() {
        skipWhitespace();
        char c = peek();
        if (c == '"') {
            parseString();
        } else if (c == '{') {
            int depth = 1;
            pos++;
            while (pos < data.size() && depth > 0) {
                if (data[pos] == '{') depth++;
                else if (data[pos] == '}') depth--;
                else if (data[pos] == '"') parseString();
                else pos++;
            }
        } else if (c == '[') {
            int depth = 1;
            pos++;
            while (pos < data.size() && depth > 0) {
                if (data[pos] == '[') depth++;
                else if (data[pos] == ']') depth--;
                else if (data[pos] == '"') parseString();
                else pos++;
            }
        } else if (c == 't' || c == 'f') {
            parseBool();
        } else if (c == 'n') {
            parseNull();
        } else {
            while (pos < data.size() && data[pos] != ',' && data[pos] != '}' && data[pos] != ']') {
                pos++;
            }
        }
    }
    
    bool parseConvolutions(std::map<std::string, SauriaConvInfo>& convolutions) {
        if (!expect('{')) return false;
        
        while (pos < data.size()) {
            skipWhitespace();
            if (peek() == '}') { pos++; break; }
            
            std::string conv_name = parseString();
            skipWhitespace();
            if (!expect(':')) return false;
            
            SauriaConvInfo info;
            if (!parseConvInfo(info)) return false;
            convolutions[conv_name] = info;
            
            skipWhitespace();
            if (peek() == ',') pos++;
        }
        
        return true;
    }
    
    bool parseConvInfo(SauriaConvInfo& info) {
        if (!expect('{')) return false;
        
        info.is_depthwise = false;
        info.num_channels = -1;
        
        while (pos < data.size()) {
            skipWhitespace();
            if (peek() == '}') { pos++; break; }
            
            std::string key = parseString();
            skipWhitespace();
            if (!expect(':')) return false;
            skipWhitespace();
            if (key == "function_name") {
                info.function_name = parseString();
            }
            else if (key == "weight_var") {
                info.weight_var = parseString();
            }
            else if (key == "bias_var") {
                if (parseNull()) {
                    info.bias_var = "";
                } else {
                    info.bias_var = parseString();
                }
            }
            else if (key == "bias_broadcast_var") {
                if (parseNull()) {
                    info.bias_broadcast_var = "";
                } else {
                    info.bias_broadcast_var = parseString();
                }
            }
            else if (key == "input_tensor") {
                info.input_tensor = parseString();
            }
            else if (key == "output_tensor") {
                info.output_tensor = parseString();
            }
            else if (key == "is_depthwise") {
                info.is_depthwise = parseBool();
            }
            else if (key == "num_channels") {
                if (parseNull()) {
                    info.num_channels = -1;
                } else {
                    info.num_channels = parseInt();
                }
            }
            else if (key == "tiling") {
                if (!parseTiling(info)) return false;
            }
            else {
                skipValue();
            }
            
            skipWhitespace();
            if (peek() == ',') pos++;
        }
        
        return true;
    }
    
    bool parseTiling(SauriaConvInfo& info) {
        if (!expect('{')) return false;
        
        while (pos < data.size()) {
            skipWhitespace();
            if (peek() == '}') { pos++; break; }
            
            std::string key = parseString();
            skipWhitespace();
            if (!expect(':')) return false;
            skipWhitespace();
            
            if (key == "C_tile_shape") {
                info.C_tile_shape = parseIntArray();
            }
            else if (key == "tile_cin") {
                info.tile_cin = parseInt();
            }
            else if (key == "X_used") {
                info.X_used = parseInt();
            }
            else if (key == "Y_used") {
                info.Y_used = parseInt();
            }
            else {
                skipValue();
            }
            
            skipWhitespace();
            if (peek() == ',') pos++;
        }
        
        return true;
    }
};


// SauriaManifest implementation

SauriaManifest& SauriaManifest::instance() {
    static SauriaManifest inst;
    return inst;
}

bool SauriaManifest::load(const std::string& filepath) {
    LOG(DEBUG) << "Loading SAURIA manifest from: " << filepath << std::endl;
    
    std::ifstream file(filepath);
    if (!file.is_open()) {
        LOG(ERROR) << "Failed to open SAURIA manifest: " << filepath << std::endl;
        return false;
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string json = buffer.str();
    
    SimpleJsonParser parser(json);
    if (!parser.parse(*this, model_name, hardware_config, convolutions)) {
        LOG(ERROR) << "Failed to parse SAURIA manifest JSON" << std::endl;
        return false;
    }
    
    loaded = true;
    LOG(INFO) << "Loaded SAURIA manifest: model=" << model_name 
              << ", hardware=" << hardware_config 
              << ", convolutions=" << convolutions.size() << std::endl;
    
    return true;
}

const SauriaConvInfo* SauriaManifest::getConvInfo(const std::string& node_name) const {
    // Try exact match first
    auto it = convolutions.find(node_name);
    if (it != convolutions.end()) {
        return &it->second;
    }
    
    // Try sanitized name match (replace invalid chars with underscore)
    std::string sanitized = node_name;
    for (char& c : sanitized) {
        if (!std::isalnum(c) && c != '_') {
            c = '_';
        }
    }
    
    it = convolutions.find(sanitized);
    if (it != convolutions.end()) {
        return &it->second;
    }
    
    return nullptr;
}

} // namespace toC
