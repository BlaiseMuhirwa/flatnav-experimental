#pragma once

#include <cstdint>

namespace flatnav::util {

/**
 * @brief Enum class for data types
 * Currently, only float32 is supported for index building.
 */
enum class DataType {
  uint8,
  uint16,
  uint32,
  uint64,
  int8,
  int16,
  int32,
  int64,
  float16,
  float32,
  float64,
  undefined
};

/**
 * @brief Get a string representation of the data type
 */
inline constexpr std::string_view name(DataType data_type) {
  switch (data_type) {
  case DataType::uint8:
    return "uint8";
  case DataType::uint16:
    return "uint16";
  case DataType::uint32:
    return "uint32";
  case DataType::uint64:
    return "uint64";
  case DataType::int8:
    return "int8";
  case DataType::int16:
    return "int16";
  case DataType::int32:
    return "int32";
  case DataType::int64:
    return "int64";
  case DataType::float16:
    return "float16";
  case DataType::float32:
    return "float32";
  case DataType::float64:
    return "float64";
  default:
    return "undefined";
  }
}

/**
 * @brief Get the data type from a string representation
 */
inline constexpr DataType type(const std::string_view &data_type) {
  if (data_type == "uint8") {
    return DataType::uint8;
  } else if (data_type == "uint16") {
    return DataType::uint16;
  } else if (data_type == "uint32") {
    return DataType::uint32;
  } else if (data_type == "uint64") {
    return DataType::uint64;
  } else if (data_type == "int8") {
    return DataType::int8;
  } else if (data_type == "int16") {
    return DataType::int16;
  } else if (data_type == "int32") {
    return DataType::int32;
  } else if (data_type == "int64") {
    return DataType::int64;
  } else if (data_type == "float16") {
    return DataType::float16;
  } else if (data_type == "float32") {
    return DataType::float32;
  } else if (data_type == "float64") {
    return DataType::float64;
  } else {
    return DataType::undefined;
  }
}

/**
 * @brief Get the size of the data type in bytes
 */
inline constexpr size_t size(DataType data_type) {
  switch (data_type) {
  case DataType::uint8:
    return sizeof(uint8_t);
  case DataType::uint16:
    return sizeof(uint16_t);
  case DataType::uint32:
    return sizeof(uint32_t);
  case DataType::uint64:
    return sizeof(uint64_t);
  case DataType::int8:
    return sizeof(int8_t);
  case DataType::int16:
    return sizeof(int16_t);
  case DataType::int32:
    return sizeof(int32_t);
  case DataType::int64:
    return sizeof(int64_t);
  case DataType::float16:
    return sizeof(float) / 2;
  case DataType::float32:
    return sizeof(float);
  case DataType::float64:
    return sizeof(double);
  default:
    return 0;
  }
}

} // namespace flatnav::util