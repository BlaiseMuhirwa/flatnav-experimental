
# Format all header files with clang-format 
# TODO: Use a recursive find solution to format headers/src files
find flatnav -iname *.h -o -iname *.cpp | xargs clang-format -i 
find tools -iname *.cpp -o -iname *.h | xargs clang-format -i 
find flatnav_python -iname *.cpp | xargs clang-format -i
find quantization -iname *.h -o -iname *.cpp | xargs clang-format -i 
find quantization/tests -iname *.h -o -iname *.cpp | xargs clang-format -i 


# Format CMakeLists.txt file 
cmake-format -i CMakeLists.txt 