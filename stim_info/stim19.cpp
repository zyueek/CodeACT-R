#include <iostream>
#include <vector>
int afunc(const std::vector<int>& numbers){
  int output = 0;
  for (int num : numbers){
    if(num % 2 == 0 && num % 4 == 0){
      output += num * num * num;
    }
  }
  return output;
}
